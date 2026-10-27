"""
Section 2.4 analysis: the five main figures, the statistics, and a short
text report, all computed from the event logs.

    python3 scripts/analyze.py                  # reads logs/, writes results/
    python3 scripts/analyze.py --logs logs --out results

Needs: numpy, matplotlib  (pip3 install --user matplotlib)

OUTPUT
  results/figures/fig1_latency_savings.png/.pdf   tau vs latency savings      (RQ1, RQ3)
  results/figures/fig2_overhead.png/.pdf          tau vs proactive overhead   (RQ1, RQ3)
  results/figures/fig3_pareto.png/.pdf            savings vs overhead         (RQ1, RQ2)
  results/figures/fig4_calibration.png/.pdf       accuracy + calibration      (RQ2, RQ3)
  results/figures/fig5_h4_mechanism.png/.pdf      the H4 causal chain, live   (H4)
  results/tables/per_run.csv                      every run, every metric
  results/tables/tau_curves.csv                   medians + IQRs behind fig 1-3
  results/tables/paired_vs_reference.csv          H1/H3 paired tests per tau
  results/tables/rq2_lstm_vs_markov.csv           RQ2 paired tests + 95% CIs
  results/tables/calibration.csv                  accuracy, ECE, Brier per scenario
  results/tables/h4_mechanism.csv                 numbers behind fig 5
  results/tables/drift_halves.csv                 drift before / after the switch
  results/analysis_report.md                      the key numbers, in words

DEFINITIONS (fixed here, used everywhere)
  Unit of analysis: one run. Seeds are matched across strategies, so every
  comparison is a PAIRED per-seed difference (Section 2.4).

  Latency savings (%) = 100 * (L_ref - L) / L_ref, per seed, where L is the
  run's mean flow-setup latency and L_ref is the reference run with the same seed:
    offline scenarios: the reactive run (latency from the replay's fixed model).
    live table_pressure: the SAME strategy at tau = 1.0, which installs (almost)
      nothing and so behaves like reactive. This removes a measured confound:
      controllers that load PyTorch (the LSTM arm) run every packet faster,
      even when they install nothing, so LSTM latency cannot be compared with
      the reactive or Markov controller directly. Hit rate and controller
      interactions are unaffected and ARE compared against reactive.

  Overhead = proactive installs per 1,000 flows.
  Prediction accuracy / calibration use EVERY prediction (tau = 0.0 runs,
  where every prediction is logged and installed). Offline, a prediction on
  row i is for flow i; live, it is for flow i+1.
  ECE: 10 equal-width confidence bins. Brier: mean (confidence - correct)^2.
  Precision (share of proactive installs that were right) is left blank for a
  run with fewer than 10 proactive installs.
  Per-run latency = MEAN flow-setup latency (the median would just be the hit
  latency, since ~90% of flows hit).

  H1 criterion (Section 2.4): median savings > 0 AND larger than the IQR of
  the reference run's latency across seeds (expressed in %), for that tau.
  Wilcoxon signed-rank: exact, two-sided, zero differences dropped.
  Holm-Bonferroni: across the 11 tau values within each scenario x strategy.
  Effect size: matched-pairs rank-biserial r (-1..1).
"""
from __future__ import annotations
import argparse
import csv
import glob
import itertools
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.event_log import read_events_list  # noqa: E402

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("This script needs numpy and matplotlib:  pip3 install --user matplotlib")

SCENARIOS = ["baseline", "cold_start", "drift", "drift_after", "unstructured", "table_pressure"]
TITLES = {"baseline": "Baseline", "cold_start": "Cold-start", "drift": "Drift (whole run)",
          "drift_after": "Drift (after the switch)", "unstructured": "Unstructured",
          "table_pressure": "Table pressure (live)"}
ARMS = ["markov", "lstm"]
COLORS = {"markov": "#2a78d6", "lstm": "#eb6834", "reactive": "#8a8986"}
LABELS = {"markov": "Markov", "lstm": "LSTM", "reactive": "Reactive"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- statistics

def wilcoxon_exact(d):
    """Exact two-sided Wilcoxon signed-rank p-value and rank-biserial r."""
    d = [x for x in d if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0.0
    a = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(a[j + 1][0] - a[i][0]) < 1e-12:
            j += 1
        for k in range(i, j + 1):
            ranks[a[k][1]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    total = sum(ranks)
    mean = total / 2
    obs = abs(w_plus - mean)
    count = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        if abs(w - mean) >= obs - 1e-9:
            count += 1
    p = count / 2 ** n
    r = (w_plus - (total - w_plus)) / total
    return p, r


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def ci95_mean(d, n_boot=10000, seed=7):
    """Percentile bootstrap 95% CI of the mean paired difference (fixed seed)."""
    if len(d) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(d)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def q(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


# ---------------------------------------------------------------- per-run metrics

def metrics(events, mode, part=None):
    events = sorted(events, key=lambda e: e.flow_id)
    if part == "second_half":
        events = events[len(events) // 2:]
    n = len(events)
    lat = [e.flow_setup_latency_ms for e in events if e.flow_setup_latency_ms is not None]
    pro = [i for i, e in enumerate(events) if e.installed_proactively]
    shift = 1 if mode == "live" else 0
    correct = sum(1 for i in pro if i + shift < n and events[i].predicted_signature == events[i + shift].signature)
    occ = [e.table_occupancy_after for e in events if e.table_occupancy_after is not None]
    return {
        "n_flows": n,
        "hit_rate": sum(e.match_type == "hit" for e in events) / n,
        "controller_interactions": sum(bool(e.controller_interaction) for e in events),
        "installs_per_1000": 1000 * len(pro) / n,
        "proactive_installs": len(pro),
        # precision from fewer than 10 installs is noise; left blank
        "precision": correct / len(pro) if len(pro) >= 10 else float("nan"),
        "evictions": sum(e.n_evictions for e in events),
        "harmful_evictions": sum(bool(e.harmful_eviction) for e in events),
        "mean_occupancy": st.mean(occ) if occ else float("nan"),
        "latency_ms": st.mean(lat) if lat else float("nan"),
    }


def calibration_pairs(events, mode):
    """(confidence, correct) for every prediction in a run."""
    events = sorted(events, key=lambda e: e.flow_id)
    shift = 1 if mode == "live" else 0
    out = []
    for i, e in enumerate(events):
        if e.confidence is None or i + shift >= len(events):
            continue
        if mode == "offline" and i == 0:
            continue  # no history yet
        out.append((float(e.confidence), e.predicted_signature == events[i + shift].signature))
    return out


def ece_brier(pairs, bins=10):
    if not pairs:
        return float("nan"), float("nan"), [], [], []
    conf = np.array([c for c, _ in pairs])
    corr = np.array([1.0 if k else 0.0 for _, k in pairs])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece, centers, accs, counts = 0.0, [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.sum() / len(conf) * abs(corr[m].mean() - conf[m].mean())
        centers.append(conf[m].mean())
        accs.append(corr[m].mean())
        counts.append(int(m.sum()))
    brier = float(np.mean((conf - corr) ** 2))
    return float(ece), brier, centers, accs, counts


# ---------------------------------------------------------------- figure helpers

def style_axes(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)


def min_span(ax, span):
    """Never let a near-flat panel blow noise up to fill the axis."""
    lo, hi = ax.get_ylim()
    if hi - lo < span:
        mid = (lo + hi) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)


def share_offline_y(axes_list, pad=1.0):
    """Same y-range on every offline panel, so shrinkage across scenarios is visible."""
    los, his = zip(*(ax.get_ylim() for ax in axes_list))
    for ax in axes_list:
        ax.set_ylim(min(los) - pad, max(his) + pad)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def band_plot(ax, curves, key, arms=ARMS):
    """Median line + IQR band per arm over tau. curves[(arm, tau)] -> list of values."""
    for arm in arms:
        taus = sorted(t for (a, t) in curves if a == arm)
        if not taus:
            continue
        med = [np.median(curves[(arm, t)]) for t in taus]
        lo = [q(curves[(arm, t)], 25) for t in taus]
        hi = [q(curves[(arm, t)], 75) for t in taus]
        ax.fill_between(taus, lo, hi, color=COLORS[arm], alpha=0.15, linewidth=0)
        ax.plot(taus, med, color=COLORS[arm], linewidth=2, marker="o", markersize=3.5, label=LABELS[arm])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    fig_dir, tab_dir = os.path.join(args.out, "figures"), os.path.join(args.out, "tables")
    os.makedirs(tab_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.logs, "offline", "*", "*.jsonl")) +
                   glob.glob(os.path.join(args.logs, "live", "*.jsonl")))
    if not files:
        sys.exit(f"No logs under {args.logs}/ -- run this from the project folder.")

    # runs[(scenario, strategy, tau, seed)] -> metrics ; calib[(scenario, strategy)] -> pairs
    runs, calib = {}, defaultdict(list)
    print(f"Reading {len(files)} logs ...")
    for f in files:
        ev = read_events_list(f)
        if not ev:
            continue
        e0 = ev[0]
        mode = "live" if e0.scenario == "table_pressure" else "offline"
        key_tau = round(e0.tau, 1)
        runs[(e0.scenario, e0.strategy, key_tau, e0.seed)] = metrics(ev, mode)
        if e0.scenario == "drift":
            runs[("drift_after", e0.strategy, key_tau, e0.seed)] = metrics(ev, mode, "second_half")
        if e0.strategy in ARMS and key_tau == 0.0:
            calib[(e0.scenario, e0.strategy)] += calibration_pairs(ev, mode)
            if e0.scenario == "drift":
                half = sorted(ev, key=lambda e: e.flow_id)
                calib[("drift_after", e0.strategy)] += calibration_pairs(half[len(half) // 2:], mode)

    seeds = sorted({k[3] for k in runs})
    taus = sorted({k[2] for k in runs if k[1] in ARMS})

    # ---- latency savings per run (paired with its reference run) ----
    def ref_key(scen, arm, seed):
        return (scen, arm, 1.0, seed) if scen == "table_pressure" else (scen, "reactive", 1.0, seed)

    for (scen, strat, tau, seed), m in runs.items():
        r = runs.get(ref_key(scen, strat, seed)) if strat in ARMS else runs.get((scen, "reactive", 1.0, seed))
        m["savings_pct"] = (100 * (r["latency_ms"] - m["latency_ms"]) / r["latency_ms"]) if r else float("nan")
        react = runs.get((scen, "reactive", 1.0, seed))
        m["miss_reduction_pct"] = (100 * ((1 - react["hit_rate"]) - (1 - m["hit_rate"])) / (1 - react["hit_rate"])
                                   if react and react["hit_rate"] < 1 else float("nan"))

    with open(os.path.join(tab_dir, "per_run.csv"), "w", newline="") as fh:
        cols = ["scenario", "strategy", "tau", "seed"] + list(next(iter(runs.values())).keys())
        w = csv.writer(fh)
        w.writerow(cols)
        for k in sorted(runs):
            w.writerow(list(k) + [round(v, 5) if isinstance(v, float) else v for v in runs[k].values()])

    def series(scen, arm, tau, key):
        return [runs[(scen, arm, tau, s)][key] for s in seeds if (scen, arm, tau, s) in runs]

    # ---- tau curves table ----
    curve_keys = ["savings_pct", "miss_reduction_pct", "installs_per_1000", "evictions", "harmful_evictions",
                  "controller_interactions", "hit_rate", "precision", "latency_ms"]
    with open(os.path.join(tab_dir, "tau_curves.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "strategy", "tau", "n_seeds"] + [f"{k}_{s}" for k in curve_keys for s in ("median", "q1", "q3")])
        for scen in SCENARIOS:
            for strat in ARMS + ["reactive"]:
                for tau in (taus if strat in ARMS else [1.0]):
                    vals = {k: [v for v in series(scen, strat, tau, k) if not math.isnan(v)] for k in curve_keys}
                    if not vals["hit_rate"]:
                        continue
                    row = [scen, strat, tau, len(vals["hit_rate"])]
                    for k in curve_keys:
                        row += [round(np.median(vals[k]), 4) if vals[k] else "", round(q(vals[k], 25), 4) if vals[k] else "",
                                round(q(vals[k], 75), 4) if vals[k] else ""]
                    w.writerow(row)

    # ---- H1 / H3: paired savings vs reference, per tau ----
    paired_rows = []
    for scen in SCENARIOS:
        for arm in ARMS:
            block = []
            for tau in taus:
                d = series(scen, arm, tau, "savings_pct")
                if not d:
                    continue
                refs = [runs[ref_key(scen, arm, s)]["latency_ms"] for s in seeds if ref_key(scen, arm, s) in runs]
                ref_iqr_pct = 100 * (q(refs, 75) - q(refs, 25)) / np.median(refs) if refs else float("nan")
                p, r = wilcoxon_exact(d)
                block.append({"scenario": scen, "strategy": arm, "tau": tau, "n": len(d),
                              "median_savings_pct": round(float(np.median(d)), 3),
                              "q1": round(q(d, 25), 3), "q3": round(q(d, 75), 3),
                              "reference_iqr_pct": round(ref_iqr_pct, 3),
                              "h1_criterion_met": bool(np.median(d) > 0 and np.median(d) > ref_iqr_pct),
                              "median_miss_reduction_pct": round(float(np.nanmedian(series(scen, arm, tau, "miss_reduction_pct"))), 3),
                              "wilcoxon_p": round(p, 4), "rank_biserial_r": round(r, 3)})
            for row, padj in zip(block, holm([b["wilcoxon_p"] for b in block])):
                row["p_holm"] = round(padj, 4)
            paired_rows += block
    with open(os.path.join(tab_dir, "paired_vs_reference.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(paired_rows[0].keys()))
        w.writeheader()
        w.writerows(paired_rows)

    # ---- RQ2: LSTM - Markov, paired per tau ----
    rq2 = []
    for scen in SCENARIOS:
        block = []
        for tau in taus:
            for metric in ("savings_pct", "hit_rate", "installs_per_1000"):
                pairs = [(runs[(scen, "lstm", tau, s)][metric], runs[(scen, "markov", tau, s)][metric])
                         for s in seeds if (scen, "lstm", tau, s) in runs and (scen, "markov", tau, s) in runs]
                if not pairs:
                    continue
                d = [a - b for a, b in pairs]
                if metric == "hit_rate":
                    d = [100 * x for x in d]
                p, r = wilcoxon_exact(d)
                lo, hi = ci95_mean(d)
                sd = st.stdev(d) if len(d) > 1 else float("nan")
                block.append({"scenario": scen, "tau": tau, "metric": metric + (" (points)" if metric == "hit_rate" else ""),
                              "n": len(d), "mean_diff": round(st.mean(d), 3), "ci95_low": round(lo, 3),
                              "ci95_high": round(hi, 3), "median_diff": round(float(np.median(d)), 3),
                              "cohens_dz": round(st.mean(d) / sd, 3) if sd and sd > 0 else "",
                              "wilcoxon_p": round(p, 4), "rank_biserial_r": round(r, 3)})
        for metric in {b["metric"] for b in block}:
            sub = [b for b in block if b["metric"] == metric]
            for b, padj in zip(sub, holm([b["wilcoxon_p"] for b in sub])):
                b["p_holm"] = round(padj, 4)
        rq2 += block
    with open(os.path.join(tab_dir, "rq2_lstm_vs_markov.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rq2[0].keys()))
        w.writeheader()
        w.writerows(rq2)

    # ---- calibration ----
    cal_rows, cal_curves = [], {}
    for scen in SCENARIOS:
        for arm in ARMS:
            pairs = calib.get((scen, arm), [])
            if not pairs:
                continue
            ece, brier, centers, accs, counts = ece_brier(pairs)
            cal_curves[(scen, arm)] = (centers, accs, counts)
            cal_rows.append({"scenario": scen, "strategy": arm, "n_predictions": len(pairs),
                             "accuracy": round(st.mean(1.0 if k else 0.0 for _, k in pairs), 4),
                             "mean_confidence": round(st.mean(c for c, _ in pairs), 4),
                             "ece": round(ece, 4), "brier": round(brier, 4)})
    with open(os.path.join(tab_dir, "calibration.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)

    # ---- H4 mechanism table (live) ----
    h4_keys = ["installs_per_1000", "mean_occupancy", "evictions", "harmful_evictions",
               "controller_interactions", "hit_rate", "precision", "savings_pct", "latency_ms"]
    h4 = []
    for strat in ARMS + ["reactive"]:
        for tau in (taus if strat in ARMS else [1.0]):
            row = {"strategy": strat, "tau": tau}
            for k in h4_keys:
                v = [x for x in series("table_pressure", strat, tau, k) if not math.isnan(x)]
                row[k] = round(float(np.median(v)), 4) if v else ""
            if any(row[k] != "" for k in h4_keys):
                h4.append(row)
    with open(os.path.join(tab_dir, "h4_mechanism.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h4[0].keys()))
        w.writeheader()
        w.writerows(h4)

    # ---- drift halves ----
    with open(os.path.join(tab_dir, "drift_halves.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "tau", "part", "median_hit_rate", "median_miss_reduction_pct",
                    "median_installs_per_1000", "median_precision"])
        for strat in ARMS + ["reactive"]:
            for tau in (taus if strat in ARMS else [1.0]):
                for scen, part in (("drift", "whole run"), ("drift_after", "after the switch")):
                    hr = series(scen, strat, tau, "hit_rate")
                    if hr:
                        w.writerow([strat, tau, part, round(float(np.median(hr)), 4),
                                    round(float(np.nanmedian(series(scen, strat, tau, "miss_reduction_pct"))), 2)
                                    if strat in ARMS else "",
                                    round(float(np.median(series(scen, strat, tau, "installs_per_1000"))), 1),
                                    round(float(np.nanmedian(series(scen, strat, tau, "precision"))), 4)
                                    if strat in ARMS and not all(math.isnan(x) for x in series(scen, strat, tau, "precision")) else ""])

    # ================================================================ FIGURES
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK,
                         "axes.labelcolor": INK2, "figure.facecolor": "white"})

    def curves_for(scen, key):
        return {(arm, tau): series(scen, arm, tau, key) for arm in ARMS for tau in taus
                if series(scen, arm, tau, key)}

    def grid_fig(key, ylabel, name, title, zero_line=True, ref=None, span=5, note="", own_scale=True):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
        for ax, scen in zip(axes.flat, SCENARIOS):
            style_axes(ax, TITLES[scen])
            if zero_line:
                ax.axhline(0, color=INK2, linewidth=0.8)
            if ref:
                ref(ax, scen)
            band_plot(ax, curves_for(scen, key), key)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        offline_axes = [ax for ax, scen in zip(axes.flat, SCENARIOS) if scen != "table_pressure"]
        share_offline_y(offline_axes, pad=span * 0.05)
        for ax, scen in zip(axes.flat, SCENARIOS):
            min_span(ax, span)
            if scen == "table_pressure" and own_scale:
                ax.set_title(TITLES[scen] + "  (own scale)", fontsize=9.5, color=INK, loc="left", pad=6)
        for ax in axes[1]:
            ax.set_xlabel("confidence threshold τ")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        h, l_ = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l_, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.0))
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
        fig.text(0.01, -0.02, "Lines: median over seeds; bands: interquartile range. " + note,
                 fontsize=7.5, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, fig_dir, name)

    grid_fig("savings_pct", "latency savings (%)", "fig1_latency_savings",
             "Figure 1. Latency savings fall as τ rises, and vanish under unstructured traffic and after drift", span=5,
             note="Offline panels: the replay's fixed latency model, reference = the reactive run with the same seed. "
                  "Live panel: measured latency, reference = the same strategy at \u03c4 = 1.0 (see analyze.py).")

    def react_overhead(ax, scen):
        pass
    grid_fig("installs_per_1000", "proactive installs per 1,000 flows", "fig2_overhead",
             "Figure 2. Proactive overhead: LSTM's confidence gate closes much earlier than Markov's", span=100,
             note="Overhead = proactive rule installs per 1,000 arriving flows.", own_scale=False)

    # fig 3: pareto
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for ax, scen in zip(axes.flat, SCENARIOS):
        style_axes(ax, TITLES[scen])
        ax.axhline(0, color=INK2, linewidth=0.8)
        for arm in ARMS:
            pts = [(np.median(series(scen, arm, t, "installs_per_1000")), np.median(series(scen, arm, t, "savings_pct")), t)
                   for t in taus if series(scen, arm, t, "savings_pct")]
            if not pts:
                continue
            xs, ys, ts = zip(*pts)
            ax.plot(xs, ys, color=COLORS[arm], linewidth=2, marker="o", markersize=4, label=LABELS[arm])
            for x, y, t in pts:
                if t in (0.0, 0.5):
                    ax.annotate(f"τ={t:.1f}", (x, y), textcoords="offset points",
                                xytext=(-26, 5) if arm == "markov" else (4, -11), fontsize=7, color=COLORS[arm])
        ax.set_xlabel("proactive installs per 1,000 flows (overhead)")
        min_span(ax, 5)
    for ax in axes[:, 0]:
        ax.set_ylabel("latency savings (%)")
    h, l_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper right", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 3. Savings vs overhead: each point is one τ (median over seeds); up and to the left is better",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, fig_dir, "fig3_pareto")

    # fig 4: calibration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    style_axes(a1, "Reliability on baseline traffic (all predictions)")
    a1.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle="--", label="perfect calibration")
    for arm in ARMS:
        if ("baseline", arm) in cal_curves:
            c, a, _ = cal_curves[("baseline", arm)]
            e = next(r["ece"] for r in cal_rows if r["scenario"] == "baseline" and r["strategy"] == arm)
            a1.plot(c, a, color=COLORS[arm], linewidth=2, marker="o", markersize=4, label=f"{LABELS[arm]} (ECE {e:.3f})")
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_xlabel("mean confidence in bin")
    a1.set_ylabel("share of predictions correct")
    a1.legend(frameon=False, fontsize=8, loc="upper left")

    style_axes(a2, "Mean confidence (filled) vs accuracy (hollow), per scenario")
    scen_c = [s for s in SCENARIOS if any((s, a) in cal_curves for a in ARMS)]
    y = np.arange(len(scen_c))
    for j, arm in enumerate(ARMS):
        off = -0.16 if j == 0 else 0.16
        for i, scen in enumerate(scen_c):
            row = next((r for r in cal_rows if r["scenario"] == scen and r["strategy"] == arm), None)
            if not row:
                continue
            a2.plot([row["accuracy"], row["mean_confidence"]], [i + off] * 2, color=COLORS[arm], linewidth=2, alpha=0.6)
            a2.plot(row["mean_confidence"], i + off, "o", color=COLORS[arm], markersize=7,
                    label=LABELS[arm] + " confidence" if i == 0 else None)
            a2.plot(row["accuracy"], i + off, "o", markerfacecolor="white", markeredgecolor=COLORS[arm],
                    markeredgewidth=1.6, markersize=7, label=LABELS[arm] + " accuracy" if i == 0 else None)
    a2.set_yticks(y)
    a2.set_yticklabels([TITLES[s] for s in scen_c])
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.set_xlabel("probability")
    a2.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.13))
    fig.suptitle("Figure 4. Prediction accuracy and calibration, measured independently of installation (τ = 0.0 runs)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.06, "Right panel: filled dot right of hollow = over-confident, left of hollow = under-confident. A prediction is correct "
             "if the predicted flow is the next to arrive (offline: same row; live: next row).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save(fig, fig_dir, "fig4_calibration")

    # fig 5: H4 chain (live)
    panels = [("installs_per_1000", "1. proactive installs / 1,000 flows"),
              ("evictions", "2. evictions per run"),
              ("harmful_evictions", "3. harmful evictions per run"),
              ("controller_interactions", "4. controller interactions (misses) per run"),
              ("precision", "5. prediction precision (installs that were right)"),
              ("savings_pct", "6. latency savings vs own τ = 1.0 (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        style_axes(ax, title)
        react = [x for x in series("table_pressure", "reactive", 1.0, key) if not math.isnan(x)]
        if react and key not in ("precision", "savings_pct", "installs_per_1000"):
            ax.axhline(np.median(react), color=COLORS["reactive"], linewidth=1.5, linestyle="--", label="Reactive (median)")
        if key == "savings_pct":
            ax.axhline(0, color=INK2, linewidth=0.8)
        band_plot(ax, curves_for("table_pressure", key), key)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if key == "precision":
            ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("confidence threshold τ")
    h, l_ = axes.flat[1].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 5. The H4 chain under table pressure (live, capacity 80): read panels 1 → 6",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, fig_dir, "fig5_h4_mechanism")

    # ================================================================ REPORT
    def pr(scen, arm, tau):
        return next((r for r in paired_rows if r["scenario"] == scen and r["strategy"] == arm and r["tau"] == tau), None)

    lines = ["# Analysis report (generated by scripts/analyze.py)", "",
             f"Logs: {len(files)} | seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]}) | tau grid: {taus[0]}-{taus[-1]}", ""]
    lines += ["## H1 / RQ1: best tau per scenario and strategy", "",
              "| Scenario | Strategy | Best tau | Median latency savings | Median misses removed | Wilcoxon p (Holm) | H1 criterion met at how many tau |",
              "|---|---|---|---|---|---|---|"]
    for scen in SCENARIOS:
        for arm in ARMS:
            rows = [r for r in paired_rows if r["scenario"] == scen and r["strategy"] == arm]
            if not rows:
                continue
            best = max(rows, key=lambda r: r["median_savings_pct"])
            lines.append(f"| {TITLES[scen]} | {LABELS[arm]} | {best['tau']:.1f} | {best['median_savings_pct']:.1f}% | "
                         f"{best['median_miss_reduction_pct']:.1f}% | {best['wilcoxon_p']:.4f} ({best['p_holm']:.4f}) | "
                         f"{sum(r['h1_criterion_met'] for r in rows)} of {len(rows)} |")
    lines += ["", "## RQ2: LSTM minus Markov (latency savings, percentage points)", "",
              "| Scenario | tau | Mean diff | 95% CI | Wilcoxon p (Holm) |", "|---|---|---|---|---|"]
    for scen in SCENARIOS:
        for tau in (0.0, 0.3, 0.5, 0.7):
            r = next((b for b in rq2 if b["scenario"] == scen and b["tau"] == tau and b["metric"] == "savings_pct"), None)
            if r:
                lines.append(f"| {TITLES[scen]} | {tau:.1f} | {r['mean_diff']:+.2f} | [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}] | "
                             f"{r['wilcoxon_p']:.4f} ({r.get('p_holm', float('nan')):.4f}) |")
    lines += ["", "## Calibration (all predictions, tau = 0.0 runs)", "",
              "| Scenario | Strategy | Accuracy | Mean confidence | ECE | Brier |", "|---|---|---|---|---|---|"]
    for r in cal_rows:
        lines.append(f"| {TITLES[r['scenario']]} | {LABELS[r['strategy']]} | {r['accuracy']:.3f} | {r['mean_confidence']:.3f} | "
                     f"{r['ece']:.3f} | {r['brier']:.3f} |")
    lines += ["", "## H4: table pressure (live), medians per run", "",
              "| Strategy | tau | Installs/1000 | Evictions | Harmful | Controller interactions | Hit rate | Savings vs own tau=1.0 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in h4:
        if r["tau"] in (0.0, 0.3, 0.5, 0.7, 1.0):
            lines.append(f"| {LABELS[r['strategy']]} | {r['tau']:.1f} | {r['installs_per_1000']} | {r['evictions']} | "
                         f"{r['harmful_evictions']} | {r['controller_interactions']} | {100 * r['hit_rate']:.1f}% | {r['savings_pct']} |")
    lines += ["", "Figures are in results/figures/, every number above is in results/tables/."]
    with open(os.path.join(args.out, "analysis_report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote 5 figures to {fig_dir}/, 7 tables to {tab_dir}/, and {args.out}/analysis_report.md")


if __name__ == "__main__":
    main()
