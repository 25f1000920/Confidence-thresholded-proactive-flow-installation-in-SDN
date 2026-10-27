# Confidence-Thresholded Proactive Flow Installation in SDN

Code, configuration and analysis for the KTH II2202 (Autumn 2026) project
*Confidence-Thresholded Proactive Flow Installation in Software-Defined Networks: A Tradeoff and Robustness Study*
by Abirami Nachammai Natarajan and Tanmay Jayanthi.

An SDN controller can install a flow rule **before** the flow arrives if it can predict the flow, which avoids a
table miss and a round trip to the controller. Every proactive rule also takes space in a finite flow table, and
it can evict a rule that is still needed. This project installs a predicted rule only when the predictor's
confidence is at least a threshold **τ**. It then measures the latency benefit and the table cost across:

- **3 strategies:** reactive-only (no prediction), a first-order **Markov** predictor and a small **LSTM**.
- **11 thresholds:** τ = 0.0, 0.1, …, 1.0, for the two predictors.
- **5 traffic scenarios:** baseline, cold-start, drift, unstructured and table pressure.
- **8 matched seeds:** 1000–1007. Every strategy sees identical traffic, so all comparisons are paired.

That gives **920 runs** of 1,500 flows each:

- **736 offline-replay runs:** four scenarios, with modelled latency.
- **184 live runs:** Mininet + Open vSwitch + Ryu, with measured end-to-end latency, under table pressure.

## Main findings

- **Prediction helps when traffic is learnable.** With learnable traffic, Markov cuts flow-setup latency by up to
  **23 %** offline and **17 %** live. The saving is proportional to the number of correct new rules.
- **The LSTM is not better here.** It saves **less than Markov at every τ** offline and shows no measurable live
  saving. The generated traffic is itself first-order Markov, so this holds for this configuration and this
  traffic only.
- **Robustness:**
  - The advantage shrinks under cold-start traffic and vanishes after drift and under unstructured traffic.
  - τ curbs wasteful installs only for the LSTM, whose confidence tracks its accuracy. Markov stays confident while
    wrong.
- **Table pressure:** harmful evictions cancel about **40 %** of the gross miss reduction, but do not reverse the
  latency benefit.
- **Replay vs live:** offline replay overstates the gain by about 1.4 hit-rate points because of install races.

The full report is `report/final_report.tex`.

---

## Repository layout

```
controller/ryu_app.py        Ryu app: reactive rules, tau-gated proactive installs, LRU eviction, per-flow event log
predictors/                  markov.py (first-order, abstains on unseen states), lstm.py (PyTorch), base.py (shared interface)
policy/threshold_policy.py   the single tau-gated installation layer shared by both predictors
table_sim/flow_table.py      flow-table simulator (capacity, idle timeout, LRU, harmful-eviction bookkeeping)
traffic/generators.py        the five traffic scenarios (fixed "world" graphs, per-seed random walks)
topo/mininet_topo.py         one OVS switch, 16 hosts
shared/                      config.py (all fixed parameters), event-log schema, live-log merging, train/eval splits
scripts/
  run_offline_matrix.py      offline replay sweep (baseline, cold_start, drift, unstructured)
  run_live_matrix.py         live Mininet/Ryu sweep (table_pressure); started via run_live.sh
  replay.py                  replays one logged run through the table simulator
  validate_replay.py         replay-vs-live agreement check
  spot_check.py              hand-traceable spot checks of logged flows
  summarize.py               one-line-per-run summary table
  analyze.py                 statistics (paired Wilcoxon, Holm, d_z, bootstrap CIs), calibration (ECE, Brier), tables
  paper_figures.py           calibration and H4 figures (Figures 3-4 of the report)
  combined_figures.py        tau-curve and saving-vs-installs figures, all scenarios per graph (Figures 1-2)
  smoke_test.py              fast end-to-end check, no Mininet needed
  setup_ryu_venv.sh          one-time Python setup for the controller
tests/                       unit tests (flow table, live merge, traffic player, controller logic, splits)
run_live.sh                  runs the live sweep with the right interpreters
run_analysis.sh              validation + statistics + figures, zips results/
```

## Fixed parameters (`shared/config.py`)

| Parameter | Value |
|---|---|
| Hosts / service ports | 16 / 5 |
| Flow rate | 250 flows/s (Poisson) |
| Flows per run | 1,500 (training: 4,000 baseline flows, seed 42) |
| Idle timeout | 5 s |
| Table capacity | 150 offline, 80 under table pressure (live) |
| τ grid | 0.0, 0.1, …, 1.0 |
| Seeds | 1000–1007 (`N_REPETITIONS = 8`) |
| Offline latency model | 0.5 ms per hit, 8 ms per miss |
| LSTM | embedding 32, one layer of 64 units, window 8; Adam, lr 1e-3, batch 64, 5 epochs |

---

## Requirements

**Offline replay and analysis** run on any OS with Python 3.10+ and need only:

- **The standard library**, for the Markov predictor and the simulator.
- **`torch`**, for the LSTM.
- **`matplotlib` and `numpy`**, for figures.

**Live runs** need Linux with the following tested combination:

| Component | Version |
|---|---|
| Ubuntu | 22.04 |
| Mininet | 2.3.0 |
| Open vSwitch | 2.17 |
| Ryu | 4.34 |
| Python | 3.10 |
| OpenFlow | 1.3 |

Ryu 4.34 is old and pins its dependencies. Use the setup script below, not `pip install ryu` on its own.

## 1. Quick check (any machine, about a minute)

```bash
git clone <this-repo> && cd sdn-proactive-flow
for t in tests/test_*.py; do python3 "$t" || echo "FAILED: $t"; done
python3 scripts/smoke_test.py
```

The smoke test prints hit rate, proactive installs, evictions and harmful evictions per τ. Proactive installs should
fall as τ rises.

## 2. Offline replay sweep (736 runs)

```bash
pip3 install --user torch numpy matplotlib     # torch only needed for the LSTM arm
python3 scripts/run_offline_matrix.py --dry-run   # tiny wiring check
python3 scripts/run_offline_matrix.py             # full sweep
```

Logs are written to `logs/offline/<scenario>/<strategy>__tau<τ>__seed<seed>.jsonl`, one JSON event per flow.
To run a subset, pass for example `--strategies markov --scenarios baseline --taus 0.0 0.5 --seeds 1000`.

## 3. Live sweep (184 runs, Linux only)

One-time setup:

```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch python3-venv python3-dev build-essential
sudo systemctl enable --now openvswitch-switch
bash scripts/setup_ryu_venv.sh        # NO sudo: creates ~/ryu-venv (setuptools 67.6.1, eventlet 0.33.3, ryu 4.34, torch CPU)
```

Then run the sweep:

```bash
sudo bash run_live.sh --dry-run       # one combination, ~10 s; stops with a clear message if anything is missing
sudo bash run_live.sh                 # full sweep: reactive + 2 predictors x 11 tau, x 8 seeds
```

Logs are written to `logs/live/<strategy>__tau<τ>__table_pressure__seed<seed>.jsonl`, plus each run's controller
output (`<run_id>.ryu.log`).

How a live run works:

- **Two interpreters.** Mininet and the traffic agents run under the system `python3` as root. The controller runs
  in `~/ryu-venv`, and `run_live.sh` sets `RYU_PYTHON` for you.
- **Every packet is logged.** Installed rules also copy each packet to the controller, so hits are logged too. A
  Packet-In from the table-miss entry is a true miss.
- **Latency is measured at the hosts.** Each host stamps its packets on send and receive, and
  `shared/live_merge.py` joins the stamps onto the controller log. Live logs therefore have the same fields as
  offline logs.

If a run crashes, run `sudo mn -c` before starting again.

## 4. Validation, statistics and figures

From the folder that contains `logs/offline` and `logs/live`:

```bash
bash run_analysis.sh
```

This runs four steps in order:

1. The replay-vs-live check (`validate_replay.py`).
2. Spot checks (`spot_check.py`).
3. The run summary (`summarize.py`).
4. Statistics and tables (`analyze.py`) and the report figures (`paper_figures.py`, `combined_figures.py`).

Everything is written to `results/` and zipped to `~/sdn_analysis_results.zip`:

| Output | Contents |
|---|---|
| `results/tables/tau_curves.csv` | per scenario × strategy × τ medians and IQRs (savings, installs, evictions, …) |
| `results/tables/per_run.csv` | one row per run |
| `results/tables/calibration.csv`, `calibration_bins.csv` | accuracy, coverage, ECE, Brier, reliability bins |
| `results/tables/paired_vs_reference.csv` | paired savings vs reactive-only: Wilcoxon, Holm, d_z, bootstrap CIs |
| `results/tables/rq2_lstm_vs_markov.csv` | paired LSTM − Markov differences |
| `results/tables/h4_mechanism.csv` | the H4 eviction chain under table pressure |
| `results/tables/drift_halves.csv`, `replay_vs_live.csv` | drift before/after the switch; replay validation |
| `results/figures/report/` | Figures 1–4 of the report (PDF + PNG) |

The whole pipeline (920 runs plus analysis) took about 1.5 hours on one laptop.

## 5. Building the report

`report/final_report.tex` is written for LuaLaTeX and needs the four PDFs in `report/figures/`:

- `fig_tau_curves.pdf`
- `fig_pareto.pdf`
- `fig_calibration.pdf`
- `fig_h4.pdf`

Copy them from `results/figures/report/`, then build:

```bash
cd report && lualatex final_report.tex && lualatex final_report.tex
```

---

## Notes and known limitations

- **Live latency is compared within each arm.** Each arm's live latency saving is relative to its own τ = 1.0 run.
  At τ = 1.0 the LSTM controller (which loads PyTorch) had lower latency than reactive-only. The cause was not
  isolated, so live comparisons *between* arms use hit rates and misses only.
- **Offline replay treats installation as instantaneous.** It therefore misses install races, so offline savings
  are upper bounds.
- **The traffic favours Markov.** All traffic is synthetic first-order Markov, which suits the Markov predictor by
  construction. The RQ2 result does not cover traffic with longer-range structure.
- **An earlier eviction-counting bug was fixed.** The fix is in `controller/ryu_app.py`, `shared/schema.py` and
  `table_sim/flow_table.py`, and all reported runs use the corrected code.

## Use of AI tools

General-purpose AI assistants (Claude and ChatGPT) helped draft and debug parts of this code. Every component was
verified against hand-traced ground truth, the unit tests and the log consistency checks. Representative prompts
are listed in Appendix C of the report.

## Authors

Abirami Nachammai Natarajan (natar@kth.se) and Tanmay Jayanthi (tanmayj@kth.se), KTH Royal Institute of Technology.
