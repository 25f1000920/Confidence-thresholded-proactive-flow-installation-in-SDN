#!/usr/bin/env bash
# Runs every post-experiment step in order (no sudo needed):
#   1. replay-vs-live validation   2. spot checks
#   3. summary table               4. figures + statistics
# and zips results/ to ~/sdn_analysis_results.zip
#
#   bash run_analysis.sh

cd "$(dirname "$0")" || exit 1
set -e

if [ ! -d logs/live ] || [ ! -d logs/offline ]; then
    echo "No logs/live and logs/offline here. Run this inside the folder that holds your results."
    exit 1
fi
python3 -c "import matplotlib" 2>/dev/null || pip3 install --user -q matplotlib
mkdir -p results

echo "== 1/4 Replay-vs-live validation"
python3 scripts/validate_replay.py
echo
echo "== 2/4 Spot checks"
python3 scripts/spot_check.py
echo
echo "== 3/4 Summary table"
python3 scripts/summarize.py > results/summary_printout.txt
tail -n 3 results/summary_printout.txt
echo
echo "== 4/4 Figures and statistics (about 30 seconds)"
python3 scripts/analyze.py
echo
rm -f ~/sdn_analysis_results.zip
zip -qr ~/sdn_analysis_results.zip results
echo "Done. Everything is in results/ and zipped to ~/sdn_analysis_results.zip"
