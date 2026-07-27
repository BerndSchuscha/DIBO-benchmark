#!/usr/bin/env bash
# Cleanup script: run from the PIBOB repo root BEFORE the first push to GitHub.
# Removes build artifacts, caches, and "copy" leftovers.
# Review the OPTIONAL section at the bottom before uncommenting anything.

set -euo pipefail

echo ">> Removing Python build artifacts and caches..."
rm -rf build/ pibob.egg-info/
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
find . -name "*.pyc" -delete

echo ">> Removing 'copy' leftovers..."
# NB: quotes matter, filenames contain spaces
rm -f  "pibob/PIBO-main/precipitator copy.py"
rm -f  "pibob/PIBO-main/01_Binary_Precipitation copy.ipynb"
rm -f  "pibob/PIBO-main/01_Binary_Precipitation  example copy.ipynb"
rm -rf "results/Structures_Problem copy"
rm -f  "requirements/kawin-env copy.txt"

echo ">> Removing debug dumps..."
rm -rf examples/mvn_fail_dumps

# ------------------------------------------------------------------
# OPTIONAL — review before enabling:
#
# 1) Legacy PIBO prototype (old notebooks, CSVs, .tdb databases).
#    If it's not referenced by the paper, remove it or move it to an
#    'archive' branch:
# rm -rf pibob/PIBO-main
#
# 2) The results/ folder (272 MB) is .gitignore'd, so it will not be
#    pushed — no need to delete it locally. If you want to publish the
#    sweep results, use a Zenodo archive or GitHub release asset
#    instead of committing them.
# ------------------------------------------------------------------

echo ">> Done. Suggested next steps:"
echo "   git init (if needed) && git add -A && git status   # inspect what will be committed"
