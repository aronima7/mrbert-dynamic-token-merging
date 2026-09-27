#!/usr/bin/env bash
#
# Prepares an anonymized copy of the repository for double-blind submission.
# Output: ./anonymous-submission/ (a fresh git repo ready for anonymous.4open.science)
#
# Usage: bash prepare_anonymous_repo.sh
#
set -euo pipefail

DEST="./anonymous-submission"

echo "=== Preparing anonymous submission ==="

# Clean slate
rm -rf "$DEST"
mkdir -p "$DEST"

# ─── 1. Copy files from the staging area (index), not HEAD ───────────────────
# This respects staged deletions (git rm) that haven't been committed yet.
# Only files currently in the index are exported — deleted files are excluded.
echo "[1/6] Exporting tracked files from index..."
git checkout-index -a --prefix="$DEST/"

# ─── 2. Remove files that should NOT be in the anonymous repo ─────────────────
echo "[2/6] Removing non-essential / identifying files..."

# IDE config (leaks project name CS224N-project)
rm -rf "$DEST/.idea"

# Compiled bytecode (should never be shared)
rm -rf "$DEST/mrbert/models/__pycache__"

# W&B metadata files (contain email, hostname, entity, git remote, paths)
find "$DEST/mrbert/analysis/wandb_plots" -name "meta.json" -delete 2>/dev/null || true
find "$DEST/mrbert/analysis/wandb_plots" -name "wandb-metadata.json" -delete 2>/dev/null || true
find "$DEST/mrbert/analysis/wandb_plots" -name "*.log" -delete 2>/dev/null || true
# Remove empty directories left behind
find "$DEST/mrbert/analysis/wandb_plots" -type d -empty -delete 2>/dev/null || true

# Old README with author name and course info
rm -f "$DEST/mrbert/README-old.md"

# RUNS.md with W&B/Modal dashboard links
rm -f "$DEST/mrbert/RUNS.md"

# CLAUDE.md (internal dev notes, not needed for submission; mentions CS224N)
rm -f "$DEST/CLAUDE.md"

# Final project report (separate from COLM paper, identifies course)
rm -rf "$DEST/final-project-report"

# CS224N final project report inside COLM dir (identifies course + authors)
rm -rf "$DEST/COLM-Paper-dynamic-token-merging/cs224n-final-project-report"

# COLM logs.md (internal dev log with personal file paths throughout)
rm -f "$DEST/COLM-Paper-dynamic-token-merging/logs.md"

# .claude/ directory (Claude Code settings, not relevant)
rm -rf "$DEST/.claude"

# This script itself (not needed in the anonymous submission)
rm -f "$DEST/prepare_anonymous_repo.sh"

# All __pycache__ directories (compiled bytecode, not needed)
find "$DEST" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# W&B download log (contains entity name)
rm -f "$DEST/mrbert/analysis/wandb_download.log"

# .DS_Store
find "$DEST" -name ".DS_Store" -delete 2>/dev/null || true

# ─── 3. Scrub identity from source code files ────────────────────────────────
echo "[3/6] Scrubbing identity from source code..."

# mrbert/models/modeling_mrbert.py — remove author line
sed -i '' 's/^# Author: Hiva Mohammadzadeh$/# Author: Anonymous/' "$DEST/mrbert/models/modeling_mrbert.py"

# mrbert/analysis/download_wandb_plots.py — replace entity
sed -i '' 's/ENTITY = "aronima7-stanford-university"/ENTITY = "anonymous"/' "$DEST/mrbert/analysis/download_wandb_plots.py"

# mrbert/analysis/launch_hard_deletion_runs.py — replace identifying refs
sed -i '' 's|https://modal.com/logs/hiva/main|https://modal.com/logs/anonymous/main|' "$DEST/mrbert/analysis/launch_hard_deletion_runs.py"
sed -i '' 's/"  - Project: aronima"/"  - Project: anonymous"/' "$DEST/mrbert/analysis/launch_hard_deletion_runs.py"
sed -i '' 's/"aronima"/"anonymous"/' "$DEST/mrbert/analysis/launch_hard_deletion_runs.py"

# mrbert/analysis/deletion_correlation_analysis.py — replace hardcoded path
sed -i '' 's|/Users/hivamoh/Desktop/CS224N/project/CS224N-project|./|' "$DEST/mrbert/analysis/deletion_correlation_analysis.py"

# util/README_modal.md — replace W&B URL
sed -i '' 's|https://wandb.ai/aronima7-stanford-university/mrbert|https://wandb.ai/anonymous/mrbert|' "$DEST/util/README_modal.md"

# Scrub files that may or may not exist (depends on whether directories were git rm'd)
for f in \
    "$DEST/mrt5/utils.py" \
    "$DEST/mrt5/README.md" \
    "$DEST/mrdiffusion-sedd/train_modal_mrdiffusion.py" \
    "$DEST/mrdiffusion-sedd/train_mrdiffusion.py" \
    "$DEST/mrdiffusion-sedd/README_diff.md" \
    "$DEST/mrdiffusion-sedd/README_runs.md" \
    "$DEST/mrdiffusion-sedd/README_architecture.md" \
    "$DEST/diffusion/Score-Entropy-Discrete-Diffusion/README.md"; do
    [ -f "$f" ] && sed -i '' \
        -e 's|aronima7-stanford-university|anonymous|g' \
        -e 's|nwuseraronima7|nwuseranonymous|g' \
        -e 's|/Users/aronimadass/Desktop/projects/stanford/CS224N-project/|./|g' \
        -e 's|/Users/aronimadass/[^[:space:]"]*|./|g' \
        -e 's|/Users/hivamoh/[^[:space:]"]*|./|g' \
        "$f" || true
done

# ─── 4. Scrub mrbert/README.md ───────────────────────────────────────────────
echo "[4/6] Scrubbing mrbert/README.md..."

# Replace author/course header
sed -i '' 's/^\*\*Author:\*\* Hiva Mohammadzadeh/\*\*Author:\*\* Anonymous/' "$DEST/mrbert/README.md"
sed -i '' 's/^\*\*Course:\*\* CS224N - Natural Language Processing with Deep Learning/\*\*Course:\*\* [Redacted for anonymous review]/' "$DEST/mrbert/README.md"

# Replace GitHub URLs
sed -i '' 's|https://github.com/HivaMohammadzadeh1/CS224N-project.git|https://github.com/anonymous/anonymous-repo.git|g' "$DEST/mrbert/README.md"
sed -i '' 's|CS224N-project|anonymous-repo|g' "$DEST/mrbert/README.md"

# Replace W&B URLs
sed -i '' 's|https://wandb.ai/aronima7-stanford-university/mrbert[^[:space:]]*|https://wandb.ai/anonymous/mrbert|g' "$DEST/mrbert/README.md"

# Replace Modal URLs
sed -i '' 's|https://modal.com/apps[^[:space:]]*|https://modal.com/apps/anonymous|g' "$DEST/mrbert/README.md"

# Replace personal file paths
sed -i '' 's|/Users/aronimadass/Desktop/projects/stanford/CS224N-project/|./|g' "$DEST/mrbert/README.md"
sed -i '' 's|/Users/aronimadass/[^[:space:]]*|[path redacted]|g' "$DEST/mrbert/README.md"

# Replace CS224N references in text
sed -i '' 's/CS224N/[course-redacted]/g' "$DEST/mrbert/README.md"
sed -i '' 's/stanfordnlp/stanfordnlp/g' "$DEST/mrbert/README.md"  # preserve dataset names (not identifying)

# ─── 5. Scrub all_runs_summary.csv (URL column contains entity) ──────────────
echo "[5/6] Scrubbing W&B entity from all_runs_summary.csv..."

sed -i '' 's/aronima7-stanford-university/anonymous/g' "$DEST/mrbert/analysis/wandb_plots/all_runs_summary.csv"

# ─── 5b. Global find-and-replace for any remaining identity strings ──────────
echo "      Running global scrub pass..."

# Catch any remaining occurrences across all text files
find "$DEST" -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.csv" -o -name "*.cfg" -o -name "*.toml" -o -name "*.sh" -o -name "*.log" \) -exec sed -i '' \
    -e 's/aronima7-stanford-university/anonymous/g' \
    -e 's/aronima7/anonymous/g' \
    -e 's/aronima\.dass@outlook\.com/anonymous@example.com/g' \
    -e 's/Aronimas-MacBook-Pro-2\.local/anonymous-host/g' \
    -e 's/Hiva Mohammadzadeh/Anonymous/g' \
    -e 's/HivaMohammadzadeh1/anonymous/g' \
    -e 's|/Users/hivamoh/[^[:space:]"]*|./|g' \
    -e 's|/Users/aronimadass/Desktop/projects/stanford/[^[:space:]"]*|./|g' \
    -e 's|/Users/aronimadass/[^[:space:]"]*|./|g' \
    -e 's|nwuseraronima7|nwuseranonymous|g' \
    -e 's|modal.com/logs/hiva/|modal.com/logs/anonymous/|g' \
    -e 's|modal.com/apps/aronima7/|modal.com/apps/anonymous/|g' \
    -e 's|CS224N-project|anonymous-project|g' \
    -e 's|CS224N|[course-redacted]|g' \
    {} +

# ─── 6. Global sweep for any remaining leaks ─────────────────────────────────
echo "[6/6] Final verification sweep..."

LEAKS=$(grep -rl "aronima\|hivamoh\|Hiva Mohammadzadeh\|aronima7\|Aronimas-MacBook" "$DEST" 2>/dev/null || true)
if [ -n "$LEAKS" ]; then
    echo ""
    echo "⚠️  WARNING: Remaining identity references found in:"
    echo "$LEAKS"
    echo ""
    echo "Review these files manually before publishing."
else
    echo "✓ No identity leaks detected."
fi

# Check for "stanford" references that aren't dataset names
STANFORD_LEAKS=$(grep -rl "stanford" "$DEST" 2>/dev/null | while read f; do
    if grep "stanford" "$f" | grep -qv "stanfordnlp"; then
        echo "$f"
    fi
done || true)
if [ -n "$STANFORD_LEAKS" ]; then
    echo ""
    echo "⚠️  WARNING: 'stanford' references (not stanfordnlp datasets) in:"
    echo "$STANFORD_LEAKS"
fi

# ─── 7. Initialize as a fresh git repo (no history) ──────────────────────────
echo ""
echo "Initializing fresh git repo (no history)..."
cd "$DEST"
git init -q
git add -A
git commit -q -m "Anonymous submission"
cd - > /dev/null

echo ""
echo "=== Done ==="
echo "Anonymous repo created at: $DEST"
echo ""
echo "Next steps:"
echo "  1. Review the output: cd $DEST && grep -r 'aronima\\|stanford\\|hivamoh' ."
echo "  2. Push to a fresh anonymous GitHub repo:"
echo "     cd $DEST && git remote add origin <anonymous-repo-url> && git push -u origin main"
echo "  3. Submit via https://anonymous.4open.science using that repo URL"
echo ""
echo "Note: anonymous.4open.science automatically strips git author info,"
echo "but file contents must already be clean (which this script handles)."