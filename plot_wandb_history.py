"""
Generate plots from downloaded W&B history JSONs.
Saves PNGs to mrbert/analysis/wandb_plots/plots/
"""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

WANDB_DIR = Path("mrbert/analysis/wandb_plots")
PLOTS_DIR = WANDB_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

def load_run(project, run_glob):
    """Load history + meta for a run matching a glob pattern."""
    proj_dir = WANDB_DIR / project
    if not proj_dir.exists():
        return None, None
    for run_dir in proj_dir.iterdir():
        if run_glob.lower() in run_dir.name.lower():
            h = run_dir / "history.json"
            m = run_dir / "meta.json"
            history = json.loads(h.read_text()) if h.exists() else []
            meta = json.loads(m.read_text()) if m.exists() else {}
            return history, meta
    return None, None

def col(history, key, fallbacks=()):
    """Extract a column from history, trying fallback keys."""
    keys = [key] + list(fallbacks)
    for k in keys:
        vals = [(r.get("_step", i), r[k]) for i, r in enumerate(history) if k in r and r[k] is not None]
        if vals:
            steps, values = zip(*vals)
            return list(steps), list(values)
    return [], []

def smooth(values, w=20):
    if len(values) < w:
        return values
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode="same").tolist()


# ═══════════════════════════════════════════════════════════
# 1. SNLI — accuracy & deletion rate curves for key runs
# ═══════════════════════════════════════════════════════════
print("Plotting SNLI training curves...")

snli_runs = {
    "BERT baseline":        ("mrbert-snli", "bert-snli-baseline"),
    "MrBERT 0%":            ("mrbert-snli", "mrbert-snli-0pct"),
    "MrBERT 30%":           ("mrbert-snli", "mrbert-snli-30pct_sak8"),
    "MrBERT 30% (hd-train)":("mrbert-snli", "mrbert-snli-30pct-hd"),
    "MrBERT 50%":           ("mrbert-snli", "mrbert-snli-50pct"),
    "MrBERT 70%":           ("mrbert-snli", "mrbert-snli-70pct"),
    "Random 30%":           ("mrbert-snli", "mrbert-snli-random"),
    "No-PI 30%":            ("mrbert-snli", "mrbert-snli-no-pi"),
    "Layer 1":              ("mrbert-snli", "mrbert-snli-layer1"),
    "Layer 6":              ("mrbert-snli", "mrbert-snli-layer6"),
    "Layer 9":              ("mrbert-snli", "mrbert-snli-layer9"),
}

# List all available runs in mrbert-snli for reference
proj_dir = WANDB_DIR / "mrbert-snli"
available = sorted([d.name for d in proj_dir.iterdir()]) if proj_dir.exists() else []
print(f"  Available runs: {available}")

# Build a lookup by run name prefix
run_lookup = {}
if proj_dir.exists():
    for d in proj_dir.iterdir():
        h = d / "history.json"
        m = d / "meta.json"
        if h.exists() and m.exists():
            history = json.loads(h.read_text())
            meta = json.loads(m.read_text())
            run_lookup[d.name] = (history, meta)

def find_run(project, name_fragment):
    proj_dir = WANDB_DIR / project
    if not proj_dir.exists():
        return None, None
    for d in sorted(proj_dir.iterdir()):
        if name_fragment.lower() in d.name.lower():
            h = d / "history.json"
            m = d / "meta.json"
            if h.exists():
                return json.loads(h.read_text()), (json.loads(m.read_text()) if m.exists() else {})
    return None, None


# ── 1a. Test accuracy over training steps ───────────────────
fig, ax = plt.subplots(figsize=(9, 5))

plot_configs = [
    ("bert-snli-baseline",  "BERT baseline",          "black",   "-",  2.0),
    ("mrbert-snli-0pct",    "MrBERT 0%",              "#2196F3", "--", 1.5),
    ("mrbert-snli-30pct",   "MrBERT 30%",             "#4CAF50", "-",  2.0),
    ("mrbert-snli-30pct-hd","MrBERT 30% (hard-train)","#8BC34A", "--", 1.5),
    ("mrbert-snli-50pct",   "MrBERT 50%",             "#FF9800", "-",  1.5),
    ("mrbert-snli-70pct",   "MrBERT 70%",             "#F44336", "-",  1.5),
    ("mrbert-snli-random",  "Random 30%",             "#9C27B0", ":",  1.5),
    ("mrbert-snli-no-pi",   "No-PI 30%",              "#795548", "-.", 1.5),
]

plotted = 0
for frag, label, color, ls, lw in plot_configs:
    history, meta = find_run("mrbert-snli", frag)
    if not history:
        continue
    steps, vals = col(history, "test/accuracy",
                      ["eval/accuracy", "train/accuracy", "accuracy"])
    if not vals:
        continue
    ax.plot(steps, [v * 100 if v <= 1.0 else v for v in vals],
            label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
    plotted += 1

ax.set_xlabel("Training Step")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("SNLI Test Accuracy During Training")
ax.legend(loc="lower right", ncol=2)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.tight_layout()
plt.savefig(PLOTS_DIR / "snli_accuracy_training_curves.png")
plt.close()
print(f"  Saved snli_accuracy_training_curves.png ({plotted} runs)")


# ── 1b. Deletion rate over training ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

del_configs = [
    ("mrbert-snli-30pct",    "MrBERT 30%",             "#4CAF50", "-"),
    ("mrbert-snli-30pct-hd", "MrBERT 30% (hard-train)","#8BC34A", "--"),
    ("mrbert-snli-50pct",    "MrBERT 50%",             "#FF9800", "-"),
    ("mrbert-snli-70pct",    "MrBERT 70%",             "#F44336", "-"),
    ("mrbert-snli-random",   "Random 30%",             "#9C27B0", ":"),
    ("mrbert-snli-no-pi",    "No-PI 30%",              "#795548", "-."),
    ("mrbert-snli-0pct",     "MrBERT 0%",              "#2196F3", "--"),
]

plotted = 0
for frag, label, color, ls in del_configs:
    history, _ = find_run("mrbert-snli", frag)
    if not history:
        continue
    steps, vals = col(history, "train/percent_non_pad_deleted_tokens",
                      ["train/percent_deleted_tokens", "percent_deleted_tokens",
                       "eval/percent_deleted_tokens"])
    if not vals:
        continue
    vals_pct = [v * 100 if v <= 1.0 else v for v in vals]
    ax.plot(steps, smooth(vals_pct, 15), label=label, color=color, linestyle=ls, linewidth=1.8, alpha=0.9)
    plotted += 1

ax.axhline(30, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="30% target")
ax.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="50% target")
ax.axhline(70, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="70% target")
ax.set_xlabel("Training Step")
ax.set_ylabel("Deletion Rate (%)")
ax.set_title("SNLI Token Deletion Rate During Training")
ax.legend(loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "snli_deletion_rate_curves.png")
plt.close()
print(f"  Saved snli_deletion_rate_curves.png ({plotted} runs)")


# ── 1c. PI controller alpha (deletion loss coeff) ───────────
fig, ax = plt.subplots(figsize=(9, 5))

pi_configs = [
    ("mrbert-snli-30pct",    "MrBERT 30% (PI)",  "#4CAF50", "-"),
    ("mrbert-snli-50pct",    "MrBERT 50% (PI)",  "#FF9800", "-"),
    ("mrbert-snli-70pct",    "MrBERT 70% (PI)",  "#F44336", "-"),
    ("mrbert-snli-no-pi",    "No-PI (fixed α)",  "#795548", "-."),
]
plotted = 0
for frag, label, color, ls in pi_configs:
    history, _ = find_run("mrbert-snli", frag)
    if not history:
        continue
    steps, vals = col(history, "train/delete_gate_loss_coeff",
                      ["delete_gate_loss_coeff"])
    if not vals:
        continue
    ax.plot(steps, smooth(vals, 20), label=label, color=color, linestyle=ls, linewidth=1.8)
    plotted += 1

ax.set_xlabel("Training Step")
ax.set_ylabel("Deletion Loss Coefficient α")
ax.set_title("PI Controller: Deletion Loss Coefficient α Over Training")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "snli_pi_controller_alpha.png")
plt.close()
print(f"  Saved snli_pi_controller_alpha.png ({plotted} runs)")


# ── 1d. Gate statistics (mean, std) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

gate_configs = [
    ("mrbert-snli-30pct",    "MrBERT 30%", "#4CAF50", "-"),
    ("mrbert-snli-50pct",    "MrBERT 50%", "#FF9800", "-"),
    ("mrbert-snli-70pct",    "MrBERT 70%", "#F44336", "-"),
    ("mrbert-snli-0pct",     "MrBERT 0%",  "#2196F3", "--"),
]
for frag, label, color, ls in gate_configs:
    history, _ = find_run("mrbert-snli", frag)
    if not history:
        continue
    s1, avg = col(history, "train/delete_gate_average", ["delete_gate_average"])
    s2, std = col(history, "train/delete_gate_std",     ["delete_gate_std"])
    if avg:
        axes[0].plot(s1, smooth(avg, 20), label=label, color=color, linestyle=ls, linewidth=1.8)
    if std:
        axes[1].plot(s2, smooth(std, 20), label=label, color=color, linestyle=ls, linewidth=1.8)

axes[0].set_title("Gate Mean Value")
axes[0].set_xlabel("Step"); axes[0].set_ylabel("Mean Gate Value")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_title("Gate Std Dev")
axes[1].set_xlabel("Step"); axes[1].set_ylabel("Gate Std Dev")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle("SNLI Delete Gate Statistics During Training", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "snli_gate_statistics.png")
plt.close()
print("  Saved snli_gate_statistics.png")


# ── 1e. Loss curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
loss_configs = [
    ("bert-snli-baseline", "BERT baseline", "black",   "-",  2.0),
    ("mrbert-snli-0pct",   "MrBERT 0%",    "#2196F3", "--", 1.5),
    ("mrbert-snli-30pct",  "MrBERT 30%",   "#4CAF50", "-",  2.0),
    ("mrbert-snli-50pct",  "MrBERT 50%",   "#FF9800", "-",  1.5),
    ("mrbert-snli-70pct",  "MrBERT 70%",   "#F44336", "-",  1.5),
    ("mrbert-snli-random", "Random 30%",   "#9C27B0", ":",  1.5),
]
plotted = 0
for frag, label, color, ls, lw in loss_configs:
    history, _ = find_run("mrbert-snli", frag)
    if not history:
        continue
    steps, vals = col(history, "train/cross_entropy_loss",
                      ["train/loss", "cross_entropy_loss", "loss"])
    if not vals:
        continue
    ax.plot(steps, smooth(vals, 20), label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
    plotted += 1

ax.set_xlabel("Training Step")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("SNLI Training Loss")
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "snli_loss_curves.png")
plt.close()
print(f"  Saved snli_loss_curves.png ({plotted} runs)")


# ═══════════════════════════════════════════════════════════
# 2. All projects — final accuracy bar chart
# ═══════════════════════════════════════════════════════════
print("\nPlotting cross-project summary bar chart...")

csv_path = WANDB_DIR / "all_runs_summary.csv"
import csv

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def get_acc(row):
    for k in ["test/accuracy", "eval/accuracy", "accuracy"]:
        v = row.get(k, "")
        try:
            f = float(v)
            return f * 100 if f <= 1.0 else f
        except Exception:
            pass
    return None

# Group by project, pick the best finished run per project
from collections import defaultdict
proj_runs = defaultdict(list)
for row in rows:
    acc = get_acc(row)
    if acc and row.get("state") == "finished" and acc > 50:
        proj_runs[row["project"]].append((acc, row["run_name"]))

fig, ax = plt.subplots(figsize=(10, 5))
projects_sorted = sorted(proj_runs.keys())
for i, proj in enumerate(projects_sorted):
    accs = sorted(proj_runs[proj], reverse=True)
    best_acc, best_run = accs[0]
    ax.bar(i, best_acc, color="#4CAF50", alpha=0.8, edgecolor="white")
    ax.text(i, best_acc + 0.3, f"{best_acc:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(range(len(projects_sorted)))
ax.set_xticklabels([p.replace("mrbert-", "") for p in projects_sorted],
                    rotation=35, ha="right")
ax.set_ylabel("Best Test Accuracy (%)")
ax.set_title("Best Accuracy per W&B Project")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "all_projects_best_accuracy.png")
plt.close()
print("  Saved all_projects_best_accuracy.png")


# ═══════════════════════════════════════════════════════════
# 3. MrXLMR-SNLI curves (if available)
# ═══════════════════════════════════════════════════════════
print("\nPlotting MrXLMR-SNLI curves...")
xlmr_dir = WANDB_DIR / "mrxlmr-snli"
if xlmr_dir.exists():
    xlmr_runs = sorted([d for d in xlmr_dir.iterdir() if (d / "history.json").exists()])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.tab10
    for i, run_dir in enumerate(xlmr_runs):
        history = json.loads((run_dir / "history.json").read_text())
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
        label = meta.get("name", run_dir.name)[:40]
        color = cmap(i % 10)
        s1, acc = col(history, "test/accuracy", ["eval/accuracy"])
        s2, dr  = col(history, "train/percent_non_pad_deleted_tokens",
                      ["train/percent_deleted_tokens"])
        if acc:
            axes[0].plot(s1, [v*100 if v<=1 else v for v in acc],
                         label=label, color=color, linewidth=1.5)
        if dr:
            axes[1].plot(s2, smooth([v*100 if v<=1 else v for v in dr], 15),
                         label=label, color=color, linewidth=1.5)

    axes[0].set_title("MrXLMR-SNLI Test Accuracy")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("MrXLMR-SNLI Deletion Rate")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Deletion Rate (%)")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mrxlmr_snli_curves.png")
    plt.close()
    print(f"  Saved mrxlmr_snli_curves.png ({len(xlmr_runs)} runs)")
else:
    print("  mrxlmr-snli project not found, skipping")


# ═══════════════════════════════════════════════════════════
# 4. TyDi QA
# ═══════════════════════════════════════════════════════════
print("\nPlotting TyDi QA curves...")
tydiqa_dir = WANDB_DIR / "mrbert-tydiqa"
if tydiqa_dir.exists():
    tydiqa_runs = sorted([d for d in tydiqa_dir.iterdir() if (d / "history.json").exists()])
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10
    for i, run_dir in enumerate(tydiqa_runs):
        history = json.loads((run_dir / "history.json").read_text())
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
        label = meta.get("name", run_dir.name)[:40]
        steps, acc = col(history, "test/accuracy",
                         ["eval/accuracy", "train/accuracy", "accuracy"])
        if acc:
            ax.plot(steps, [v*100 if v<=1 else v for v in acc],
                    label=label, color=cmap(i % 10), linewidth=1.5)
    ax.set_title("TyDi QA Accuracy During Training")
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tydiqa_accuracy_curves.png")
    plt.close()
    print(f"  Saved tydiqa_accuracy_curves.png ({len(tydiqa_runs)} runs)")


# ═══════════════════════════════════════════════════════════
# 5. All projects — training loss grid
# ═══════════════════════════════════════════════════════════
print("\nPlotting per-project loss grids...")
all_projects = [d for d in WANDB_DIR.iterdir()
                if d.is_dir() and d.name not in {"plots"}]

for proj_dir in sorted(all_projects):
    run_dirs = [d for d in proj_dir.iterdir() if (d / "history.json").exists()]
    if not run_dirs:
        continue
    fig, axes = plt.subplots(1, min(2, len(run_dirs)),
                              figsize=(6 * min(2, len(run_dirs)), 4), squeeze=False)
    cmap = plt.cm.tab20
    for i, run_dir in enumerate(sorted(run_dirs)[:20]):
        history = json.loads((run_dir / "history.json").read_text())
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
        label = meta.get("name", run_dir.name)[:30]
        ax = axes[0][0]
        steps, vals = col(history, "train/cross_entropy_loss",
                          ["train/loss", "cross_entropy_loss", "loss", "eval/loss"])
        if vals:
            ax.plot(steps, smooth(vals, 10), label=label, color=cmap(i % 20), linewidth=1.2, alpha=0.85)

        if len(axes[0]) > 1:
            ax2 = axes[0][1]
            s2, acc = col(history, "test/accuracy",
                          ["eval/accuracy", "train/accuracy", "accuracy"])
            if acc:
                ax2.plot(s2, [v*100 if v<=1 else v for v in acc],
                         label=label, color=cmap(i % 20), linewidth=1.2, alpha=0.85)
                ax2.set_title("Accuracy"); ax2.set_xlabel("Step")
                ax2.grid(True, alpha=0.3)

    axes[0][0].set_title("Loss"); axes[0][0].set_xlabel("Step")
    axes[0][0].grid(True, alpha=0.3)
    if len(run_dirs) <= 8:
        axes[0][0].legend(fontsize=7)
    plt.suptitle(f"Project: {proj_dir.name}", fontsize=12)
    plt.tight_layout()
    safe = proj_dir.name.replace("/", "_")
    plt.savefig(PLOTS_DIR / f"{safe}_overview.png")
    plt.close()

print(f"\n✓ All plots saved to {PLOTS_DIR}/")
print("Files:")
for f in sorted(PLOTS_DIR.iterdir()):
    print(f"  {f.name}")
