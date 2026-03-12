"""
Per-example deletion rate vs. loss correlation analysis for MrBERT-30% on SNLI.

For each test example, computes:
  - per-example deletion rate (fraction of non-padding tokens deleted)
  - per-example cross-entropy loss

Then plots and analyses the correlation across multiple models:
  - MrBERT-30% (main result)
  - MrBERT-50%
  - MrBERT-70%
  - Random-30% (baseline)

Saves figures to final-project-report/figures/
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "mrbert/models")

DEVICE = "cpu"
FIG_DIR = Path("final-project-report/figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_snli_test(tokenizer, max_examples=2000):
    """Load SNLI test set and tokenize."""
    from datasets import load_dataset
    ds = load_dataset("snli", split="test")
    ds = ds.filter(lambda x: x["label"] != -1)
    examples = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        enc = tokenizer(
            ex["premise"], ex["hypothesis"],
            max_length=128, truncation=True, padding="max_length",
            return_tensors="pt"
        )
        examples.append({
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(128, dtype=torch.long)).squeeze(0),
            "label": torch.tensor(ex["label"]),
        })
    return examples


def compute_per_example_stats(model, examples, batch_size=64):
    """Returns list of dicts with loss and deletion_rate per example."""
    from torch.nn import CrossEntropyLoss
    ce_loss = CrossEntropyLoss(reduction="none")
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            input_ids      = torch.stack([e["input_ids"] for e in batch]).to(DEVICE)
            attention_mask = torch.stack([e["attention_mask"] for e in batch]).to(DEVICE)
            token_type_ids = torch.stack([e["token_type_ids"] for e in batch]).to(DEVICE)
            labels         = torch.stack([e["label"] for e in batch]).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            # Per-example loss
            logits = outputs.logits  # (B, num_labels)
            losses = ce_loss(logits, labels).cpu().numpy()

            # Per-example deletion rate from gate values
            if hasattr(outputs, "gate_values") and outputs.gate_values is not None:
                gate = outputs.gate_values  # (B, seq_len) in [-30, 0]
                threshold = model.config.sigmoid_mask_scale / 2  # -15
                # count non-padding tokens deleted
                pad_mask = attention_mask.bool()  # True = real token
                deleted = (gate < threshold) & pad_mask
                # exclude [CLS] and [SEP] from denominator
                special_ids = set([
                    model.config.pad_token_id or 0,
                ])
                # simpler: non-padding, non-[CLS] at pos 0
                non_pad = pad_mask.float().sum(dim=1)  # total non-pad per example
                del_count = deleted.float().sum(dim=1)
                del_rate = (del_count / non_pad.clamp(min=1)).cpu().numpy()
            else:
                del_rate = np.zeros(len(batch))

            correct = (logits.argmax(-1) == labels).cpu().numpy()

            for j in range(len(batch)):
                results.append({
                    "loss": float(losses[j]),
                    "deletion_rate": float(del_rate[j]),
                    "correct": bool(correct[j]),
                    "label": int(labels[j].item()),
                })
            if (i // batch_size) % 5 == 0:
                print(f"  {i+len(batch)}/{len(examples)} examples...", flush=True)

    return results


def plot_correlation(results, model_name, color, ax_scatter, ax_bins):
    """Plot scatter and binned analysis on provided axes."""
    dr = np.array([r["deletion_rate"] * 100 for r in results])
    loss = np.array([r["loss"] for r in results])
    correct = np.array([r["correct"] for r in results])

    pr, pp = stats.pearsonr(dr, loss)
    sr, sp = stats.spearmanr(dr, loss)

    # Scatter (subsample 500 for clarity)
    idx = np.random.choice(len(dr), min(500, len(dr)), replace=False)
    ax_scatter.scatter(dr[idx], loss[idx], alpha=0.25, s=8, color=color)
    # Regression line
    m, b = np.polyfit(dr, loss, 1)
    x_line = np.linspace(dr.min(), dr.max(), 100)
    ax_scatter.plot(x_line, m * x_line + b, color=color, lw=2,
                    label=f"{model_name}\nr={pr:.3f} (p={pp:.3f})")

    # Binned means
    bins = np.arange(0, 105, 10)
    bin_means, bin_edges, _ = stats.binned_statistic(dr, loss, statistic="mean", bins=bins)
    bin_sems,  _,           _ = stats.binned_statistic(dr, loss, statistic="sem",  bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(bin_means)
    ax_bins.plot(bin_centers[valid], bin_means[valid], color=color, lw=2,
                 marker="o", ms=5, label=model_name)
    ax_bins.fill_between(bin_centers[valid],
                         bin_means[valid] - bin_sems[valid],
                         bin_means[valid] + bin_sems[valid],
                         alpha=0.2, color=color)

    return {"model": model_name, "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
            "mean_dr": float(dr.mean()), "mean_loss": float(loss.mean()),
            "accuracy": float(correct.mean() * 100)}


def run_analysis():
    print("Loading models and tokenizer...")
    from transformers import BertTokenizerFast
    sys.path.insert(0, "mrbert/models")
    from modeling_mrbert import MrBertForSequenceClassification
    from configuration_mrbert import MrBertConfig

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Model checkpoints
    checkpoints = {
        "MrBERT 30%":    "mrbert_nli/final",
        "MrBERT 50%":    None,   # will skip if not available
        "MrBERT 70%":    None,
        "Random 30%":    None,
    }

    # Find available checkpoints
    available = {}
    for name, path in checkpoints.items():
        if path and Path(path).exists():
            available[name] = path
            print(f"  Found: {name} at {path}")
        else:
            # Try to find in mrbert_* dirs
            for d in Path(".").glob("mrbert_*/final"):
                cfg_path = d / "config.json"
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text())
                    dr = cfg.get("target_deletion_rate", cfg.get("deletion_rate", None))
                    dt = cfg.get("deletion_type", "")
                    if name == "MrBERT 50%" and dr and abs(dr - 0.5) < 0.05:
                        available[name] = str(d); print(f"  Found: {name} at {d}")
                    elif name == "MrBERT 70%" and dr and abs(dr - 0.7) < 0.05:
                        available[name] = str(d); print(f"  Found: {name} at {d}")
                    elif name == "Random 30%" and dt == "random":
                        available[name] = str(d); print(f"  Found: {name} at {d}")

    if not available:
        print("No model checkpoints found. Using deletion_patterns JSON instead.")
        return run_analysis_from_json()

    print(f"\nLoading test data...")
    examples = load_snli_test(tokenizer, max_examples=2000)
    print(f"  {len(examples)} test examples")

    colors = {"MrBERT 30%": "#4CAF50", "MrBERT 50%": "#FF9800",
              "MrBERT 70%": "#F44336", "Random 30%": "#9C27B0"}

    fig_scatter, axes_s = plt.subplots(1, len(available), figsize=(6*len(available), 5))
    if len(available) == 1: axes_s = [axes_s]

    fig_bins, ax_bins = plt.subplots(figsize=(9, 5))
    fig_bins2, ax_bins2 = plt.subplots(figsize=(9, 5))

    all_stats = []
    for i, (name, path) in enumerate(available.items()):
        print(f"\nAnalysing {name} ({path})...")
        config = MrBertConfig.from_pretrained(path)
        model = MrBertForSequenceClassification.from_pretrained(path, config=config)
        model = model.to(DEVICE)

        results = compute_per_example_stats(model, examples)
        stats_row = plot_correlation(results, name, colors.get(name, "blue"),
                                     axes_s[i], ax_bins)
        all_stats.append(stats_row)

        # Save per-example results
        out = Path(f"deletion_analysis_snli/per_example_{name.replace(' ','_').replace('%','pct')}.json")
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f)
        print(f"  Saved {len(results)} examples to {out}")

        # Correct vs incorrect deletion rates on ax_bins2
        dr = np.array([r["deletion_rate"]*100 for r in results])
        correct = np.array([r["correct"] for r in results])
        bins = np.arange(0, 105, 10)
        acc_bins, edges, _ = stats.binned_statistic(dr, correct.astype(float),
                                                     statistic="mean", bins=bins)
        cnt_bins, _, _ = stats.binned_statistic(dr, correct.astype(float),
                                                 statistic="count", bins=bins)
        centers = (edges[:-1]+edges[1:])/2
        valid = ~np.isnan(acc_bins) & (cnt_bins > 5)
        ax_bins2.plot(centers[valid], acc_bins[valid]*100,
                      color=colors.get(name,"blue"), lw=2, marker="o", ms=5, label=name)

        axes_s[i].set(xlabel="Per-Example Deletion Rate (%)",
                      ylabel="Cross-Entropy Loss",
                      title=f"{name}\nPearson r={stats_row['pearson_r']:.3f} (p={stats_row['pearson_p']:.3f})")
        axes_s[i].legend(fontsize=8)
        axes_s[i].grid(alpha=0.25)
        del model

    # Finalise binned loss figure
    ax_bins.set(xlabel="Per-Example Deletion Rate (%)",
                ylabel="Mean Cross-Entropy Loss",
                title="Loss vs. Per-Example Deletion Rate\n(binned means ± SEM)")
    ax_bins.legend(); ax_bins.grid(alpha=0.25)
    fig_bins.tight_layout()
    fig_bins.savefig(FIG_DIR / "deletion_rate_vs_loss_binned.png")
    print(f"\nSaved deletion_rate_vs_loss_binned.png")

    # Finalise accuracy figure
    ax_bins2.set(xlabel="Per-Example Deletion Rate (%)",
                 ylabel="Accuracy (%)",
                 title="Accuracy vs. Per-Example Deletion Rate\n(are high-deletion examples harder?)")
    ax_bins2.legend(); ax_bins2.grid(alpha=0.25)
    fig_bins2.tight_layout()
    fig_bins2.savefig(FIG_DIR / "deletion_rate_vs_accuracy_binned.png")
    print(f"Saved deletion_rate_vs_accuracy_binned.png")

    # Scatter grid
    fig_scatter.suptitle("Per-Example Deletion Rate vs. Cross-Entropy Loss", fontsize=13)
    fig_scatter.tight_layout()
    fig_scatter.savefig(FIG_DIR / "deletion_rate_vs_loss_scatter.png")
    print(f"Saved deletion_rate_vs_loss_scatter.png")

    # Print summary table
    print("\n\n=== Correlation Summary ===")
    print(f"{'Model':<20} {'Mean DR':>8} {'Mean Loss':>10} {'Acc':>7} {'Pearson r':>10} {'p':>8} {'Spearman r':>11}")
    for s in all_stats:
        print(f"{s['model']:<20} {s['mean_dr']:>7.1f}% {s['mean_loss']:>10.4f} "
              f"{s['accuracy']:>6.2f}% {s['pearson_r']:>10.4f} {s['pearson_p']:>8.4f} "
              f"{s['spearman_r']:>11.4f}")

    return all_stats


def run_analysis_from_json():
    """
    Fallback: use the existing deletion_patterns JSON which has gate_values
    and correct/incorrect per token, but no per-example loss.
    We compute a proxy: fraction of high-gate-value tokens deleted as a
    proxy for 'deletion of important tokens'.
    """
    print("Running analysis from existing deletion_patterns JSON...")
    data = json.load(open("mrbert/analysis/deletion_patterns/mrbert-snli-30pct_test.json"))
    print(f"  {len(data)} examples")

    per_example = []
    for ex in data:
        gate = np.array(ex["gate_values"])       # shape (seq_len,)
        mask = np.array(ex["attention_mask"])    # 1=real, 0=pad
        del_mask = np.array(ex["deletion_mask"]) # 1=deleted

        non_pad = mask.sum()
        if non_pad == 0:
            continue

        # Deletion rate (non-pad tokens)
        del_rate = del_mask[mask == 1].mean()

        # Mean gate value for kept vs deleted tokens
        kept_gate = gate[(mask == 1) & (del_mask == 0)]
        del_gate  = gate[(mask == 1) & (del_mask == 1)]

        # Label and prediction
        correct = ex["correct"]
        label = ex["label"]

        per_example.append({
            "deletion_rate": float(del_rate),
            "correct": bool(correct),
            "label": int(label),
            "label_str": ex.get("label_str", ""),
            "mean_gate_kept": float(kept_gate.mean()) if len(kept_gate) > 0 else 0.0,
            "mean_gate_deleted": float(del_gate.mean()) if len(del_gate) > 0 else -30.0,
            "num_tokens": int(non_pad),
        })

    dr = np.array([e["deletion_rate"] * 100 for e in per_example])
    correct = np.array([e["correct"] for e in per_example])
    label = np.array([e["label"] for e in per_example])

    # ── Figure 1: Deletion rate histogram coloured by correct/incorrect ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(dr[correct], bins=30, alpha=0.6, color="#4CAF50",
                 label=f"Correct (n={correct.sum()})", density=True)
    axes[0].hist(dr[~correct], bins=30, alpha=0.6, color="#F44336",
                 label=f"Incorrect (n={(~correct).sum()})", density=True)
    axes[0].set(xlabel="Deletion Rate (%)", ylabel="Density",
                title="Deletion Rate: Correct vs. Incorrect Predictions")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(dr[correct], dr[~correct])
    axes[0].text(0.05, 0.95, f"KS stat={ks_stat:.3f}\np={ks_p:.4f}",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # ── Figure 1b: Accuracy by deletion rate bin ──────────────────────
    bins = np.arange(0, 105, 10)
    acc_bins, edges, _ = stats.binned_statistic(dr, correct.astype(float),
                                                 statistic="mean", bins=bins)
    cnt_bins, _, _ = stats.binned_statistic(dr, correct.astype(float),
                                             statistic="count", bins=bins)
    sem_bins, _, _ = stats.binned_statistic(dr, correct.astype(float),
                                             statistic="sem", bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    valid = ~np.isnan(acc_bins) & (cnt_bins >= 5)

    bar_colors = plt.cm.RdYlGn(acc_bins[valid])
    bars = axes[1].bar(centers[valid], acc_bins[valid] * 100,
                       width=8, color=bar_colors, alpha=0.85, edgecolor="white")
    axes[1].errorbar(centers[valid], acc_bins[valid] * 100,
                     yerr=sem_bins[valid] * 100, fmt="none", color="black",
                     capsize=3, linewidth=1.5)
    for i, (c, a, n) in enumerate(zip(centers[valid], acc_bins[valid], cnt_bins[valid])):
        axes[1].text(c, a*100+1.5, f"n={int(n)}", ha="center", va="bottom",
                     fontsize=7.5, color="gray")
    # Overall accuracy line
    axes[1].axhline(correct.mean()*100, color="black", ls="--", lw=1.5,
                    label=f"Overall acc. {correct.mean()*100:.1f}%")
    axes[1].set(xlabel="Per-Example Deletion Rate (%)", ylabel="Accuracy (%)",
                title="Accuracy vs. Deletion Rate\n(MrBERT-30%, SNLI test, n=1000)")
    axes[1].set_ylim(60, 105); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.25)

    # ── Figure 1c: Mean gate value for kept vs deleted tokens by bin ──
    kept_gate = np.array([e["mean_gate_kept"] for e in per_example])
    del_gate  = np.array([e["mean_gate_deleted"] for e in per_example])

    kg_bins, _, _ = stats.binned_statistic(dr, kept_gate, statistic="mean", bins=bins)
    dg_bins, _, _ = stats.binned_statistic(dr, del_gate,  statistic="mean", bins=bins)

    axes[2].plot(centers[valid], kg_bins[valid], color="#4CAF50", lw=2,
                 marker="o", ms=5, label="Mean gate (kept tokens)")
    axes[2].plot(centers[valid], dg_bins[valid], color="#F44336", lw=2,
                 marker="s", ms=5, label="Mean gate (deleted tokens)")
    axes[2].axhline(-15, color="gray", ls=":", lw=1.2, label="Deletion threshold (−15)")
    axes[2].set(xlabel="Per-Example Deletion Rate (%)",
                ylabel="Mean Gate Value",
                title="Gate Values vs. Deletion Rate\n(kept vs. deleted tokens)")
    axes[2].legend(); axes[2].grid(alpha=0.25)

    plt.suptitle("Per-Example Deletion Analysis (MrBERT-30%, SNLI test set)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "per_example_deletion_analysis.png", bbox_inches="tight")
    plt.close()
    print("Saved per_example_deletion_analysis.png")

    # ── Figure 2: Deletion rate by label ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    label_colors = {0: "#2196F3", 1: "#FF9800", 2: "#4CAF50"}

    for lbl, lname in label_names.items():
        mask = label == lbl
        axes[0].hist(dr[mask], bins=20, alpha=0.55, density=True,
                     color=label_colors[lbl], label=f"{lname} (n={mask.sum()})")
        axes[1].scatter(dr[mask], [lbl + np.random.uniform(-0.25, 0.25)
                                   for _ in range(mask.sum())],
                        alpha=0.2, s=6, color=label_colors[lbl])

    axes[0].set(xlabel="Deletion Rate (%)", ylabel="Density",
                title="Deletion Rate Distribution by Label Class")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    # Accuracy by label and deletion bin
    for lbl, lname in label_names.items():
        mask = label == lbl
        if mask.sum() < 20: continue
        acc_b, ed, _ = stats.binned_statistic(dr[mask], correct[mask].astype(float),
                                               statistic="mean", bins=np.arange(0,105,15))
        cnt_b, _, _ = stats.binned_statistic(dr[mask], correct[mask].astype(float),
                                              statistic="count", bins=np.arange(0,105,15))
        ctrs = (ed[:-1]+ed[1:])/2
        vld = ~np.isnan(acc_b) & (cnt_b>=5)
        axes[1].plot(ctrs[vld], acc_b[vld]*100, color=label_colors[lbl],
                     lw=2, marker="o", ms=5, label=f"{lname}")

    axes[1].set(xlabel="Per-Example Deletion Rate (%)", ylabel="Accuracy (%)",
                title="Accuracy vs. Deletion Rate by Label\n(does neutral suffer more?)")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "deletion_by_label.png", bbox_inches="tight")
    plt.close()
    print("Saved deletion_by_label.png")

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    print(f"Overall accuracy: {correct.mean()*100:.2f}%")
    print(f"Mean deletion rate: {dr.mean():.1f}% ± {dr.std():.1f}%")
    print(f"KS test (correct vs incorrect deletion rate): stat={ks_stat:.3f}, p={ks_p:.4f}")
    print(f"\nAccuracy by deletion bin:")
    for c, a, n in zip(centers[valid], acc_bins[valid], cnt_bins[valid]):
        print(f"  {c-5:.0f}--{c+5:.0f}%: {a*100:.1f}% (n={int(n)})")
    print(f"\nAccuracy by label:")
    for lbl, lname in label_names.items():
        m = label == lbl
        print(f"  {lname}: {correct[m].mean()*100:.1f}%  mean_dr={dr[m].mean():.1f}%")

    return per_example


if __name__ == "__main__":
    import os
    os.chdir("/Users/hivamoh/Desktop/CS224N/project/CS224N-project")
    results = run_analysis()
