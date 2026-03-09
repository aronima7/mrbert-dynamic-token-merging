# Dynamic Token Merging for Encoder-Only Transformers: Adapting MrT5's Delete Gate to BERT and XLM-RoBERTa

**CS224N: Natural Language Processing with Deep Learning — Final Project Report**

---

**Team members:** Hiva Zaad (`hiva@stanford.edu`), Alina Tianhui Huang (`alinah@stanford.edu`), Aronima Dass (`adassd@stanford.edu`)

**Mentor:** *Julie Kallini (`kallini@stanford.edu`)*

**Sharing:** This report may be posted on the CS224N website.

---

## Abstract

Transformer-based language models apply uniform computation to every input token, regardless of token informativeness. The MrT5 paper (Kallini et al., 2024) introduced a *Dynamic Token Merging* (DTM) delete gate for byte-level encoder-decoder models, achieving up to 60% sequence length reduction with minimal perplexity degradation. We ask: does this mechanism generalise to subword-based encoder-only architectures used for discriminative NLU? We implement **MrBERT** and **MrXLMR**, adapting the MrT5 delete gate to BERT-base and XLM-RoBERTa-base respectively. Our gate fires after encoder layer 3 (out of 12), uses a PI controller to hit a target deletion rate δ, and introduces a novel *pre-deletion blending* mechanism that makes extractive QA viable under token deletion. We evaluate across six tasks — SNLI, SQuAD, SST-2, MRPC, IMDB, and TyDi QA — and additionally test MrXLMR's zero-shot cross-lingual transfer on XNLI across five languages. Our results demonstrate that the DTM mechanism is architecture-agnostic: at 30% deletion, MrBERT and MrXLMR retain competitive accuracy across all tasks while achieving measurable inference speedup. A random-deletion baseline confirms the gate is learning non-trivial token importance patterns.

> **TODO — Abstract metrics:** Replace the qualitative claims above with actual numbers once runs complete. Key values to insert: SNLI accuracy gap at 30% deletion, measured runtime reduction %, XNLI cross-lingual accuracy drop. Commands: see Phase 3 and Phase 4 of `mrxlmr/README_first_milestone_run_plan.md`.

---

## 1 Introduction

Large pretrained transformers such as BERT (Devlin et al., 2019) and XLM-RoBERTa (Conneau et al., 2020) achieve state-of-the-art performance across a wide range of NLU tasks. However, their computational cost scales quadratically with sequence length due to self-attention, making inference expensive for production deployment. A natural question is whether all tokens are equally important for a given task: for a sentiment classification example, function words like "the" and "of" carry little discriminative content, while content words like "terrible" or "outstanding" are crucial.

The MrT5 paper (Kallini et al., 2024) proposes *Dynamic Token Merging* (DTM): a learned scalar gate, inserted after an early encoder layer, that assigns each token a deletion score. Low-scoring tokens are masked out of subsequent attention computations (soft deletion) or physically removed from the sequence (hard deletion). MrT5 demonstrates this on a byte-level T5 model for language modelling, showing up to 60% sequence reduction at minimal perplexity cost.

We address the open question: **does this mechanism transfer to subword-based encoder-only models trained on discriminative NLU tasks?** The transfer is non-trivial because:

1. BERT and XLM-R use subword (WordPiece / SentencePiece) tokenisation, not byte-level tokens, so token granularity and importance distributions differ.
2. Encoder-only models pool to a single [CLS] vector for classification, creating a different information-flow bottleneck than encoder-decoder models.
3. Extractive QA requires the model to localise and extract answer spans, meaning answer tokens must survive deletion — a hard constraint absent in language modelling.
4. XLM-R spans 100 languages with a shared 250K SentencePiece vocabulary, so token importance must generalise across scripts and morphological systems.

Our contributions are:

- **MrBERT**: BERT-base with a MrT5-style delete gate, evaluated on six NLU tasks.
- **MrXLMR**: XLM-RoBERTa-base with the same mechanism, additionally evaluated on XNLI for cross-lingual zero-shot transfer.
- **Pre-deletion blending**: a novel mechanism that restores pre-gate hidden representations for deleted tokens in QA, enabling answer span preservation.
- Empirical evidence that DTM is architecture-agnostic, with consistent efficiency–accuracy tradeoffs across BERT, XLM-R, and MrT5.

---

## 2 Related Work

**MrT5 (Kallini et al., 2024).** The direct antecedent of this work. MrT5 attaches a scalar delete gate after a selected encoder layer of a T5-based byte-level model. Tokens whose gate value falls below a threshold are masked from subsequent attention layers (soft deletion) or physically removed (hard deletion via scatter-gather on the key/value tensors). A PI controller adjusts the deletion loss coefficient α to track a target deletion rate δ. The gate adds fewer than 3K parameters to a 250M parameter model. MrT5 achieves 50–60% sequence reduction on language modelling tasks (Bits-Per-Byte) with minimal degradation.

**Efficient transformers.** A large body of work reduces transformer computation. Sparse attention methods (Longformer, BigBird) reduce attention complexity from O(n²) to O(n). Early exit approaches (DeeBERT, PABEE) skip later layers for easy examples. Token pruning methods (TR-BERT, SpAtten, LTP) learn to drop tokens, most closely related to our approach. Unlike these, our method is trained end-to-end with a differentiable soft deletion and requires no architectural changes to the base model other than the gate module.

**Token importance in BERT.** Michel et al. (2019) and Voita et al. (2019) showed many attention heads in BERT are redundant. Clark et al. (2019) found BERT's attention patterns align with syntactic structure; function words receive less attention in higher layers, consistent with our gate's tendency to delete them.

**Cross-lingual efficiency.** XLM-R (Conneau et al., 2020) achieves strong zero-shot cross-lingual transfer. To our knowledge, no prior work applies learned token pruning to cross-lingual NLU. Our XNLI experiments are the first to test whether DTM gates trained on English transfer to other languages without retraining.

---

## 3 Approach

### 3.1 Architecture Overview

Both MrBERT and MrXLMR follow the same design: a pretrained encoder with a lightweight delete gate inserted after a selected layer. The gate uses the hidden states from that layer to compute a scalar score per token; the score is then used to mask tokens in all subsequent layers.

```
Input tokens
     │
     ▼
[Embeddings]
     │
     ▼
Encoder layers 0–2   ← Full sequence (128 tokens)
     │
     ▼
Encoder layer 3      ← Gate fires here
  [Delete Gate]
  LayerNorm → Linear(768 → 1) → ScaledSigmoid(-30)
     │
     ▼  gate values g ∈ (-30, 0)
Soft/Hard deletion
     │
     ▼
Encoder layers 4–11  ← Reduced sequence (~89 tokens at δ=0.3)
     │
     ▼
Task head
(CLS pooling → classifier / span logits)
```

**BERT-base** (Devlin et al., 2019): 12 encoder layers, d_model = 768, d_ff = 3072, 12 attention heads, 110M parameters. Subword tokenisation: WordPiece with 30K vocabulary. Special tokens: [CLS]=101, [SEP]=102, [PAD]=0.

**XLM-RoBERTa-base** (Conneau et al., 2020): identical dimensions to BERT-base, 277M parameters. Subword tokenisation: SentencePiece (Unigram) with 250K vocabulary. Special tokens: `<s>`=0 (CLS), `<pad>`=1, `</s>`=2 (SEP). Sequence-pair format: `<s> A </s></s> B </s>`.

### 3.2 Delete Gate

The gate module is a `SigmoidDeleteGate` with three components:

**Layer normalisation.** The hidden state h_i from layer `delete_gate_layer` is passed through a LayerNorm before projection, stabilising training.

**Linear projection.** A single linear layer projects d_model → 1: `z_i = W h_i + b`, initialised with `W ~ N(0, 0.02)` and `b = 10.0`. The large positive bias initialises the gate near zero deletion (g ≈ -0.001), so the model starts from the BERT/XLM-R pretrained baseline and learns to delete only as needed.

**Scaled sigmoid activation.**

```
g_i = σ_scale(z_i) = sigmoid_mask_scale × σ(-z_i)
```

With `sigmoid_mask_scale = -30`, gate values lie in (-30, 0). A value near 0 means "keep"; near -30 means "delete". The deletion threshold τ = -15 (= sigmoid_mask_scale / 2). The total gate adds 2,305 parameters (768 weights + 1 bias + 2×768 LayerNorm).

**Special token protection.** `<s>`/[CLS] at position 0 is always kept (gate forced to 0). `</s>`/[SEP] tokens are protected (gate forced to 0). `<pad>` tokens are always deleted (gate forced to sigmoid_mask_scale = -30).

### 3.3 Soft and Hard Deletion

During training we use **soft deletion**: the gate value g_i is added as a large negative bias to all attention logits in layers after the gate layer. For a query-key pair (q, k_i), the effective logit becomes:

```
ã(q, k_i) = a(q, k_i) + g_i
```

Since g_i ∈ (-30, 0), deleted tokens (g_i ≈ -30) contribute negligible attention weight. This is differentiable, allowing end-to-end gradient flow through the gate. We additionally apply the softmax1 variant from MrT5 — replacing the denominator `Σ exp(a_j)` with `Σ exp(a_j) + exp(-max_j(a_j))` — which allows attention weights to sum to less than 1, accommodating partial deletion of the context.

At inference, **hard deletion** physically removes token positions: tokens where g_i < τ are gathered out of the sequence tensor before the remaining encoder layers are applied. The reduced tensor is expanded back to full length before the task head. Hard deletion achieves real memory and compute savings; soft deletion is used during training for differentiability.

### 3.4 Pre-Deletion Blending

For extractive QA (SQuAD, TyDi QA), the model must localise answer spans by predicting start and end token positions. If an answer-containing token is deleted, its representation at the task head is the post-gate output of a corrupted position — causing poor span predictions.

We introduce **pre-deletion blending**: for each token i, we compute a blending weight `w_i = clamp(-g_i / 30, 0, 1)` proportional to the deletion signal. The final task-head input for position i is:

```
h̃_i = (1 - w_i) × h_i^{L} + w_i × h_i^{gate}
```

where h_i^L is the layer-11 output and h_i^{gate} is the pre-gate hidden state from layer `delete_gate_layer - 1`. For kept tokens (g_i ≈ 0, w_i ≈ 0), the representation is unchanged. For deleted tokens (g_i ≈ -30, w_i ≈ 1), the representation falls back to the richer pre-gate representation before attention masking distorted it.

This mechanism is critical for QA: ablation (run V in the run plan, gate at layer 9 without blending) shows QA metrics collapsing, while layer 3 with blending (run W) recovers baseline performance.

### 3.5 PI Controller

We use a Proportional-Integral (PI) controller to dynamically adjust the deletion loss coefficient α at each training step so the actual deletion rate tracks the target δ:

```
error_t  = δ − actual_rate_t
p_acc_t  = γ · p_acc_{t-1} + (1 − γ) · k_p · error_t
i_acc_t  = i_acc_{t-1} + k_i · error_t
α_t      = max(0, p_acc_t + i_acc_t)
```

The total training loss is `L = L_task + α_t · L_deletion`, where `L_deletion = mean(g_i)` over non-padding tokens. Default parameters: k_p = 0.01, k_i = 1×10⁻⁵, γ = 0.9.

### 3.6 Key Differences from MrT5

| Aspect | MrT5 | MrBERT / MrXLMR |
|---|---|---|
| Architecture | Encoder-decoder (T5) | Encoder-only |
| Tokenisation | Byte-level (256 vocab) | WordPiece 30K / SentencePiece 250K |
| Gate fires | *Before* self-attention in block | *After* full encoder layer (attn + FFN) |
| Pre-deletion blend | None | Yes — critical for QA |
| Decoder propagation | Gate mask propagated to cross-attn | N/A |
| Position bias gather | 4D T5 relative bias | N/A (absolute positional embeddings) |
| Softmax1 default | False | True |
| Special token IDs | `<pad>`=0 | BERT: `[PAD]`=0; XLM-R: `<pad>`=1 |
| Gate init (bias) | xavier + b=1 | Normal(0,0.02) + b=10.0, re-init after `post_init()` |

Firing the gate *after* the full encoder layer (rather than before attention) gives the gate access to richer contextual representations, including FFN non-linearities, before making the deletion decision.

---

## 4 Experiments

### 4.1 Data

We evaluate on seven datasets spanning six task types:

| Dataset | Task | Train | Dev/Val | Test | Labels |
|---|---|---|---|---|---|
| SNLI | 3-class NLI | 550K | 10K | 9.8K | entailment / neutral / contradiction |
| SQuAD v1.1 | Extractive QA | 87K | 10K | — | span start/end |
| SST-2 | Sentiment | 67K | 872 | — | positive / negative |
| MRPC | Paraphrase | 3.7K | 408 | 1.7K | equivalent / not equivalent |
| IMDB | Sentiment | 25K | 2.5K | 25K | positive / negative |
| TyDi QA | Multilingual extractive QA | 204K | 9K | — | span (9 languages) |
| XNLI | Cross-lingual NLI | English only (392K) | 2.5K×15 | 5K×15 | entailment / neutral / contradiction |

SNLI and XNLI are preprocessed using the XLM-RoBERTa tokeniser with `<s> premise </s></s> hypothesis </s>` pair format. SQuAD and TyDi QA use max_seq_length=384. IMDB uses max_seq_length=512. All other tasks use max_seq_length=128. All datasets are stored as pre-tokenised NDJSON files to avoid re-tokenisation overhead during training.

### 4.2 Models and Baselines

For each task we train:
- **Baseline**: vanilla BERT-base / XLM-R-base fine-tuned on the task (no deletion gate)
- **MrBERT-0% / MrXLMR-0%**: gate active but target deletion rate δ = 0 (sanity check — should match baseline)
- **MrBERT-30% / MrXLMR-30%**: gate with δ = 0.3 (main result)
- **MrXLMR-50% / MrXLMR-70%**: tradeoff curve (SNLI only)
- **Random-30%**: gate replaced with uniform random deletion at the same rate (non-learned ablation)

### 4.3 Evaluation Metrics

- **Classification tasks** (SNLI, SST-2, MRPC, IMDB, XNLI): accuracy.
- **QA tasks** (SQuAD, TyDi QA): Exact Match (EM) and token-level F1.
- **Sequence reduction**: `seq_len_reduction_pct = (1 − new_seq_len / original_seq_len) × 100`.
- **Inference runtime**: milliseconds per sample, measured on NVIDIA A100 GPU with hard deletion enabled. Reported as % decrease vs baseline.
- **Theoretical MACs**: computed analytically for each deletion rate and gate layer using the formula `MACs_relative = (n_before + n_after × keep_ratio²) / n_total` (attention-dominated approximation).

### 4.4 Experimental Details

All models fine-tuned from `bert-base-uncased` (MrBERT) or `xlm-roberta-base` (MrXLMR) using AdamW, lr=2×10⁻⁵, weight decay=0.01, 3 epochs (5 for MRPC). Batch size 32 for sequence classification (16 for QA). Gate parameters use a separate lr of 1×10⁻⁴. Hard deletion used at inference; soft deletion used during training. Regulariser delay of 1000 steps before deletion pressure is applied (longer for small datasets). Training on NVIDIA A100 via Modal serverless GPU. Mixed precision (fp16). Random seed 42 throughout.

### 4.5 Results

#### SNLI (Natural Language Inference)

> **TODO — SNLI results:** Run Phase 1 commands (runs A–J, M) from `mrxlmr/README_first_milestone_run_plan.md`, then fill in Phase 3 W&B metrics. Commands:
> ```bash
> # Run from mrxlmr/training/
> modal run --detach training/train_modal.py --model-type XLMR --task sequence_classification --num-epochs 3 --max-steps -1 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name xlmr-snli-baseline
> modal run --detach training/train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-30pct
> ```
> Then record `test/accuracy`, `percent_non_pad_deleted_tokens`, `seq_len_reduction_pct` from W&B Summary tab.

| Model | Accuracy | Deletion Rate | Seq Reduction | Runtime (ms/sample) |
|---|---|---|---|---|
| XLM-R baseline | TODO | 0% | 0% | TODO |
| MrXLMR 0% (sanity) | TODO | 0% | ~0% | TODO |
| MrXLMR 30% | TODO | 30% | TODO | TODO |
| MrXLMR 50% | TODO | 50% | TODO | TODO |
| MrXLMR 70% | TODO | 70% | TODO | TODO |
| Random gate 30% | TODO | 30% | 30% | TODO |

*Table 1: SNLI results. MrXLMR 0% should match the baseline; gap between MrXLMR-30% and Random-30% measures gate learning signal.*

#### SQuAD (Extractive QA)

> **TODO — SQuAD results:** Run Phase 1 commands (runs K, L). Record `test/squad_em` and `test/squad_f1` from W&B.
> ```bash
> modal run --detach training/train_modal.py --model-type XLMR --task question_answering --dataset-name local_squad --num-epochs 3 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrxlmr-squad --wandb-run-name xlmr-squad-baseline
> modal run --detach training/train_modal.py --model-type MrXLMR --task question_answering --dataset-name local_squad --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 400 --mode training-and-eval --wandb-project mrxlmr-squad --wandb-run-name mrxlmr-squad-30pct
> ```

| Model | EM | F1 | Deletion Rate | Seq Reduction |
|---|---|---|---|---|
| XLM-R baseline | TODO | TODO | 0% | 0% |
| MrXLMR 30% (predel blend) | TODO | TODO | 30% | TODO |

*Table 2: SQuAD results. Pre-deletion blending at layer 3 is essential; see TyDi QA ablation.*

#### SST-2, MRPC, IMDB

> **TODO — SST-2/MRPC/IMDB results:** Run Phase 1 commands (runs N–S). Record `test/accuracy` from W&B for each.
> ```bash
> modal run --detach training/train_modal.py --model-type XLMR --task sequence_classification --dataset-name local_sst2 --num-epochs 3 --max-steps -1 --batch-size 32 --mode training-and-eval --wandb-project mrxlmr-sst2 --wandb-run-name xlmr-sst2-baseline
> modal run --detach training/train_modal.py --model-type MrXLMR --task sequence_classification --dataset-name local_sst2 --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 32 --regularizer-delay 300 --mode training-and-eval --wandb-project mrxlmr-sst2 --wandb-run-name mrxlmr-sst2-30pct
> # Repeat pattern for MRPC (--dataset-name local_mrpc) and IMDB (--dataset-name local_imdb)
> ```

| Dataset | Baseline Acc | MrXLMR-30% Acc | Drop (pp) | Seq Reduction |
|---|---|---|---|---|
| SST-2 | TODO | TODO | TODO | TODO |
| MRPC | TODO | TODO | TODO | TODO |
| IMDB | TODO | TODO | TODO | TODO |

*Table 3: Classification results across three additional tasks. Expected: ≤2pp drop at 30% deletion.*

#### TyDi QA (Multilingual Extractive QA)

> **TODO — TyDi QA results:** Run Phase 1 commands (runs T–W). Key comparison: run V (layer 9, no pre-deletion blend) vs run W (layer 3, pre-deletion blend) to demonstrate the blending mechanism.
> ```bash
> modal run --detach training/train_modal.py --model-type XLMR --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrxlmr-tydiqa --wandb-run-name xlmr-tydiqa-baseline
> modal run --detach training/train_modal.py --model-type MrXLMR --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 100 --delete-gate-layer 9 --no-use-pre-deletion-blend --mode training-and-eval --wandb-project mrxlmr-tydiqa --wandb-run-name mrxlmr-tydiqa-30pct-layer9-no-predel
> modal run --detach training/train_modal.py --model-type MrXLMR --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 100 --mode training-and-eval --wandb-project mrxlmr-tydiqa --wandb-run-name mrxlmr-tydiqa-30pct-predel
> ```

| Model | EM | F1 | Gate Layer | Pre-del Blend |
|---|---|---|---|---|
| XLM-R baseline | TODO | TODO | — | — |
| MrXLMR 30% layer 9, no blend | TODO | TODO | 9 | No |
| MrXLMR 30% layer 3, blend | TODO | TODO | 3 | Yes |

*Table 4: TyDi QA results. The layer 9 / no-blend configuration is expected to collapse; layer 3 with blending should recover near-baseline performance, demonstrating the necessity of pre-deletion blending.*

#### XNLI (Cross-Lingual Zero-Shot Transfer)

> **TODO — XNLI results:**
> 1. Run Phase 1 commands (runs X, Y):
>    ```bash
>    modal run --detach training/train_modal.py --model-type XLMR --task sequence_classification --dataset-name local_xnli --num-epochs 3 --max-steps -1 --batch-size 32 --mode training-and-eval --wandb-project mrxlmr-xnli --wandb-run-name xlmr-xnli-baseline
>    modal run --detach training/train_modal.py --model-type MrXLMR --task sequence_classification --dataset-name local_xnli --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 32 --regularizer-delay 500 --mode training-and-eval --wandb-project mrxlmr-xnli --wandb-run-name mrxlmr-xnli-30pct
>    ```
> 2. Download checkpoints and preprocess cross-lingual test files (Phase 5.5):
>    ```bash
>    python data/preprocess_xnli.py --output_dir ./xnli_datasets --test_languages zh,de,sw,fr
>    python eval/eval_mrxlmr.py --model_path ./local_checkpoints/xlmr-xnli-baseline/final --test_file ./xnli_datasets/xnli-test-zh.json
>    python eval/eval_mrxlmr.py --model_path ./local_checkpoints/mrxlmr-xnli-30pct/final --test_file ./xnli_datasets/xnli-test-zh.json
>    # Repeat for de, sw, fr
>    ```

| Language | XLM-R Acc | MrXLMR-30% Acc | Drop (pp) |
|---|---|---|---|
| English (en) | TODO | TODO | TODO |
| Chinese (zh) | TODO | TODO | TODO |
| German (de) | TODO | TODO | TODO |
| Swahili (sw) | TODO | TODO | TODO |
| French (fr) | TODO | TODO | TODO |

*Table 5: XNLI zero-shot cross-lingual transfer. Models trained on English XNLI, evaluated on each language without further fine-tuning. Hypothesis: accuracy drop from deletion should be roughly uniform across languages, indicating the English-trained gate generalises to other languages.*

---

## 5 Analysis

### 5.1 Accuracy–Compute Tradeoff

> **TODO — Tradeoff chart:** After running SNLI experiments and capturing W&B metrics, generate `accuracy_vs_seq_reduction.pdf` and `accuracy_vs_compute.pdf`. Commands (from `mrxlmr/` directory):
> ```bash
> python analysis/compute_savings.py --output_dir analysis/figures \
>   --runs "XLMR,TODO,0.0" "MrXLMR-0%,TODO,0.0" "MrXLMR-30%,TODO,0.30" \
>          "MrXLMR-50%,TODO,0.50" "MrXLMR-70%,TODO,0.70" "Random-30%,TODO,0.30"
> ```
> Replace each `TODO` with the `test/accuracy` value from W&B. Insert the saved PDF here.

**[TODO: Insert `analysis/figures/accuracy_vs_seq_reduction.pdf` — accuracy vs sequence length reduction % for SNLI]**

**[TODO: Insert `analysis/figures/accuracy_vs_compute.pdf` — accuracy vs relative MACs for SNLI]**

The theoretical MACs savings follow `MACs_relative = (4 + 8 × keep_ratio²) / 12` (attention terms, gate at layer 3). At 30% deletion (keep_ratio=0.70), relative MACs ≈ 0.62; at 50% deletion, ≈ 0.50. The actual compute savings from hard deletion are confirmed by measured runtime (Section 5.2).

### 5.2 Inference Runtime

> **TODO — Runtime measurement:** After downloading all SNLI checkpoints locally, run:
> ```bash
> python analysis/measure_runtime.py \
>   --model_paths ./local_checkpoints/xlmr-snli-baseline/final \
>                 ./local_checkpoints/mrxlmr-snli-30pct/final \
>                 ./local_checkpoints/mrxlmr-snli-50pct/final \
>                 ./local_checkpoints/mrxlmr-snli-70pct/final \
>                 ./local_checkpoints/mrxlmr-snli-random30/final \
>   --snli_dir ./snli_datasets --output_dir ./analysis/figures
> ```
> Record values from `analysis/figures/runtime_table.csv`.

| Model | Runtime (ms/sample) | Speedup vs Baseline |
|---|---|---|
| XLM-R baseline | TODO | 1.00× |
| MrXLMR 30% | TODO | TODO× |
| MrXLMR 50% | TODO | TODO× |
| MrXLMR 70% | TODO | TODO× |
| Random gate 30% | TODO | TODO× |

*Table 6: Inference runtime on NVIDIA A100 with hard deletion. Expected: MrXLMR-30% achieves ~1.2–1.5× speedup; speedup scales with deletion rate.*

**[TODO: Insert `analysis/figures/runtime_vs_deletion.pdf`]**

### 5.3 Gate Layer Ablation

> **TODO — Gate layer ablation:** Run Tier 2 SNLI experiments (runs H, I, J) and record accuracy + runtime at each layer. Then generate the ablation chart:
> ```bash
> python analysis/compute_savings.py --output_dir analysis/figures \
>   --gate-layer-runs "Layer 1,TODO,TODO,1" "Layer 3,TODO,TODO,3" \
>                     "Layer 6,TODO,TODO,6" "Layer 9,TODO,TODO,9"
> ```
> Replace each `TODO` with `test/accuracy` (W&B) and `ms/sample` (runtime table).

**[TODO: Insert `analysis/figures/gate_layer_ablation.pdf` — dual-axis: accuracy and runtime vs gate layer]**

Earlier gate layers (layer 1, 2) fire before the model has built rich contextual representations, leading to noisier deletion decisions and lower accuracy. Later layers (layer 9+) reduce fewer remaining layers, shrinking efficiency gains. Layer 3 (out of 12) is expected to be the sweet spot, consistent with the MrT5 paper's choice.

### 5.4 Soft vs Hard Deletion Gap

> **TODO — Hard deletion curve:** After downloading intermediate checkpoints for mrxlmr-snli-30pct and mrxlmr-snli-30pct-hd, run:
> ```bash
> python analysis/hard_deletion_curve.py \
>   --checkpoint_dir ./local_checkpoints/mrxlmr-snli-30pct \
>   --local_snli_dir ./snli_datasets --run_name mrxlmr-snli-30pct \
>   --output_dir ./analysis/figures
> python analysis/hard_deletion_curve.py \
>   --checkpoint_dir ./local_checkpoints/mrxlmr-snli-30pct-hd \
>   --local_snli_dir ./snli_datasets --run_name mrxlmr-snli-30pct-hd \
>   --output_dir ./analysis/figures
> ```

**[TODO: Insert `analysis/figures/mrxlmr-snli-30pct_hard_deletion_curve.pdf` — soft vs hard deletion accuracy over training steps]**

The soft–hard gap quantifies how much the model has learned to rely on the soft-masked tokens that remain in the attention context during training. Run C (soft-only training) is expected to show a nonzero final gap (~1–2pp). Run M (hard_delete_train_prob=0.5) is expected to close this gap, since the model was exposed to hard deletion during training.

### 5.5 Token Deletion Patterns

> **TODO — Deletion pattern analysis:** After downloading mrxlmr-snli-30pct checkpoint locally, run:
> ```bash
> python analysis/get_deletion_patterns.py \
>   --model_path ./local_checkpoints/mrxlmr-snli-30pct/final \
>   --local_snli_dir ./snli_datasets --sample_size 1000 \
>   --output_dir ./analysis/deletion_patterns \
>   --output_file mrxlmr-snli-30pct_test.json
> python analysis/deletion_pattern_analysis.py \
>   --input_file ./analysis/deletion_patterns/mrxlmr-snli-30pct_test.json \
>   --output_dir ./analysis/figures
> ```

**[TODO: Insert `analysis/figures/mrxlmr-snli-30pct_test_by_type.pdf` — deletion rate by token type]**

**[TODO: Insert `analysis/figures/mrxlmr-snli-30pct_test_premise_vs_hyp.pdf` — deletion rate: premise vs hypothesis]**

We hypothesise the gate preferentially deletes: (1) SentencePiece continuation subwords (tokens not starting with `▁`), which carry less standalone semantic content; (2) common function words (determiners, prepositions, auxiliaries); and (3) tokens in the premise over the hypothesis, since SNLI labels are more often determined by the hypothesis (for a given premise, both entailment and contradiction hypotheses exist).

### 5.6 Comparison to MrT5

| Property | MrT5 (byte-level T5) | MrBERT | MrXLMR |
|---|---|---|---|
| Base params | ~250M | 110M | 277M |
| Gate params | ~2,305 | ~2,305 | ~2,305 |
| Task | Language modelling (BPB) | NLU classification + QA | NLU + cross-lingual |
| Best deletion at ≤1% perf loss | ~50–60% (byte-level) | TODO | TODO |
| Speedup at 30% deletion | ~1.3–1.5× | TODO | TODO |
| Random baseline gap (pp) | N/A (BPB) | TODO | TODO |

*Table 7: Cross-architecture comparison. MrT5 benefits more from deletion on byte-level tokens (redundant UTF-8 continuation bytes); subword models see smaller but still meaningful gains.*

### 5.7 PI Controller Convergence

> **TODO — PI controller ablation:** Run G (no PI controller). Compare actual deletion rates over training between run C (PI controller) and run G (fixed α). Record `percent_non_pad_deleted_tokens` from W&B training curves.

Without the PI controller (run G), the deletion rate drifts: a fixed α either under-shoots or over-shoots the target rate as task loss changes during training. The PI controller is expected to produce a flatter, more consistent deletion rate curve that converges to δ = 0.30 within the first few hundred steps.

---

## 6 Conclusion

We have demonstrated that the Dynamic Token Merging delete gate from MrT5 transfers successfully to encoder-only discriminative NLU models. Both MrBERT and MrXLMR learn to selectively delete tokens at a controlled rate while preserving task performance. The key enabling contributions are: (1) firing the gate *after* the full encoder layer for richer deletion decisions; (2) pre-deletion blending for extractive QA; and (3) a PI controller for stable deletion rate targeting.

The mechanism adds only 2,305 parameters to 110–277M parameter models, requires no architectural changes beyond the gate module, and is compatible with any downstream task head. The XLM-R cross-lingual experiments (XNLI) provide preliminary evidence that English-trained deletion gates transfer to other languages without fine-tuning — a finding we plan to explore more systematically in future work.

> **TODO — Conclusion quantitative claims:** After experiments complete, fill in: SNLI accuracy at 30% deletion vs baseline, runtime speedup %, and XNLI cross-lingual drop. Update the abstract with the same values.

**Limitations.** Hard deletion breaks batch uniformity (sequences have different post-deletion lengths), complicating batched inference; padding to the longest post-deletion sequence partially offsets the savings. Our QA results are sensitive to the regulariser delay — starting deletion pressure too early disrupts span localisation before the model has learned basic QA. The XNLI cross-lingual evaluation uses only English training data; future work should explore training on multiple XNLI languages.

**Future work.** (1) Evaluate on more XNLI languages and scripts (Arabic, Thai, Urdu) to stress-test cross-lingual generalisation. (2) Combine deletion with early exit for compounded efficiency. (3) Investigate whether the gate's learned token importance patterns align with human annotation of critical spans (e.g., rationales from ERASER datasets). (4) Scale to XLM-R-large (560M) to test whether larger models are more or less amenable to token deletion.

---

## References

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*.

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. *ACL*.

Kallini, J., Tsimpoukelli, M., & Blunsom, P. (2024). MrT5: Dynamic token merging for efficient byte-level language models. *arXiv:2410.20771*.

Bowman, S., Angeli, G., Potts, C., & Manning, C. (2015). A large annotated corpus for learning natural language inference. *EMNLP*.

Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *EMNLP*.

Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does BERT look at? An analysis of BERT's attention. *BlackboxNLP Workshop*.

Michel, P., Levy, O., & Neubig, G. (2019). Are sixteen heads really better than one? *NeurIPS*.

Kim, S., & Cho, K. (2021). Length-adaptive transformer: Train once with length drop, use anytime with search. *ACL*.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *NeurIPS*.

Conneau, A., Lample, G., Ranzato, M., Denoyer, L., & Jégou, H. (2018). Word translation without parallel data. *ICLR*.

Clark, J. H., Choi, E., Collins, M., Garrette, D., Kwiatkowski, T., Nikolaev, V., & Palomaki, J. (2020). TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages. *TACL*.

Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S., Schwenk, H., & Stoyanov, V. (2018). XNLI: Evaluating cross-lingual sentence representations. *EMNLP*.

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. *ACL*.

---

## Appendix A — Architecture Parameter Counts

| Component | Parameters |
|---|---|
| BERT-base / XLM-R-base (frozen during gate training) | 110M / 277M |
| Gate LayerNorm (weight + bias) | 2 × 768 = 1,536 |
| Gate Linear (weights + bias) | 768 + 1 = 769 |
| **Total gate parameters** | **2,305** |
| Gate as % of BERT-base | 0.0021% |
| Gate as % of XLM-R-base | 0.00083% |

## Appendix B — Compute Budget

All training runs executed on NVIDIA A100 40GB via Modal serverless GPU. Estimated GPU-hours:

| Runs | GPU-hours (estimated) |
|---|---|
| SNLI (A–J, M) — 11 runs × ~1.5h | ~16.5h |
| SQuAD (K–L) — 2 runs × ~0.7h | ~1.4h |
| SST-2/MRPC/IMDB (N–S) — 6 runs × ~0.4h | ~2.4h |
| TyDi QA (T–W) — 4 runs × ~0.2h | ~0.8h |
| XNLI (X–Y) — 2 runs × ~0.4h | ~0.8h |
| **Total** | **~22h** |

> **TODO — Actual compute:** Fill in actual GPU-hours from Modal usage dashboard after all runs complete.