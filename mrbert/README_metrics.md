# MrBERT W&B Metrics Reference

All metrics are logged to Weights & Biases during training and evaluation.
Prefixes follow HuggingFace Trainer conventions: `train/` during training steps,
`eval/` during validation, and `test/` for the final held-out test evaluation.

---

## Loss Metrics

| Metric                     | Split(s) | Description                                                                    | Expectation                                                        |
|----------------------------|----------|--------------------------------------------------------------------------------|--------------------------------------------------------------------|
| `train/loss`               | train    | Total combined loss: `cross_entropy_loss + α × delete_gate_loss`               | Lower is better; should decrease steadily                          |
| `train/cross_entropy_loss` | train    | Task loss only (CE for classification, masked LM, QA span)                     | Lower is better; primary measure of task learning                  |
| `eval/loss`                | eval     | Validation task loss reported by HF Trainer                                    | Lower is better; should track training loss without diverging      |
| `train/delete_gate_loss`   | train    | Deletion regularization loss — penalizes deviation from `target_deletion_rate` | Lower is better; should approach 0 once deletion rate is on target |

---

## Task Performance Metrics

| Metric           | Split(s) | Description                                                                                       | Expectation                                                |
|------------------|----------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| `train/accuracy` | train    | Batch-level accuracy: fraction of examples where `argmax(logits) == label` (classification tasks) | Higher is better; expect increase over training            |
| `eval/accuracy`  | eval     | Validation accuracy from `compute_metrics` (sequence classification / MLM tasks)                  | Higher is better; main downstream performance signal       |
| `test/accuracy`  | test     | Accuracy on the held-out test set, re-logged explicitly with `test/` prefix                       | Higher is better; final reported number                    |
| `eval/start_acc` | eval     | QA only — fraction of examples where predicted start token index is exactly correct               | Higher is better; partial signal for span prediction       |
| `eval/end_acc`   | eval     | QA only — fraction of examples where predicted end token index is exactly correct                 | Higher is better; partial signal for span prediction       |
| `eval/span_em`   | eval     | QA only — exact match of the full predicted span (start and end both correct)                     | Higher is better; stricter than start/end acc individually |

---

## Delete Gate Distribution Metrics (MrBERT only)

| Metric                        | Split(s) | Description                                                                 | Expectation                                                                                       |
|-------------------------------|----------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `train/delete_gate_average`   | train    | Mean gate value across all tokens and all sequences in the logging interval | Should settle near `1 - target_deletion_rate` once trained (e.g. ~0.7 for 30% deletion)           |
| `eval/delete_gate_average`    | eval     | Same as above, computed on validation batches                               | Same target as train; should be consistent with train value                                       |
| `train/delete_gate_std`       | train    | Std of gate values across token positions (averaged over batch)             | Higher means sharper bimodal gate (tokens clearly kept or deleted); too low signals gate collapse |
| `train/delete_gate_max_value` | train    | Mean of the per-sequence maximum gate value in each batch                   | Should approach 1.0 as the gate learns to confidently keep important tokens                       |
| `train/delete_gate_min_value` | train    | Mean of the per-sequence minimum gate value in each batch                   | Should approach 0.0 as the gate learns to confidently delete unimportant tokens                   |

---

## Deletion Rate / Sequence Length Metrics (MrBERT only)

| Metric                                 | Split(s) | Description                                                                                  | Expectation                                                                              |
|----------------------------------------|----------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `train/percent_deleted_tokens`         | train    | % of all tokens deleted (including PAD tokens, which are always deleted)                     | Should converge toward `target_deletion_rate` + pad fraction; reflects total compression |
| `eval/percent_deleted_tokens`          | eval     | Same as above, measured on validation batches                                                | Should be consistent with training value                                                 |
| `test/percent_deleted_tokens`          | test     | Same metric on the test set                                                                  | Ideally close to `target_deletion_rate`                                                  |
| `train/percent_non_pad_deleted_tokens` | train    | % of *non-PAD* tokens deleted — the true learned deletion rate, excluding structural padding | Should converge to `target_deletion_rate` (e.g. 0.30 for 30%)                            |
| `eval/percent_non_pad_deleted_tokens`  | eval     | Same as above on validation batches                                                          | Should match training value; large gap suggests overfitting of gate                      |
| `test/percent_non_pad_deleted_tokens`  | test     | Same metric on the test set                                                                  | Key efficiency number to report alongside task accuracy                                  |
| `train/new_seq_len`                    | train    | Average effective sequence length (tokens with gate > threshold), in token count             | Should decrease relative to `max_seq_length`; reflects actual compute saved              |
| `eval/new_seq_len`                     | eval     | Same, measured on validation                                                                 | Should match training; used to compute `seq_len_reduction_pct`                           |
| `test/new_seq_len`                     | test     | Same on the test set                                                                         | Together with `percent_non_pad_deleted_tokens`, confirms efficiency at inference         |
| `train/seq_len_reduction_pct`          | train    | `(1 - new_seq_len / max_seq_length) × 100` — % reduction vs. the full padded length          | Higher means more tokens dropped; should approach `target_deletion_rate × 100`           |
| `eval/seq_len_reduction_pct`           | eval     | Same, measured on validation                                                                 | Should closely match training; directly represents inference speed-up potential          |
| `test/seq_len_reduction_pct`           | test     | Same on test set                                                                             | Final reported efficiency number                                                         |

---

## PI Controller Metric (MrBERT only)

| Metric                         | Split(s) | Description                                                                                | Expectation                                                                                                               |
|--------------------------------|----------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `train/delete_gate_loss_coeff` | train    | Current value of α — the PI-controller-adjusted weight on the deletion regularization loss | Starts at `deletion_loss_weight`; rises if deletion rate is below target, falls if above; should stabilize once on target |

---

## Training Dynamics Metrics (HF Trainer built-ins)

| Metric                    | Split(s) | Description                                    | Expectation                                                            |
|---------------------------|----------|------------------------------------------------|------------------------------------------------------------------------|
| `train/learning_rate`     | train    | Current learning rate from the scheduler       | Should decrease smoothly under cosine/linear decay                     |
| `train/grad_norm`         | train    | Global gradient norm before clipping           | Should be bounded and not spike; large spikes may indicate instability |
| `train/epoch`             | train    | Fractional epoch number at the time of logging | Increases monotonically from 0 to `num_epochs`                         |
| `eval/runtime`            | eval     | Wall-clock seconds for one evaluation pass     | Lower is better; can compare MrBERT vs BERT to measure actual speedup  |
| `eval/samples_per_second` | eval     | Evaluation throughput in examples/sec          | Higher is better; key efficiency benchmark                             |
| `eval/steps_per_second`   | eval     | Evaluation throughput in optimizer steps/sec   | Higher is better                                                       |

---

## Display-only Metrics (not numeric, not averaged)

| Metric           | Description                                                                              |
|------------------|------------------------------------------------------------------------------------------|
| `epoch_progress` | Human-readable string e.g. `"Epoch 1.42/3"` — shows fractional progress through training |
| `elapsed_time`   | Wall-clock time since training started, formatted as `"Xh MMm SSs"`                      |