# Logging Validation Metrics to W&B via HuggingFace Trainer

## Problem

W&B dashboard shows only **test**, **train**, and **System** sections — no **eval** (validation) section. This happens when the Trainer isn't configured to run periodic evaluation during training.

## Solution

Three things must be in place: an `eval_dataset`, an `evaluation_strategy`, and a `compute_metrics` function.

### 1. Define `compute_metrics`

Whatever this function returns gets automatically logged with the `eval/` prefix in W&B.

```python
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}  # → logged as eval/accuracy in W&B
```

### 2. Set Evaluation Strategy in `TrainingArguments`

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",   # run eval at the end of each epoch
    # evaluation_strategy="steps", # alternative: run eval every N steps
    # eval_steps=500,              # required if strategy is "steps"
    report_to="wandb",
    # ... other args
)
```

### 3. Pass `eval_dataset` and `compute_metrics` to the Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,        # ← required for periodic eval
    compute_metrics=compute_metrics,  # ← required for custom metrics
)

trainer.train()
```

After this, W&B will show an **eval** section with panels like `eval/loss`, `eval/accuracy`, etc., updated each epoch (or every `eval_steps`).

## Troubleshooting

| Symptom | Likely Cause |
|---|---|
| No eval section at all | `evaluation_strategy` is set to `"no"` (the default) |
| Only one eval data point | Eval is only happening via `trainer.evaluate()` after training, not periodically |
| Eval section exists but no custom metrics | `compute_metrics` not passed to Trainer |
| Run crashed before first eval | Training failed before completing the first epoch/eval checkpoint |

## Notes

- The **test** section in W&B typically comes from `trainer.predict(test_dataset)` or manual `wandb.log({"test/...": ...})` calls — this is separate from periodic validation.
- If you want to log additional metrics beyond what `compute_metrics` returns, you can subclass `Trainer` and override `evaluation_loop` or use a custom `WandbCallback`.
