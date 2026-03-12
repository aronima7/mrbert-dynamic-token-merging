-----------
XLM-R
-----------
   * Which XLM-R model size should we target as the base model? → xlm-roberta-base (Recommended)
   * For TyDi QA, XLM-R is multilingual — should we use all 11 languages or English-only (like MrBERT)? → English only (same as MrBERT)                                                                  
   * Should I also create a Modal training script (train_modal.py) for mrxlmr, or just the model + training scripts? → Yes, include Modal script (Recommended)
   * For data preprocessing (SNLI, SQuAD, etc.) — should we re-preprocess with XLM-R tokenizer or reuse MrBERT's pre-tokenized files? → Re-preprocess with XLM-R tokenizer (Recommended)  
-----------
❯ does xlmr, like bert have special tokens like CLS to be protected from deletion?                                                                                                                                                                  
  ⎿  ⧉ Selected 1 lines from ../README_local.md in PyCharm                         
                                                                                                                                                                                                                                                    
⏺ Yes. XLM-R has two special tokens that are protected, and one that is always deleted:                                                                                                                                                           
                                                                                                                                                                                                                                                    
  ┌───────┬─────┬────────────────────────────────────┬─────────────────────────────────────────┐                                                                                                                                                    
  │ Token │ ID  │                Role                │             Gate behaviour              │                                                                                                                                                    
  ├───────┼─────┼────────────────────────────────────┼─────────────────────────────────────────┤                                                                                                                                                    
  │ <s>   │ 0   │ CLS equivalent (always position 0) │ Always kept — forced to gate = 0.0      │                                                                                                                                                  
  ├───────┼─────┼────────────────────────────────────┼─────────────────────────────────────────┤
  │ </s>  │ 2   │ SEP equivalent                     │ Always kept — forced to gate = 0.0      │
  ├───────┼─────┼────────────────────────────────────┼─────────────────────────────────────────┤
  │ <pad> │ 1   │ Padding                            │ Always deleted — forced to gate = -30.0 │
  └───────┴─────┴────────────────────────────────────┴─────────────────────────────────────────┘

  This is implemented in SigmoidDeleteGate.forward() in modeling_mrxlmr.py:181-198:

  # Never delete <s> (position 0, CLS)
  gate_values[:, 0, :] = 0.0

  # Protect </s> tokens (SEP, token_id=2)
  sep_mask = (input_ids == XLM_R_SEP_TOKEN_ID).unsqueeze(-1)
  gate_values = torch.where(sep_mask, torch.zeros_like(gate_values), gate_values)

  # Always delete <pad> tokens (token_id=1)
  pad_mask = (input_ids == self.pad_token_id).unsqueeze(-1)
  gate_values = torch.where(pad_mask, torch.full_like(gate_values, -30.0), gate_values)

  One important difference from BERT: XLM-R sequence pairs use two </s> tokens as the separator (<s> A </s></s> B </s>), so there are three protected SEP positions per pair input vs BERT's one [SEP]. All three are caught by the token_id == 2
  mask.
-----------
❯ can I run multiple modal runs at the same time and in that scenario, are the volume of data being shared between runs?                                                                                                                            
                                                                                                                         
⏺ Yes to both — and the sharing is mostly fine, but there's one gotcha.                                                                                                                                                                             
                                                                                                                                                                                                                                                  
  Parallelism: Modal is designed for this. Each modal run --detach spawns an independent container with its own A100. You can launch all runs from the run plan at once and Modal handles scheduling.                                               
   
  Volume sharing: All runs share the single mrxlmr-checkpoints volume. In practice this means:                                                                                                                                                      
                                                                                                                                                                                                                                                  
  ┌─────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
  │                        What's shared                        │                                       Safe?                                       │
  ├─────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ Dataset files (snli_datasets/, etc.)                        │ ✅ Read-only after preprocessing — all runs read from the same files, no conflict │
  ├─────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ Checkpoints (mrxlmr-snli-30pct/, xlmr-snli-baseline/, etc.) │ ✅ Each run writes to its own named subdirectory, no overlap                      │
  ├─────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ Preprocessing (the first time a dataset is written)         │ ⚠️ Race condition                                                                 │
  └─────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

  The race condition: If two runs both need SNLI and neither has preprocessed it yet, both will pass the sentinel check simultaneously (neither sees snli-validation.json), both launch preprocess_snli.py concurrently, and both write to the same
  files at the same time — corrupting the output.

  Fix — preprocess once before launching all training runs:

  # Step 1: preprocess all datasets (single run, not detached, wait for it)
  modal run train_modal.py --model-type XLMR --max-steps 1 --dataset-name local_snli
  modal run train_modal.py --model-type XLMR --max-steps 1 --dataset-name local_squad
  modal run train_modal.py --model-type XLMR --max-steps 1 --dataset-name local_sst2
  modal run train_modal.py --model-type XLMR --max-steps 1 --dataset-name local_mrpc
  modal run train_modal.py --model-type XLMR --max-steps 1 --dataset-name local_imdb
  modal run train_modal.py --model-type XLMR --max-steps 1 --dataset-name local_tydiqa
  modal run train_modal.py --model-type XLMR --max-steps 1 --dataset-name local_xnli

  # Step 2: now launch all training runs in parallel with --detach
  modal run --detach train_modal.py --model-type XLMR ...
  modal run --detach train_modal.py --model-type MrXLMR ...
  # etc.

  With --max-steps 0 the run exits immediately after preprocessing, writing the sentinel file. All subsequent --detach training runs will see the sentinel and skip preprocessing safely.
-----------
⏺ The volume is in a corrupted state — snli-train.json was deleted (from our earlier fix) but snli-validation.json still exists, so the sentinel check passes but training fails.

  Clean out the partial files from the volume and let it fully re-preprocess:

  modal volume rm mrxlmr-checkpoints snli_datasets/snli-validation.json
  modal volume rm mrxlmr-checkpoints snli_datasets/snli-test.json

  Then re-run the preprocess step (without --max-steps 0 since that still tries to load the dataset):

  modal run train_modal.py --model-type XLMR --max-steps 1 --dataset-name local_snli

  --max-steps 1 does one training step — enough to trigger full preprocessing, write all three files to the volume, and confirm the dataset is intact. Once that completes successfully, you can Ctrl-C or
  let it finish, then launch all your real training runs in parallel with --detach.

  Alternatively, since you already have the full local snli_datasets files from the earlier preprocess_snli.py run, you can upload them directly to skip the wait:

  modal volume put mrxlmr-checkpoints ./snli_datasets/snli-train.json snli_datasets/snli-train.json
  modal volume put mrxlmr-checkpoints ./snli_datasets/snli-validation.json snli_datasets/snli-validation.json
  modal volume put mrxlmr-checkpoints ./snli_datasets/snli-test.json snli_datasets/snli-test.json

  Note those local files only have 1000 samples (--max_samples 1000) — fine for a smoke test but not for real training. For actual runs you'd want the full dataset, so the modal run --max-steps 1
  approach (which downloads and preprocesses the full SNLI dataset on the A100) is the right path for production.
-----------
 Yes, metrics are being published to W&B. Here's what's wired up:                                                                                                                                         
                            
  Setup (__post_init__, lines 260–268):                                                                                                                                                                    
  - report_to = ["wandb"] when disable_wandb=False (the default for Modal runs)                                                                                                                            
  - os.environ["WANDB_PROJECT"] set to wandb_project (default "mrxlmr")                                                                                                                                    
  - WANDB_API_KEY injected via the wandb-secret Modal secret                                                                                                                                               
                  
  Training metrics (every logging_steps=50, via MrXLMRTrainer.log()):
  - loss, cross_entropy_loss, delete_gate_loss
  - accuracy, percent_deleted_tokens, percent_non_pad_deleted_tokens
  - delete_gate_average, delete_gate_std, delete_gate_max_value, delete_gate_min_value
  - delete_gate_loss_coeff (the PI controller's current α)
  - new_seq_len, elapsed_time

  Eval metrics (every eval_steps, now fixed to 500): standard HuggingFace eval logs (eval_loss, eval_accuracy, etc.)

  After training completes (lines 905–920): wandb.config.update() with all hyperparams (model_type, task, delete_gate_layer, target_deletion_rate, etc.)

  Final test metrics (lines 1199–1206): pushed to wandb.run.summary — accuracy, deletion stats, gate averages

  So you should see all of those in your W&B dashboard under project mrxlmr, run name mrxlmr-MrXLMR-local_snli (or whatever wandb_run_name was passed).
-----------
  ---                                                                                                                                                                                                      
  How gate values actually work:                                                                                                                                                                           
                                                                                                                                                                                                           
  ScaledSigmoid computes: -30 * sigmoid(-logit), so gate values are in [-30, 0]:                                                                                                                           
  - gate ≈ 0 → no attention bias → token kept
  - gate ≈ -30 → large negative attention bias → token deleted

  deletion_threshold = -15.0 (midpoint) — correct in config.

  ---
  Why 0pct deletes more tokens than 30pct and underperforms the baseline:

  The PI controller update is:
  error = target_rate - actual_rate
  alpha = max(0, p_acc + i_acc)

  The max(0, ...) clamp means alpha can never go negative. For target=0.0:
  - Any actual deletion rate > 0% → error is negative → alpha drops to 0 immediately
  - Once alpha=0, the deletion loss term vanishes from the total loss
  - But the gate is still active in the forward pass — it still adds an attention bias to every subsequent layer
  - The task loss gradient (NLI classification) flows back through the gate and can push it toward deleting tokens if that helps the classifier

  The 30pct run has the PI controller actively driving alpha up when actual < 30%, keeping the gate engaged. The 0pct run's controller instantly clamps to alpha=0 and can't counteract task-loss-driven
  deletion.

  ---
  The mrxlmr-snli-0pct run is not a valid "0% deletion" control. With target=0.0, the gate learns from task gradients unchecked, ends up deleting ~80% of tokens, and hurts accuracy because it's noisily
  masking informative tokens.

  The correct 0% deletion control is --bypass_gate, which disables the gate mechanism entirely. That's what you want for a "same architecture but no deletion" comparison. The true run to compare against
  the 30pct model is:

  modal run --detach train_modal.py \
    --model-type MrXLMR --bypass-gate \
    --dataset-name local_snli \
    --wandb-run-name mrxlmr-snli-bypass \
    --wandb-project mrxlmr-snli \
    --max-steps -1

  Or, if you only care about the upstream "does deletion hurt task performance vs a clean XLM-R baseline", the xlmr-snli-baseline run you already have is the right comparison. The 0pct run is not useful
  as designed.
-----------
⏺ Several compounding reasons explain this gap:                                                                                                                                                            
   
  1. Competing objectives during fine-tuning                                                                                                                                                               
  The total loss is task_loss + α * deletion_loss. The model has to simultaneously classify correctly AND hit a 30% deletion target. These conflict: the tokens most useful for NLI (e.g., negation words,
  key premise/hypothesis alignment tokens) are exactly the ones the gate should preserve, but the PI controller is blindly pushing toward 30% deletion regardless of token importance.                     
                  
  2. Fine-tuning a pretrained model vs. training from scratch
  MrT5 trained from scratch — the encoder and gate co-evolved. Here you're injecting a randomly-initialized gate into a pretrained XLM-R that already learned representations optimized for full-sequence
  attention. Layers 4–11 (8 of 12 layers, 66% of the model) now receive disrupted attention coverage during fine-tuning. The pretrained weights need to simultaneously un-learn full-sequence
  representations AND re-learn partial-sequence representations while also fitting the task — a significantly harder optimization problem.

  3. CLS token loses access to 30% of the sequence in upper layers
  For SNLI, the CLS token needs to compare premise and hypothesis. With 30% of tokens masked in layers 4–11, the CLS representation in the final layer reflects an incomplete view of both sentences. If
  any critical alignment tokens (e.g., "not", "never", entity names) are deleted, the classification degrades.

  4. The curves haven't converged
  The MrXLMR-30pct curve is still rising at 50k steps (~3 epochs on SNLI). XLM-R baseline plateaued around 20k steps. The deletion model is learning more slowly because the gate is also learning — you'd
  likely need 5–8 epochs minimum to see the final accuracy. The MrT5 paper trained for much longer.

  5. regularizer_delay=1000 creates a training discontinuity
  Steps 0–999: pure task loss, gate contributes noise to attention. Steps 1000+: PI controller starts ramping α, abruptly changing the loss landscape. The model that was learning NLI now has to also
  optimize the gate — this resets progress somewhat, visible as the slight plateau/inflection around 5–10k steps in your chart.

  Bottom line: An ~8% accuracy gap at ~3 epochs is within expected range for this type of modification. To narrow it you'd want:
  - More training epochs (5–8+)
  - A higher delete_gate_layer (e.g., 6 or 8) so fewer layers are affected
  - Potentially freezing the base XLM-R weights for a few thousand steps while the gate initializes, then unfreezing
    (It's done via a HuggingFace TrainerCallback that toggles requires_grad on the base model parameters at a specified step.)
-----------
### Each run ~7.5 hrs on A100, costing 
-----------
  The model architectures are nearly identical (BERT base: 110M params, XLM-R base: 125M params, both 12 layers hidden=768), so the size difference alone isn't it. The likely causes are:                 
                                                        
  1. Vocabulary size — biggest structural difference                                                                                                                                                       
  - BERT: 30,522 vocab tokens → embedding matrix = 23M params                                                                                                                                              
  - XLM-R: 250,002 vocab tokens → embedding matrix = 192M params (8× larger)                                                                                                                               
                                                                                                                                                                                                           
  Even for sequence classification (no LM head), the embedding layer backward pass must update a table 8× larger during every step. More memory bandwidth → slower throughput on A100.

  2. Training steps — most likely the dominant reason

  SNLI has 549K training examples. At batch=32, 3 full epochs = ~51,500 steps. The question is: what were the BERT runs actually doing?

  - If BERT was capped at --max-steps 30000 → ~2.3 hrs at ~170ms/step
  - XLM-R running --max-steps -1 (full 51K steps) → ~7.5 hrs

  That 1.7× step ratio, combined with the vocab overhead, gets you close to 3×.

  3. Eval frequency before the fix
  If the BERT runs had already been configured with --eval_steps 500 but XLM-R runs were launched before that fix (evaluating every 50 steps), that's 10× more evaluation passes on XLM-R.

  ---
  Quick diagnostic: Check your W&B charts — what's the total step count for the BERT run vs the XLM-R run? If the XLM-R run shows ~50K steps and BERT shows ~30K, that explains most of it. If they're the
  same steps but wall-clock is still 3×, then the vocab/embedding compute is the culprit.

  What task/dataset and --max-steps were the BERT runs using when they completed in 2.5 hrs?
-----------
❯ both bert and xlmr are 50k steps (verified from wandb charts)                                                                                                                                            
                                                                
⏺ Then the per-step compute time is the issue: BERT takes ~0.18s/step, XLM-R takes ~0.54s/step. The transformer body is nearly identical (both 12 layers, hidden=768) so the difference is almost entirely 
  from the vocabulary embedding table:                                                                                                                                                                   
                                                                                                                                                                                                           
  ┌───────────────────────────────┬────────────┬─────────────┐
  │                               │    BERT    │    XLM-R    │                                                                                                                                             
  ├───────────────────────────────┼────────────┼─────────────┤
  │ Vocab size                    │ 30,522     │ 250,002     │
  ├───────────────────────────────┼────────────┼─────────────┤
  │ Embedding matrix              │ 23M params │ 192M params │
  ├───────────────────────────────┼────────────┼─────────────┤
  │ Embedding in fp16             │ ~47MB      │ ~384MB      │
  ├───────────────────────────────┼────────────┼─────────────┤
  │ Adam state (2 moment vectors) │ ~93MB      │ ~768MB      │
  └───────────────────────────────┴────────────┴─────────────┘

  The bottleneck is Adam's optimizer step on the embedding layer.

  Adam maintains dense first and second moment estimates for every vocabulary entry — not just tokens seen in the current batch. So every optimizer step must read and write 768MB of Adam state for the
  embedding table alone, regardless of how many unique tokens appeared in the batch. On an A100 with 2TB/s HBM bandwidth, writing 768MB takes ~0.4ms just for memory movement — multiplied across 50K
  steps, this adds up significantly.

  The transformer body (12 layers × attention + FFN) is the same computation. The difference is entirely this embedding overhead.

  Options to speed up future runs:

  1. Freeze the embedding layer — embeddings are pretrained and don't need fine-tuning for NLI. Add model.roberta.embeddings.requires_grad_(False) after model creation. This removes 192M params from Adam
   state entirely, likely cutting step time roughly in half:
  # in train_mrxlmr.py, after model = create_model(...)
  if args.freeze_embeddings:
      model.roberta.embeddings.requires_grad_(False)

  2. Use Adafactor instead of AdamW — Adafactor uses factored second moments, drastically reducing optimizer memory. HuggingFace Trainer supports --optim adafactor.
  3. Accept the runtime — freezing embeddings is the clean fix with no accuracy cost (XLM-R embeddings are already well-trained for multilingual NLI).

❯ implement the freeze embeddings change                                                                                                                                                                   
-----------
  To use it on the next run:
  modal run --detach train_modal.py \
    --model-type XLMR --dataset-name local_snli \
    --freeze-embeddings --max-steps -1 \
    --wandb-run-name xlmr-snli-frozen-emb

  The Trainable parameters line in the logs will drop from ~125M to ~(125M - 192M embedding) = effectively the transformer layers only, confirming the freeze took effect.
-----------