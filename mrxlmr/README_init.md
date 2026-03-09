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