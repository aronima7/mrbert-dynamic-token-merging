❯ I have a review with my advisor. Recommend a Run Plan that I can run before hand and present to her. Also recommend the best way to present all the information to her.                                  
                                                                                                                                                                           
⏺ Run Plan                                                                                                                                                                                                 
                                                                                                                                                                                                         
  Goal: Show MrBERT learns to delete tokens while maintaining competitive accuracy vs BERT baseline, across two task types.                                                                                
                                                                                                                                                                                                         
  Runs to execute (6 total)                                                                                                                                                                                
                                                                                                                                                                                                           
  ┌─────┬────────┬───────┬───────────────┬──────────────┬─────────────────────┐
  │ Run │ Model  │ Task  │ Deletion Rate │ W&B Project  │      Run Name       │
  ├─────┼────────┼───────┼───────────────┼──────────────┼─────────────────────┤
  │ A   │ BERT   │ SNLI  │ —             │ mrbert-snli  │ bert-snli-baseline  │
  ├─────┼────────┼───────┼───────────────┼──────────────┼─────────────────────┤
  │ B   │ MrBERT │ SNLI  │ 0%            │ mrbert-snli  │ mrbert-snli-0pct    │
  ├─────┼────────┼───────┼───────────────┼──────────────┼─────────────────────┤
  │ C   │ MrBERT │ SNLI  │ 30%           │ mrbert-snli  │ mrbert-snli-30pct   │
  ├─────┼────────┼───────┼───────────────┼──────────────┼─────────────────────┤
  │ D   │ BERT   │ SQuAD │ —             │ mrbert-squad │ bert-squad-baseline │
  ├─────┼────────┼───────┼───────────────┼──────────────┼─────────────────────┤
  │ E   │ MrBERT │ SQuAD │ 0%            │ mrbert-squad │ mrbert-squad-0pct   │
  ├─────┼────────┼───────┼───────────────┼──────────────┼─────────────────────┤
  │ F   │ MrBERT │ SQuAD │ 30%           │ mrbert-squad │ mrbert-squad-30pct  │
  └─────┴────────┴───────┴───────────────┴──────────────┴─────────────────────┘

  Runs B and E (0% deletion) verify that MrBERT with no deletion pressure matches the baseline — this is the sanity check your advisor will ask about. Run C and F show the accuracy/efficiency tradeoff
  from deletion.

  ---
  Modal commands

  # SNLI
  modal run --detach train_modal.py --model-type BERT --task sequence_classification --num-epochs 3 --max-steps -1 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name bert-snli-baseline

  modal run --detach train_modal.py --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.0 --mode training-and-eval --wandb-project mrbert-snli
  --wandb-run-name mrbert-snli-0pct

  modal run --detach train_modal.py --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --mode training-and-eval --wandb-project mrbert-snli
  --wandb-run-name mrbert-snli-30pct

  # SQuAD
  modal run --detach train_modal.py --model-type BERT --task question_answering --num-epochs 3 --max-steps -1 --mode training-and-eval --wandb-project mrbert-squad --wandb-run-name bert-squad-baseline

  modal run --detach train_modal.py --model-type MrBERT --task question_answering --num-epochs 3 --max-steps -1 --target-deletion-rate 0.0 --mode training-and-eval --wandb-project mrbert-squad
  --wandb-run-name mrbert-squad-0pct

  modal run --detach train_modal.py --model-type MrBERT --task question_answering --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --mode training-and-eval --wandb-project mrbert-squad
  --wandb-run-name mrbert-squad-30pct

  ---
  What to Present

  Structure (30-45 min meeting)

  1. Motivation (2 min)
  - BERT processes all tokens equally regardless of importance
  - MrT5 showed token deletion works for T5 — can we adapt this to BERT for classification and QA?

  2. Architecture (5 min)
  - Show the delete gate: after encoder layer 3, a sigmoid gate scores each token
  - Low-scoring tokens get a large negative bias added to their attention scores in all subsequent layers
  - PI controller dynamically adjusts deletion pressure to hit target rate
  - Key diagram: BERT encoder with gate inserted at layer 3

  3. Results — SNLI (5 min)

  Show these three charts side by side:
  - eval/cross_entropy_loss — all 3 runs on same chart (overfitting check)
  - Final test accuracy table:

  ┌───────────────┬───────────────┬───────────────┐
  │     Model     │ Test Accuracy │ Deletion Rate │
  ├───────────────┼───────────────┼───────────────┤
  │ BERT baseline │ ~84%          │ 0%            │
  ├───────────────┼───────────────┼───────────────┤
  │ MrBERT 0%     │ ~84%          │ 0%            │
  ├───────────────┼───────────────┼───────────────┤
  │ MrBERT 30%    │ ~??%          │ ~30%          │
  └───────────────┴───────────────┴───────────────┘

  - percent_non_pad_deleted_tokens — shows the gate actually learned to delete

  4. Results — SQuAD (5 min)

  Same structure, but metrics are EM and F1:

  ┌───────────────┬─────┬─────┬───────────────┐
  │     Model     │ EM  │ F1  │ Deletion Rate │
  ├───────────────┼─────┼─────┼───────────────┤
  │ BERT baseline │ ~%  │ ~%  │ 0%            │
  ├───────────────┼─────┼─────┼───────────────┤
  │ MrBERT 0%     │ ~%  │ ~%  │ 0%            │
  ├───────────────┼─────┼─────┼───────────────┤
  │ MrBERT 30%    │ ~%  │ ~%  │ ~30%          │
  └───────────────┴─────┴─────┴───────────────┘

  5. Deletion samples (3 min)
  - Show the printed deletion samples — qualitative evidence of what the gate learned to delete
  - Point out patterns: does it delete function words? stopwords? padding?

  6. Discussion (10 min)
  - What accuracy cost does 30% deletion impose?
  - Does the gate learn linguistically meaningful deletions?
  - What would you do next: harder deletion rates (50%), earlier gate layer, hard deletion vs soft deletion

  ---
  Presentation format

  Use a W&B Report — create one report per project (SNLI, SQuAD) with:
  - Custom charts comparing all 3 runs
  - A summary table of final test metrics
  - Embedded deletion sample text

  This is much cleaner than sharing raw dashboard links and lets your advisor see everything in one scrollable page.
---
VARIATIONS

⏺ Here are the most scientifically valuable variations to add, grouped by what question they answer:

  ---                                                                                                                                                                                                      
  Ablation: Does adaptive α (PI controller) matter?
                                                                                                                                                                                                           
  MrBERT 30% — no PI controller (fixed α = 0.01 throughout)                                                                                                                                              
  modal run --detach train_modal.py \
    --model-type MrBERT \
    --target-deletion-rate 0.3 \
    --no-use-pi-controller \
    --max-steps 30000 \
    --wandb-project mrbert-snli \
    --wandb-run-name mrbert-snli-nopi
  This shows whether the gate learns to delete at the right rate without adaptive control, or whether it collapses / ignores the deletion loss entirely.

  ---
  Ablation: Does softmax1 matter?

  MrBERT 30% — no softmax1
  modal run --detach train_modal.py \
    --model-type MrBERT \
    --target-deletion-rate 0.3 \
    --no-use-softmax1 \
    --max-steps 30000 \
    --wandb-project mrbert-snli \
    --wandb-run-name mrbert-snli-nosoftmax1
  The MrT5 paper specifically motivates softmax1 (n+1 denominator) as necessary for stable attention when tokens are deleted. This ablation tests whether it matters at the BERT scale.

  ---
  Accuracy/efficiency tradeoff curve

  MrBERT 50% deletion rate
  modal run --detach train_modal.py \
    --model-type MrBERT \
    --target-deletion-rate 0.5 \
    --max-steps 30000 \
    --wandb-project mrbert-snli \
    --wandb-run-name mrbert-snli-del50

  MrBERT 70% deletion rate
  modal run --detach train_modal.py \
    --model-type MrBERT \
    --target-deletion-rate 0.7 \
    --max-steps 30000 \
    --wandb-project mrbert-snli \
    --wandb-run-name mrbert-snli-del70
  Together with 0% and 30%, this gives you a curve: accuracy vs. tokens deleted. The key question is where the cliff is.

  ---
  Ablation: Does gate layer placement matter?

  MrBERT 30% — gate at layer 6 (middle of the network)
  modal run --detach train_modal.py \
    --model-type MrBERT \
    --target-deletion-rate 0.3 \
    --delete-gate-layer 6 \
    --max-steps 30000 \
    --wandb-project mrbert-snli \
    --wandb-run-name mrbert-snli-layer6
  Layer 3 deletes early (less context); layer 6 deletes after more processing. This tests whether the gate needs deeper representations to make good deletion decisions.

  ---
  Baseline: Random deletion (learned gate vs. random gate)

  MrBERT 30% — random gate (deletion_type=random)

  This requires passing an --extra-args style parameter. You'd need to add deletion_type to train_modal.py, or run locally:
  python train_mrbert.py \
    --task sequence_classification \
    --dataset_name local_snli \
    --local_snli_dir ./snli_datasets \
    --model_type MrBERT \
    --deletion_type random \
    --target_deletion_rate 0.3 \
    --max_steps 30000 \
    --output_dir ./mrbert_random_gate
  This is a strong sanity check: if learned deletion doesn't beat random deletion, the gate isn't learning anything meaningful.

  ---
  Priority recommendation

  For the advisor meeting, I'd prioritize in this order:

  ┌──────────┬─────────────────────────────┬─────────────────────────────────────────┐
  │ Priority │             Run             │                   Why                   │
  ├──────────┼─────────────────────────────┼─────────────────────────────────────────┤
  │ 1        │ MrBERT 50% + 70% SNLI       │ Makes the tradeoff curve, very visual   │
  ├──────────┼─────────────────────────────┼─────────────────────────────────────────┤
  │ 2        │ MrBERT 30% no PI controller │ Direct ablation of a design choice      │
  ├──────────┼─────────────────────────────┼─────────────────────────────────────────┤
  │ 3        │ MrBERT 30% layer 6          │ Tests architecture sensitivity          │
  ├──────────┼─────────────────────────────┼─────────────────────────────────────────┤
  │ 4        │ MrBERT 30% no softmax1      │ Validates a specific MrT5 design choice │
  ├──────────┼─────────────────────────────┼─────────────────────────────────────────┤
  │ 5        │ Random deletion baseline    │ Gold standard sanity check              │
  └──────────┴─────────────────────────────┴─────────────────────────────────────────┘

  The tradeoff curve (0%, 30%, 50%, 70%) is probably the single most compelling visualization — it directly answers "what are you buying with each deleted token?"
---
❯ is there a chart that would help best visualize the impact of the different target deletion percentages?                                                                                                 
                                                                                                          
⏺ The classic visualization for this is an accuracy vs. deletion rate tradeoff curve — it directly answers the core research question.                                                                     
                                                                                                                                                                                                         
  The chart                                                                                                                                                                                                
                  
  X-axis: Actual deletion rate (0%, 30%, 50%, 70%)                                                                                                                                                         
  Y-axis: Accuracy (SNLI) or F1 (SQuAD)                                                                                                                                                                    
  Lines: One for MrBERT, one horizontal dashed line for BERT baseline                                                                                                                                      
                  
  Accuracy
    89% ─────────────────────────── BERT baseline (0% deletion)
    88%        ●  MrBERT 30%
    86%                  ●  MrBERT 50%
    79%                            ●  MrBERT 70%
         0%   30%   50%   70%   deletion rate →

  The gap between the MrBERT line and the BERT baseline is the cost of compression. A shallow slope means the gate is deleting tokens cheaply; a steep cliff means it's hitting the limit.

  ---
  What makes this compelling for an advisor

  - If the curve is flat from 0→30%, that's the headline result: 30% fewer tokens, same accuracy
  - The shape of the dropoff shows where the gate breaks down
  - If you have both SNLI and SQuAD, two lines on the same chart show whether task complexity affects the tradeoff

  ---
  Generating it

  Once you have all four MrBERT runs done, here's a script to produce the chart:

⏺ Searched for 1 pattern, read 1 file (ctrl+o to expand)     
                                       
⏺ You already have exactly the right chart in compute_savings.py — it's Figure 3: accuracy vs compute (accuracy_vs_compute.pdf).
                                                   
  The --runs flag is how you feed in results from each run. Once your Modal runs finish, you pull the final test accuracy from W&B and run:                                                                
                                                                                                                                                                                                           
  python analysis/compute_savings.py \                                                                                                                                                                     
    --runs \                                                                                                                                                                                               
      "BERT,0.906,0.0" \
      "MrBERT-30%,0.XXX,0.30" \
      "MrBERT-50%,0.XXX,0.50" \
      "MrBERT-70%,0.XXX,0.70"

  This produces a scatter plot where:
  - X-axis = relative compute vs BERT (derived from actual deletion rate, not target)
  - Y-axis = accuracy %
  - Each run is a labelled dot
  - The background curve shows the theoretical efficiency envelope

  The chart directly answers "is MrBERT on the Pareto frontier?" — i.e., for a given compute budget, is MrBERT as accurate as possible?

  The --seq_len argument adjusts the MACs calculation — use your actual mean post-deletion sequence length from the logs (printed as new_seq_len during training). The default of 27 is tuned for SNLI at
  30% deletion.
---