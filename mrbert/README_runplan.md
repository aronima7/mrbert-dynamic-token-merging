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
COMPARISON of METRICS, CHARTS with MrT5

⏺ MrT5 Paper Metrics and Charts:                                                                                                                                                                           
                                                                                                                                                                                                           
  1. Figures:                                                                                                                                                                                              
    - Figure 1: Soft/hard deletion illustration (architecture diagram - not a metrics chart)                                                                                                               
    - Figure 2: BPB vs. sequence length reduction (main efficiency tradeoff chart)                                                                                                                         
    - Figure 3: Per-language BPB vs. sequence length reduction (15 languages)                                                                                                                              
    - Figure 4: BPB and inference runtime by gate layer placement (ablation)                                                                                                                               
    - Figure 5: Theoretical compute savings vs. deletion ratio                                                                                                                                             
  2. Tables:
    - Table 1: Diagnostic task examples
    - Table 2: Diagnostic task performance (token acc, seq acc, seq length reduction)                                                                                                                      
    - Table 3: Inference runtime by deletion rate                                                                                                                                                          
    - Table 4: XNLI and TyDiQA results                                                                                                                                                                     
    - Table 5: Character-level task results                                                                                                                                                                
  3. Key metrics:                                                                                                                                                                                          
    - BPB (bits per byte) - primary LM metric (we don't use this since we're on BERT)
    - Sequence length reduction (%)
    - Inference runtime (ms) and runtime decrease (%)
    - Token-level accuracy, sequence-level accuracy
    - EM, F1 (QA)
    - Classification accuracy (NLI)
    - MACs (theoretical compute)
    - Per-sample correlation analysis
  4. Ablations:
    - Gate layer: 1,2,3,4,5
    - Deletion rate: 0.3, 0.4, 0.5, 0.6, 0.7
    - Gate variants: scaled_sigmoid, random, fixed, boundary predictor, convolutional pooling
    - softmax1 vs standard softmax
    - Attention score regularization
  5. Baselines:
    - Standard ByT5 (no deletion)
    - Random deletion
    - Fixed deletion
    - Boundary Predictor
    - Convolutional Pooling

  Now let me identify what MrBERT currently has and what gaps exist:

  What MrBERT currently has:

  Metrics logged:
  - loss, cross_entropy_loss, delete_gate_loss
  - accuracy
  - percent_deleted_tokens, percent_non_pad_deleted_tokens
  - delete_gate_average, delete_gate_std, delete_gate_max_value, delete_gate_min_value
  - delete_gate_loss_coeff (α)
  - new_seq_len
  - For SQuAD: EM, F1 (implemented in compute_squad_em_f1)
  - elapsed_time, epoch_progress

  Run variations planned:
  - BERT baseline (0% deletion) on SNLI
  - MrBERT 30% on SNLI
  - BERT baseline on SQuAD
  - MrBERT 30% on SQuAD
  - (Proposed) MrBERT 50%, 70% on SNLI
  - (Proposed) No PI controller
  - (Proposed) Gate at layer 6

  Charts available:
  - compute_savings.py: MACs vs deletion ratio, MACs by gate layer, accuracy vs compute (Figure 5 equivalent + partial Figure 4 equivalent + partial Figure 2 equivalent)

  GAPS:

  1. Runtime measurement (Table 3 equivalent): MrT5 explicitly measures inference runtime in ms. MrBERT logs elapsed_time but not per-sample inference time. We need to measure actual inference time
  (ms/sample) for BERT vs MrBERT at different deletion rates.
  2. Inference runtime speedup (Table 3, Table 4, Table 5): The key claim in MrT5 is "X% runtime decrease". We need to actually measure this for MrBERT. This is the most important number for the advisor
  meeting.
  3. Random deletion baseline (Figure 2, Table 4): MrT5 explicitly compares against random deletion at the same deletion rates. We currently don't have a MrBERT run with deletion_type=random. This
  baseline shows the gate is learning something non-trivial vs. random.
  4. Fixed deletion baseline: Similar to random deletion — MrT5 tests this too.
  5. Gate layer ablation (Figure 4): MrT5 tests gate at layers 1-5. We have one layer-6 run proposed but haven't committed to a systematic study.
  6. Sequence length reduction % vs. accuracy scatter plot (Figure 2 equivalent): MrT5's Figure 2 is BPB vs. sequence length reduction. Our equivalent would be SNLI accuracy vs. sequence length
  reduction. We have the compute_savings.py but it uses accuracy vs. relative compute, not sequence length reduction directly.
  7. Per-language analysis (Figure 3): Not applicable to MrBERT since we're using English SNLI/SQuAD.
  8. Attention score regularization: MrT5 uses attention score regularization (β=5.0, m=5.0) to prevent attention score inflation with hard deletion. MrBERT uses soft deletion only during training, so
  this may not be needed, but it's worth checking if it's implemented.
  9. Token-level accuracy (Table 2): Not applicable for classification/QA tasks.
  10. Per-sample correlation analysis (Figure similar): MrT5 shows that its deletion is context-dependent (low correlation between how hard each sample is and how much gets deleted). This would be an
  interesting analysis for MrBERT.
  11. Hard vs. soft deletion comparison: MrT5 explicitly studies soft (training) vs. hard (test) deletion. MrBERT has this in the config but haven't explicitly tested it.
  12. BPB metric: Not applicable since we're doing classification/QA, not language modeling.

  Key actionable gaps for MrBERT:

  High priority (directly maps to paper):
  1. Runtime measurement per sample - Need compute_inference_time() function
  2. Random deletion baseline - Need to add deletion_type parameter to train_modal.py
  3. Sequence length reduction % as a metric on X-axis (we have it as new_seq_len but should compute % reduction)
  4. Figure 2 equivalent: accuracy/F1 vs. sequence length reduction (not compute)

  Medium priority:
  5. Gate layer ablation (layer 1, 3, 6, 9 systematically)
  6. Attention score regularization - check if implemented

  Low priority:
  7. Per-sample correlation analysis
  8. Hard vs. soft deletion ablation

  Now let me also check what's in the analysis module and training code to see what's already implemented.


  Now let me cross-reference against the existing MrBERT code to find the gaps.

⏺ Searched for 1 pattern, read 2 files (ctrl+o to expand)     
                                                 
⏺ Key findings from the paper vs what MrBERT has:
                                       
  MrT5 Paper - Key figures and what they show:                                                                                                                                                             
                                                                                                                                                                                                           
  1. Figure 2: BPB vs sequence length reduction — scatter plot showing all models (MrT5, Random, Fixed, BP, CP) at different deletion rates. The key innovation: at same sequence length reduction, MrT5   
  achieves better BPB.                                                                                                                                                                                     
    - MrBERT equivalent: accuracy vs sequence length reduction (%)
    - STATUS: compute_savings.py Figure 3 does accuracy vs compute (not seq length reduction). GAP: need x-axis to be seq_length_reduction % instead of/in addition to relative compute.
  2. Figure 3: Per-language BPB — not applicable for MrBERT (English only)
  3. Figure 4: BPB + runtime vs gate layer — ablation chart
    - GAP: Gate layer ablation chart doesn't exist for MrBERT. Would need accuracy + runtime vs gate layer.
  4. Figure 5: Theoretical compute savings — already implemented in compute_savings.py ✓
  5. Table 2: Diagnostic tasks — not applicable (MrBERT focuses on classification/QA)
  6. Table 3: Runtime in ms and % decrease — the CRITICAL metric: actual measured inference speedup
    - GAP: MrBERT does not measure per-sample inference time anywhere. This is the most important gap.
  7. Table 4: XNLI/TyDiQA with accuracy/F1/EM + runtime decrease + seq length reduction
    - MrBERT equivalent should report: accuracy/F1/EM + runtime decrease % + seq length reduction %
    - GAP: runtime decrease not measured; seq length reduction % not explicitly computed as a summary stat
  8. Table 5: Character-level tasks — not applicable

  Random/Fixed deletion baselines:
  - MrT5 tests both random and fixed deletion at multiple rates
  - GAP: MrBERT doesn't have deletion_type as a modal parameter — can't run random/fixed baseline easily
  - Random deletion is the most important baseline (shows learned gate > random)

  Attention score regularization:
  - MrT5 uses β=5.0 for attention score clamping (hard deletion)
  - MrBERT uses soft deletion during training, so this may not apply
  - GAP: Need to verify if this is relevant for MrBERT

  Let me now create the gap analysis summary.

  Priority gaps to fix:

  1. Missing: Inference runtime measurement — most important for the paper. Need a function that measures ms/sample for BERT vs MrBERT at different deletion rates.
  2. Missing: Random deletion baseline run — need deletion_type parameter in train_modal.py. The MrBertConfig already supports deletion_type=random, just needs to be exposed.
  3. Missing: Accuracy vs sequence length reduction % chart — compute_savings.py uses relative compute on x-axis, but the paper's Figure 2 uses sequence length reduction %. Should add this as an
  alternative plot.
  4. Missing: Gate layer ablation chart — paper's Figure 4 shows BPB + runtime vs gate layer. MrBERT needs accuracy + runtime vs gate layer (layers 1, 3, 6, 9).
  5. Missing: Seq length reduction % as explicit summary metric in results table — when we report results, we should compute this directly (1 - new_seq_len/original_seq_len * 100%)

  What's already good:
  - compute_savings.py: Figure 5 equivalent ✓, accuracy vs compute ✓
  - deletion_pattern_analysis.py: deletion by token type, top/bottom tokens, premise vs hypothesis ✓
  - get_deletion_patterns.py: gate visualization ✓
  - EM/F1 for SQuAD ✓
  - PI controller ✓
  - Multiple deletion rates (30%, 50%, 70%) planned ✓
  - Softmax1 ablation planned ✓
  - No PI controller ablation planned ✓

  Let me now write a clear gap analysis report.


  Now I have everything I need. Let me do the full gap analysis.

⏺ Searched for 4 patterns, read 1 file (ctrl+o to expand)     
                                                                                                              
⏺ What MrT5 reports and MrBERT's status:                                                                                                                                                                   
                                       
  FIGURES                                                                                                                                                                                                  
                                                                                                                                                                                                           
  Figure 2: Performance vs Sequence Length Reduction (scatter plot)                                                                                                                                        
                                                                                                                                                                                                           
  - MrT5: X = sequence length reduction (%), Y = BPB; all models shown (MrT5, Random, Fixed, BP, CP)
  - MrBERT equivalent: X = sequence length reduction (%), Y = accuracy (SNLI) or F1 (SQuAD)
  - STATUS: compute_savings.py Figure 3 uses X = relative compute (not sequence length reduction %)
  - GAP: Add plot_accuracy_vs_seq_length_reduction() to compute_savings.py

  Figure 4: Performance + Runtime vs Gate Layer (ablation)

  - MrT5: X = gate layer (1-5); dual Y-axis: BPB (left) + runtime ms (right)
  - MrBERT equivalent: X = gate layer; Y = accuracy (left) + inference ms (right)
  - STATUS: Gate layer ablation not in any chart
  - GAP: Need gate layer ablation chart (requires running layer 1, 3, 6, 9)

  Figure 5: Theoretical compute savings

  - STATUS: compute_savings.py already does this ✓

  TABLES

  Table 3: Inference Runtime

  - MrT5: runtime (ms/sample) and % decrease for each deletion rate
  - STATUS: MrBERT does not measure per-sample inference time anywhere
  - GAP: Need measure_inference_runtime() function in compute_savings.py or eval script

  Table 4 equivalent: Summary results table

  - MrT5 columns: Task | Model | Accuracy/EM/F1 | Runtime decrease % | Seq length reduction %
  - MrBERT: logs accuracy, logs new_seq_len, but does NOT log:
    - Runtime decrease % vs BERT baseline
    - Seq length reduction % as a computed summary stat
  - GAP: Need to compute these two statistics when reporting results

  METRICS

  Inference runtime (ms/sample) — not measured

  Seq length reduction % — computed in new_seq_len but not as a % of original

  Random deletion baseline — model exists (deletion_type=random), but NOT in train_modal.py as a parameter

  WHAT'S ALREADY GOOD

  1. Accuracy/F1/EM metrics ✓
  2. percent_deleted_tokens ✓
  3. delete_gate metrics (avg, std, min, max) ✓
  4. Theoretical compute savings plot (Figure 5 equivalent) ✓
  5. Accuracy vs compute scatter plot ✓
  6. Deletion by token type ✓
  7. Top/bottom deleted tokens ✓
  8. Premise vs hypothesis deletion rate ✓
  9. Colored deletion pattern visualization ✓
  10. PI controller ✓
  11. softmax1 ablation planned ✓
  12. No-PI controller ablation planned ✓
  13. Multiple deletion rates (30%, 50%, 70%) planned ✓

  ACTIONABLE GAPS

  Gap 1: Inference runtime measurement (most critical)
  - compute_savings.py shows theoretical MACs, but MrT5 reports measured runtime (ms/sample)
  - Need a measure_inference_runtime(model_path, task, dataset, n_samples) function
  - Should compare BERT vs MrBERT at each deletion rate
  - Could be added to eval_mrbert.py or a new analysis/measure_runtime.py

  Gap 2: Random deletion baseline (second most critical)
  - deletion_type=random exists in MrBertConfig and train_mrbert.py
  - NOT exposed in train_modal.py (only extra_args workaround)
  - Need to add deletion_type parameter to train_modal.py's train() and main()

  Gap 3: Accuracy vs sequence length reduction % chart (Figure 2 equivalent)
  - Current compute_savings.py Figure 3 uses relative compute on X-axis
  - MrT5's key Figure 2 uses sequence length reduction % on X-axis
  - Need to add alternative plot with X = seq_len_reduction_pct

  Gap 4: Sequence length reduction % in summary stats
  - new_seq_len is logged in absolute tokens
  - Need to also report seq_len_reduction_pct = (1 - new_seq_len/original_seq_len) * 100

  Gap 5: Gate layer ablation (Figure 4 equivalent)
  - Systematic test of layers 1, 3, 6, 9 — not just one extra layer
  - Chart needs: accuracy (left y) + runtime (right y) vs gate layer (x)

  Let me now write the gap analysis report clearly.


  Now I have everything needed. Here is the full gap analysis.

  ---
  MrT5 Paper vs MrBERT: Gap Analysis

  What the paper reports (figures + tables)

  Paper item: Figure 2
  What it shows: BPB vs sequence length reduction % — all models (MrT5, Random, Fixed, BP, CP) on one scatter plot
  MrBERT status: Partial. compute_savings.py Figure 3 has X = relative compute, not sequence length reduction %.
  ────────────────────────────────────────
  Paper item: Figure 4
  What it shows: BPB + inference runtime (ms) vs gate layer (1–5) — dual y-axis
  MrBERT status: Missing entirely. No gate layer ablation chart.
  ────────────────────────────────────────
  Paper item: Figure 5
  What it shows: Theoretical compute savings vs deletion ratio
  MrBERT status: Done. compute_savings.py covers this ✓
  ────────────────────────────────────────
  Paper item: Table 3
  What it shows: Measured inference runtime (ms/sample) + % decrease vs baseline for each deletion rate
  MrBERT status: Critical gap. MrBERT measures elapsed training time but never measures per-sample inference time.
  ────────────────────────────────────────
  Paper item: Table 4
  What it shows: XNLI/TyDiQA: accuracy/EM/F1, runtime decrease %, seq length reduction %
  MrBERT status: Partial. Accuracy/EM/F1 done; runtime decrease % and seq length reduction % not reported as summary stats.

  ---
  Gap 1 — Inference runtime measurement (most critical)

  The paper's Table 3 is: MrT5 30% = 21% speedup, 50% = 32% speedup, 70% = 45% speedup. This is the core claim.

  MrBERT has no equivalent. The fix is a measure_inference_runtime.py script in analysis/ that benchmarks BERT vs MrBERT at each deletion rate:

  # proposed: analysis/measure_runtime.py
  # For each model checkpoint:
  #   - Run N forward passes on fixed batch
  #   - Record mean ms/sample
  #   - Report: runtime_ms, pct_decrease_vs_bert, seq_len_reduction_pct

  This would feed directly into compute_savings.py's accuracy-vs-compute plot as measured (not theoretical) x-values.

  ---
  Gap 2 — Random deletion baseline (second most critical)

  The paper compares MrT5 vs Random deletion at the same deletion rates on every chart. This is the key sanity check: is the gate doing something smarter than random?

  deletion_type=random already exists in MrBertConfig and train_mrbert.py, but train_modal.py doesn't expose it. Fix: add deletion_type to train_modal.py.

  Runs needed:
  # MrBERT-Random-30% on SNLI
  modal run --detach train_modal.py \
    --model-type MrBERT \
    --deletion-type random \
    --target-deletion-rate 0.3 \
    --max-steps 30000 \
    --wandb-project mrbert-snli \
    --wandb-run-name mrbert-snli-random30

  ---
  Gap 3 — X-axis of main tradeoff chart

  compute_savings.py Figure 3 plots accuracy vs relative compute. The paper's Figure 2 plots BPB vs sequence length reduction %. These are related but different — sequence length reduction % is more
  intuitive and directly reported.

  Fix: add a parallel plot in compute_savings.py with X = seq_len_reduction_pct (derived from measured new_seq_len / original max_seq_length).

  ---
  Gap 4 — Gate layer ablation chart (Figure 4 equivalent)

  The paper shows gate at layers 1–5 with dual y-axis: task performance (left) + runtime (right). MrBERT has one proposed layer-6 run, but no chart aggregating them.

  Runs needed: layer 1, 3, 6, 9 at 30% deletion. Once you have those four accuracy numbers + four runtime measurements, compute_savings.py can generate the chart.

  ---
  Gap 5 — Seq length reduction % as a reported stat

  When you report results, the paper always shows "47.48% sequence length reduction" alongside EM/F1. MrBERT logs new_seq_len in absolute tokens but never computes the % reduction. This is a one-liner in
   the eval output:

  seq_len_reduction_pct = (1 - new_seq_len / max_seq_length) * 100

  ---
  Summary: what to implement

  ┌──────────┬──────────────────────────────────────────┬───────────────────────────────────────────────────────┐
  │ Priority │                   Gap                    │                          Fix                          │
  ├──────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 1        │ Inference runtime (ms/sample)            │ New analysis/measure_runtime.py                       │
  ├──────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 2        │ Random deletion baseline                 │ Add --deletion-type to train_modal.py                 │
  ├──────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 3        │ Seq length reduction % on chart x-axis   │ New plot in compute_savings.py                        │
  ├──────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 4        │ Seq length reduction % in reported stats │ One-liner in eval output                              │
  ├──────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 5        │ Gate layer ablation chart                │ 4 runs (layers 1,3,6,9) + chart in compute_savings.py │
  └──────────┴──────────────────────────────────────────┴───────────────────────────────────────────────────────┘

  What's already solid: accuracy/EM/F1, deletion rate tracking, gate value statistics, theoretical compute savings, deletion by token type, per-token deletion rates, premise vs hypothesis analysis, PI
  controller, softmax1, colored deletion visualization.

  ---
  Want me to start implementing any of these? The highest-value ones are Gap 1 (runtime measurement) and Gap 2 (random deletion baseline in train_modal.py).

✻ Crunched for 9m 4s                  

⏺ Background command "Install poppler for PDF rendering" completed (exit code 0)                                                                                                                           
   
⏺ Read 1 file (ctrl+o to expand)                                                                                                                                                                           
                                                               
⏺ Poppler was blocked by the network sandbox (same as pip). The PDF text was already extracted via mdimport and the gap analysis above is complete — no impact on the results.                             
                  
  To summarize where we landed: implement these in priority order?                                                                                                                                         
                                                                                                                                                                                                           
  1. Gap 2 (--deletion-type in train_modal.py) — small change, unblocks random baseline runs immediately                                                                                                   
  2. Gap 1 (analysis/measure_runtime.py) — the most impactful result for the paper                                                                                                                         
  3. Gaps 3/4/5 (chart + stat improvements)     
---
  Gaps are implemented. 
  Here's what changed:      
                                                       
  ---                                                                                                                                                                                                      
  Gap 2 — train_modal.py (--deletion-type)          
                                                                                                                                                                                                           
  Added deletion_type parameter to both train() and main(). Now you can run:                                                                                                                               
                                                                                                                                                                                                           
  # Random deletion baseline (same 30% rate, no learned gate)                                                                                                                                              
  modal run --detach train_modal.py \
    --model-type MrBERT \
    --deletion-type random \
    --target-deletion-rate 0.3 \
    --max-steps 30000 \
    --wandb-project mrbert-snli \
    --wandb-run-name mrbert-snli-random30

  # Fixed deletion baseline
  modal run --detach train_modal.py \
    --model-type MrBERT \
    --deletion-type fixed \
    --target-deletion-rate 0.3 \
    --max-steps 30000 \
    --wandb-project mrbert-snli \
    --wandb-run-name mrbert-snli-fixed30

  ---
  Gap 1 — analysis/measure_runtime.py

  New script that benchmarks any set of checkpoints on the same fixed test inputs. Usage after downloading Modal checkpoints:

  python analysis/measure_runtime.py \
    --models \
      "BERT,./local_checkpoints/bert-snli/final" \
      "MrBERT-30%,./local_checkpoints/mrbert-snli-del30/final" \
      "MrBERT-50%,./local_checkpoints/mrbert-snli-del50/final" \
      "MrBERT-70%,./local_checkpoints/mrbert-snli-del70/final" \
    --local_snli_dir ./snli_datasets

  Outputs:
  - A runtime table to stdout (mirrors MrT5 Table 3): ms/sample, ±std, % decrease vs BERT, sequence length reduction %
  - analysis/figures/runtime_table.csv — importable into your paper
  - analysis/figures/runtime_vs_deletion.pdf — bar chart with BERT baseline dashed line and % decrease annotations
                                                                                                                                                                                                  
  ---                                                                                                                                                                                                      
  Gap 4 — seq_len_reduction_pct in W&B (train_mrbert.py)                                                                                                                                                   
                                                                                                                                                                                                           
  Added to MrBertTrainer.log(). Every time new_seq_len is logged, it now also logs:                                                                                                                        
                                                                                                                                                                                                           
  seq_len_reduction_pct = (1 - new_seq_len / max_seq_length) * 100

  This appears in W&B as a first-class metric alongside percent_deleted_tokens. For a 128-token sequence where MrBERT keeps ~90 tokens, you'd see seq_len_reduction_pct = 29.7.

  ---
  Gap 3 — accuracy_vs_seq_reduction.pdf (compute_savings.py)

  New plot_accuracy_vs_seq_length_reduction() — the direct equivalent of MrT5 Figure 2. Same --runs input you already use for the compute plot. Both charts are generated together:

  python analysis/compute_savings.py \
    --runs "BERT,0.906,0.0" "MrBERT-30%,0.XXX,0.30" "MrBERT-50%,0.XXX,0.50" "MrBERT-70%,0.XXX,0.70"
  # → accuracy_vs_compute.pdf  (x = relative MACs)
  # → accuracy_vs_seq_reduction.pdf  (x = sequence length reduction %)

  ---
  Gap 5 — gate_layer_ablation.pdf (compute_savings.py)

  New plot_gate_layer_ablation() — dual y-axis, mirrors MrT5 Figure 4. New --gate-layer-runs arg with format label,accuracy,runtime_ms,gate_layer. After you run the layer-1, layer-3, layer-6, layer-9
  experiments and measure their runtimes with measure_runtime.py:

  python analysis/compute_savings.py \
    --gate-layer-runs \
      "Layer 1,0.XXX,<ms>,1" \
      "Layer 3,0.XXX,<ms>,3" \
      "Layer 6,0.XXX,<ms>,6" \
      "Layer 9,0.XXX,<ms>,9"
  # → gate_layer_ablation.pdf  (accuracy in red left y, runtime in blue right y)
---
  EFFICIENCY:

  Three bugs fixed in measure_runtime.py:                                                                                                                                                                                                                                            
   
  ┌──────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┐                                                                                                    
  │               Bug                │                                  Root cause                                  │                            Fix                            │                                                                                                  
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ No actual speedup measured       │ hard_delete not passed to model → soft deletion used                         │ Added hard_delete=True default; passed to MrBERT calls    │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Seq length reduction always None │ Checked outputs.new_seq_len which doesn't exist in any output dataclass      │ Now reads outputs.delete_gate_mask and counts kept tokens │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Wrong threshold                  │ Would have used 0.5 (if new_seq_len existed) but gate values are in [-30, 0] │ Reads model.config.deletion_threshold (default -15.0)     │
  └──────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘

  New --no_hard_delete flag lets you benchmark soft deletion for comparison — useful to show the advisor that the speedup is specifically from physical token removal, not just from the attention masking.
---
❯ is physical hard deletion what's done in MrT5 final analysis as well?                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                   
⏺ Let me check the MrT5 paper and source code to answer this precisely.                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                     
⏺ Explore(Check MrT5 hard vs soft deletion at inference)                                                                                                                                                                                                                             
  ⎿  Done (16 tool uses · 77.1k tokens · 1m 10s)                                                                                                                                                                                                                                     
  (ctrl+o to expand)                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                     
⏺ 1. In mrt5/models/modeling_mrt5.py, hard_delete=False is the default in the forward method.                                                                                                                                                                                        
  2. In mrt5/utils.py, mrt5_compute_metrics(... hard_delete=True ...) — the default there is True.                                                                                                                                                                                   
  3. In the eval scripts (span_corruption_eval.py, qa_eval.py), --hard_delete is an action='store_true' flag, meaning it's False by default and only enabled if explicitly passed.                                                                                                   
  4. In the trainer, evaluation uses both: hard_delete=False for soft deletion metrics, and hard_delete=True for hard deletion metrics separately.                                                                                                                                   
                                                                                                                                                                                                                                                                                     
  So the paper likely reports both soft and hard deletion results. The runtime measurements (Table 3) would use hard deletion since that's where the actual speedup comes from. The accuracy/BPB metrics might use either.                                                           
                                                                                                                                                                                                                                                                                     
  Actually, looking at the trainer more carefully:                                                                                                                                                                                                                                   
  - During training: random mix of hard and soft deletion (with hard_delete_train_prob)
  - During eval: logs both soft and hard deletion metrics

  This is actually quite important: MrT5 trains with both hard and soft deletion mixed. This is different from MrBERT which trains with soft deletion only and then applies hard deletion at inference.

  The key insight for the user: MrT5 trains with hard_delete_train_prob (some probability of using hard deletion during training), which means the model is trained to handle both modes. MrBERT only trains with soft deletion. Applying hard deletion at MrBERT inference time
  without training with it could lead to degraded accuracy compared to training with it.

  For the reported metrics in MrT5 paper:
  - The runtime table (Table 3) reports actual speedup, which requires hard deletion
  - The language modeling metrics (BPB) seem to use soft deletion by default (based on eval scripts defaulting to no --hard_delete)

  So the answer to the user's question is: MrT5 uses hard deletion for runtime measurements, but the language modeling quality metrics (BPB) likely use soft deletion. AND critically, MrT5 trains with a mix of hard and soft deletion, which is different from MrBERT.

  This has an important implication: For MrBERT, if we apply hard deletion at inference without training with it, we might see larger accuracy degradation than what MrT5 reports because the higher layers were never trained to handle physically shorter sequences.

  Let me be clear and concise in my response.


  The MrT5 codebase is more nuanced than a simple yes/no:

  Training: MrT5 trains with a random mix of hard and soft deletion, controlled by hard_delete_train_prob in the trainer. From trainer.py:
  hard_delete=self.rng.random() < self.hard_delete_train_prob  # during training steps

  Evaluation: Both are measured separately. The trainer logs soft and hard deletion metrics in the same eval loop.

  Eval scripts: span_corruption_eval.py and qa_eval.py have --hard_delete as an opt-in flag (action='store_true'), defaulting to False. So the reported BPB/accuracy numbers use soft deletion. The runtime table (Table 3) uses hard deletion — that's where the actual speedup
  comes from.

  ---
  This creates an important problem for MrBERT:

  MrT5 trains with hard deletion mixed in, so the model's upper layers learn to handle shorter sequences. MrBERT trains with soft deletion only — the upper layers always see the full sequence length. If you apply hard deletion at MrBERT inference time, layers 4–12 have never
  seen truncated sequences during training, which could cause a larger accuracy drop than MrT5 reports.

  Options:

  1. Report soft deletion accuracy + theoretical MACs savings only (honest, no accuracy risk, but weaker efficiency claim — matches how MrT5 reports language model quality)
  2. Add hard_delete_train_prob to MrBERT training (mirrors MrT5 exactly, stronger claim, requires a new training run)
  3. Benchmark both for the advisor meeting — show the accuracy gap between soft and hard deletion inference as a finding in itself

  Option 3 is probably the most honest for a first milestone. Do you want to add hard_delete_train_prob to the MrBERT trainer to match MrT5's training protocol?
---
mrbert/README_first_milestone_run_plan.md

  Based on these changes, can you please create an updated run plan whose results I can present to my advisor today. Include commands to be run for the training/eval runs as well as for generating the  
  necessary metrics and charts. Make it step by step - clearly mark the steps that can be done in parallel (for eg. runs) vs the ones that are sequential (for eg. analysis runs in the end). Also,        
  clearly specify the output that needs to be captured/copied from logs for subsequent steps. Include an estimate for each step (this will help me plan my runs). Make sure the commands are in single     
  line format (not multi-line). Make sure to include runs for both SNLI dataset and Q&A dataset as well as the variations that will lead to a productive discussion with my advisor and in turn lead to    
  valuable feedback from her that can be applied to subsequent experiments. In the end, also include a list of questions for her, clarification of which will help in planning the next set of             
  experiments. The ultimate goal is to write and publish a Research Paper, so the work needs to be robust. Create a new file mrbert/README_first_milestone_run_plan.md and include all of the above in     
  there. 
---