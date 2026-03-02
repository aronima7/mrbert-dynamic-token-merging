# TRAINING + EVAL RUNS

#### WANDDB
Find logs and metric charts for runs here: https://wandb.ai/aronima7-stanford-university/mrbert/table?nw=nwuseraronima7
Modal runs: https://modal.com/apps/aronima7/main
(Note: logs have 20 samples of model output)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## SNLI dataset
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### use_softmax1 = False
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. (MrBERT 3 epochs)
modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --target-deletion-rate 0.3 --mode training-and-eval --delete-gate-layer 3 --wandb-run-name run1-mrbert-deletion_rate_30pct
Time Taken For Run: 2 h 22 m
Cost For Run: $5.15
GPU: A100
Eval results:
  eval/loss: 0.5483
  eval/accuracy: 0.78
  eval/percent_deleted_tokens: 33.6186
  eval/avg_seq_len: 18.22
Analysis: The primary issue seems to be insufficient training. Run the BERT baseline for direct comparison (3 epochs), and run MrBERT for more epochs (5 epochs).
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2.
a. (MrBERT 5 epochs)
modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 5 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name run2-mrbert-5epochs
Time Taken For Run: 4 hr
Cost For Run: $8.52
GPU: A100
Eval results:                                                                                                                                                                              
  eval/loss: 0.487                                                                                                                                                                         
  eval/accuracy: 0.8141                                                                                                                                                                    
  eval/percent_deleted_tokens: 32.0694                                                                                                                                                     
  eval/avg_seq_len: 18.65  

b. (baseline BERT 3 epochs)
modal run --detach train_modal.py --model-type BERT --max-steps -1 --num-epochs 3 --mode training-and-eval --wandb-run-name run2-bert-baseline
Time Taken For Run: 2 h 22 m
Cost For Run: $5.15
GPU: A100
Eval results: 
  eval/loss: 0.3014                                                                                                                                                                        
  eval/accuracy: 0.9055                                                                                                                                                                    
  eval/percent_deleted_tokens: 0.0                                                                                                                                                         
  eval/avg_seq_len: 27.46  
Analysis: Run 2 shows the model is learning and the gate is stable. The 9.1pp gap vs BERT is partly a training duration artifact (MrBERT converges slower under deletion pressure), and partly the cost of
deletion. Run 3 (softmax1) is the immediate priority — it's the architecturally correct setting and could materially improve the result.
Once you have a good MrBERT baseline (Run 3), vary the deletion rate to plot the **accuracy-efficiency curve**. 50% deletion would give ~55% compute savings in layers 4–11 but will hit accuracy harder.
Another run to try: PI controller off. With the PI controller off, deletion_loss_weight is fixed at whatever the initial value is (default 0.01 from DEFAULT_ARGS). This means deletion pressure is constant throughout training rather than
**dynamically adjusted. The deletion rate will likely not hit 30% — it'll settle wherever the fixed α=0.01 equilibrium lands. That's the point of the ablation: it isolates how much of the gate's behavior depends on the PI controller vs. the gate learning on its own.**

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### use_softmax1 = True
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
3. 
a. (MrBERT 5 epochs; 30% target deletion rate; softmax1 = true)
modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 5 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name run3-mrbert-softmax1-5epochs
Time Taken For Run: 
Cost For Run: 
GPU: A100
Eval results: 

b. (MrBERT 5 epochs; 50% target deletion rate; softmax1 = true)
modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 5 --target-deletion-rate 0.5 --mode training-and-eval --wandb-run-name run4-mrbert-softmax1-50pct-5epochs
Time Taken For Run: 
Cost For Run: 
GPU: A100
Eval results: 

c. (MrBERT 5 epochs; no PI controller; softmax1 = true)
modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 5 --no-use-pi-controller --mode training-and-eval --wandb-run-name run5-mrbert-softmax1-no-pi-5ep
Time Taken For Run: 
Cost For Run: 
GPU: A100
Eval results: 
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
4. (MrBERT 3 epochs; 0% target deletion rate; softmax1 = true) -> **try to get the same accuracy as BERT baseline**
modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --target-deletion-rate 0.0 --mode training-and-eval --wandb-run-name run6-mrbert-softmax1-0pct-3epochs
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Refactored MrBERT
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
5. (MrBERT 3 epochs; 0% target deletion rate; softmax1 = true) -> try to get the same accuracy as BERT baseline
modal run --detach train_modal.py --model-type MrBERT --num-epochs 3 --max-steps -1 --target-deletion-rate 0.0 --mode training-and-eval --wandb-run-name run7-mrbert-softmax1-Trainer-0pct-3epochs

6. (BERT 3 epochs; softmax1 = true) -> new BERT baseline after refactoring
modal run --detach train_modal.py --model-type BERT --num-epochs 3 --max-steps -1 --mode training-and-eval --wandb-run-name run8-bert-Trainer-baseline-3epochs

7. (MrBERT 3 epochs; 30% target deletion rate; softmax1 = true) 
modal run --detach train_modal.py --model-type MrBERT --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name run9-mrbert-softmax1-Trainer-30pct-3epochs
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Refactored MrBERT -> init from pretrained BERT (not random weights/scratch); softmax1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
8. (MrBERT 3 epochs; 0% target deletion rate) -> try to get the same accuracy as BERT baseline
modal run --detach train_modal.py --model-type MrBERT --num-epochs 3 --max-steps -1 --target-deletion-rate 0.0 --mode training-and-eval --wandb-run-name run10-mrbert-Trainer-FromPretrained-0pct-3epochs

9. (BERT 3 epochs) -> new BERT baseline after metric naming changes so its easier to compare
modal run --detach train_modal.py --model-type BERT --num-epochs 3 --max-steps -1 --mode training-and-eval --wandb-run-name run11-bert-Trainer-baseline-3epochs

10. (MrBERT 3 epochs;
modal run --detach train_modal.py --model-type MrBERT --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name run12-mrbert-Trainer-FromPretrained-30pct-3epochs
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## SQUAD (Q&A) dataset
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

TODO: 
* regularization delay
* run with --batch_size 64 to stabilize training
* other tasks for BERT
* hard deletion being used at eval time?
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------