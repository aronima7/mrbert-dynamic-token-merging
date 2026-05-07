  MrDiffusion-SEDD: Architecture Overview

  Core Concept: Adapts MrT5/MrBERT's delete gate to SEDD (Score Entropy Discrete Diffusion). After a specified transformer block, a learned gate compresses the sequence to reduce computation
  during denoising.

  ---
  Model Structure (modeling_mrdiffusion.py)

  MrSEDD (line 533) wraps standard SEDD with three phases per forward pass:

  1. Phase 1 (blocks 0..delete_gate_layer): Normal FlashAttention on full sequence [B, L, D]
  2. Gate fires (after block delete_gate_layer): Assigns each token a scalar in [sigmoid_mask_scale, 0] (e.g. [-30, 0])
  3. Phase 2 (compressed blocks): Soft or hard deletion
  4. Restoration (at restore_gate_layer): Full [B, L, D] reassembled
  5. Phase 3 (final blocks): Full sequence again

  ---
  Two Deletion Modes

  Soft (default): Gate values become additive attention biases ([-30, 0]) in subsequent blocks. Tokens physically remain but are nearly ignored. Requires falling back from FlashAttention →
  F.scaled_dot_product_attention since FlashAttn doesn't support attention bias.

  Hard: Tokens where gate < deletion_threshold are physically removed. Sequence compresses to [B, L_kept, D]. _restore_hidden_states() (line 624) scatters survivors back; deleted positions get
  filled with pre-gate hidden states (not zeros — critical for valid diffusion loss gradients).

  ---
  Gate Variants (lines 101-288)

  ┌────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │                Type                │                                  Description                                   │
  ├────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ SigmoidDeleteGate                  │ LayerNorm → Linear → ScaledSigmoid; bias init=2.0 (starts in "keep all" state) │
  ├────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ SigmoidDeleteGateWithSigma         │ Same + concatenates σ embedding (recommended, gate_sigma_conditioned=True)     │
  ├────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ BottleneckDeleteGate               │ 2-layer MLP with Gumbel-sigmoid + temperature annealing                        │
  ├────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ RandomDeleteGate / FixedDeleteGate │ Non-learned baselines                                                          │
  └────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

  Sigma conditioning (gate_sigma_conditioned=True) is crucial: at high noise levels most tokens are corrupted/redundant → gate can delete more freely; at low noise every token matters → gate
  should be conservative.

  ---
  Loss Function (losses_mrdiffusion.py)

  total_loss = score_entropy_loss           # standard SEDD loss
             + deletion_loss_weight * MSE(actual_rate, target_rate(σ))
             + gate_logit_reg_weight * ||gate_logits||²

  Sigma-dependent deletion schedule (line 34-62): target rate scales with noise level:
  - constant: fixed rate (e.g. 30%)
  - linear_sigma / power_sigma: ramps from r_min=0.05 (clean) → r_max=0.5 (fully noised)

  ---
  Key Design Decisions

  1. Restoration uses pre-gate states, not zeros (line 644): deleted positions need valid hidden states so diffusion loss doesn't explode on those positions
  2. Bias init=2.0 on gate (line 115): ensures gate starts near "keep all", avoiding training instability early on
  3. Pre-deletion blend (line 291): before hard deletion, interpolates each deleted token into its next survivor — reduces information loss
  4. Hard deletion pads to min_kept (line 610): equalizes batch tensor shapes by finding minimum survivors across batch

  ---
  Config Highlights (configuration_mrdiffusion.py)

  ┌────────────────────────┬──────────┬─────────────────────────────────────────┐
  │         Param          │ Default  │                  Role                   │
  ├────────────────────────┼──────────┼─────────────────────────────────────────┤
  │ delete_gate_layer      │ 3        │ Which block fires the gate              │
  ├────────────────────────┼──────────┼─────────────────────────────────────────┤
  │ restore_gate_layer     │ None     │ When to restore (None = before output)  │
  ├────────────────────────┼──────────┼─────────────────────────────────────────┤
  │ sigmoid_mask_scale     │ -30.0    │ Gate output range [scale, 0]            │
  ├────────────────────────┼──────────┼─────────────────────────────────────────┤
  │ deletion_mode          │ soft     │ soft or hard                            │
  ├────────────────────────┼──────────┼─────────────────────────────────────────┤
  │ gate_sigma_conditioned │ True     │ Gate sees σ as additional input         │
  ├────────────────────────┼──────────┼─────────────────────────────────────────┤
  │ deletion_rate_schedule │ constant │ How target rate varies with noise level │
  ├────────────────────────┼──────────┼─────────────────────────────────────────┤
  │ deletion_loss_weight   │ 0.1      │ Auxiliary loss trade-off                │
  └────────────────────────┴──────────┴─────────────────────────────────────────┘

  ---
  Structural Difference from Base SEDD

  Base SEDD just runs all DDiTBlocks sequentially on [B, L, D]. MrSEDD replaces all blocks with MrDDiTBlock (which supports attention masking) and inserts gate logic mid-forward — everything
  else (score entropy loss, noise/graph framework, sampling) stays identical.

  The main unsolved challenge (per README): SEDD corrupts tokens to random vocabulary tokens (not [MASK]), making it hard for the gate to learn what's redundant. The authors note BD3-LM (uniform
   [MASK] corruption) would be a better base for this approach.
  ---
  REGULARIZATION
  There are three regularization mechanisms:

  1. Gate Logit L2 Regularization (losses_mrdiffusion.py:157-161)

  gate_logit_reg_weight = getattr(mr_config, "gate_logit_reg_weight", 0.001)
  logit_reg = gate_logit_reg_weight * gate_logits_out.pow(2).mean()
  total_loss = total_loss + logit_reg
  Penalizes large gate logit magnitudes. Prevents the gate from collapsing to extreme "delete all" or "keep all" states. Default weight: 0.001.

  2. Deletion Rate Loss (losses_mrdiffusion.py:149-155)

  target_rates = target_rate_at_sigma(sigma, mr_config)
  del_loss = deletion_rate_loss(gate_output, target_rates, ...)
  total_loss += mr_config.deletion_loss_weight * del_loss
  MSE between actual deletion fraction and target — not traditional regularization, but it constrains gate behavior. Default weight: 0.1.

  3. Gradient Clipping (losses_mrdiffusion.py:210)

  torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
  Applied in optimize_fn before every optimizer step.

  What's notably absent

  - No weight decay on the gate parameters specifically (AdamW weight decay from config.optim.weight_decay applies uniformly to all params)
  - No dropout in the gate modules themselves (only what's inherited from the base DDiTBlocks)
                                                                                                                                                                                                  
  The logit L2 (item 1) is the only explicit regularizer targeting gate collapse. If you're seeing the gate saturating or dying early in training, increasing gate_logit_reg_weight beyond 0.001  
  is the lever to pull.

  ┌───────────────────────────────────────────┬──────────────────────────────────────────────────────────────┬─────────────────┐                                                                         
  │                 Mechanism                 │                         In command?                          │ Effective value │                                                                         
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼─────────────────┤                                                                         
  │ deletion_loss_weight (rate MSE)           │ Yes (--deletion-loss-weight 0.1, PI takes over after warmup) │ Adaptive        │
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼─────────────────┤
  │ gate_logit_reg_weight (L2 on gate logits) │ No — uses default                                            │ 0.001           │                                                                         
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼─────────────────┤                                                                         
  │ gate_warmup_steps (ramp from 0)           │ Yes (--gate-warmup-steps 2500)                               │ Active          │                                                                         
  └───────────────────────────────────────────┴──────────────────────────────────────────────────────────────┴─────────────────┘  
  ---
  PI controller

  - After warmup (new_step > args.gate_warmup_steps): pi_controller.update(deletion_rate) is called and its output replaces mr_config_for_loss.deletion_loss_weight each log step                 
  - During warmup: PI controller is silent — the ramp logic at line 542-545 still controls the weight uninterrupted                                                                               
  - Logging: del_loss_w is printed to console and tracked as train/deletion_loss_weight in W&B when PI is active                                                                                  
                                                                                                                                                                                                  
  Usage:                                                                                                                                                                                          
  python train_mrdiffusion.py --use_pi_controller --gate_warmup_steps 1000 ...                                                                                                                    
                                                                                                                                                                                                  
  The PI gains (kp=0.5, ki=1e-5, gamma=0.9) are conservative — ki is intentionally tiny since the integral accumulates every log step, not every train step.
  ---
  Two-phase training by default:                                                                                                                                                                  
                                
  Steps 0 → 2,000 (--freeze_transformer_steps 2000, default): only gate parameters ("delete_gate" in name) have requires_grad=True. The pretrained transformer is completely frozen. Gate trains  
  alone against the score entropy loss + deletion rate loss.                                                                                                                                      
                                                                                                                                                                                                  
  Steps 2,000 → end: all parameters unfrozen, gate and transformer train jointly. EMA is also rebuilt at this point to cover the full model (line 490 / 535-539).                                 
                                                                                                                                                                                                  
  You can disable the freeze phase entirely with --freeze_transformer_steps 0, which trains everything jointly from step 0.                                                                       
                                                                                                                                                                                                  
  The rationale: if the gate starts modifying attention patterns immediately on a pretrained transformer, the transformer has no opportunity to adapt, and early gradient chaos from the gate can 
  corrupt the pretrained weights before the gate has learned anything meaningful. The 2k-step freeze gives the gate a stable signal to learn from while the transformer stays at its pretrained   
  quality.
  ---
  PARAMETER-EFFICIENT TRAINING

  is it possible to conduct an experiment where we keep the baseline model frozen throughout the duration of training the deletion gate to see how it performs?                                   
                                                                                                                                                                
  Yes, and no code changes are needed. The unfreeze triggers only when step == freeze_transformer_steps (line 527). If you set that value above --max-steps, it never fires.                      
                                                                                                                                                                                                
  modal run --detach train_modal_mrdiffusion.py::main \                                                                                                                                           
    --pretrained-from louaaron/sedd-small \                                                                                                                                                       
    --deletion-mode soft \                                                                                                                                                                        
    --deletion-type scaled_sigmoid \                                                                                                                                                              
    --delete-gate-layer 3 \                                                                                                                                                                       
    --target-deletion-rate 0.3 \                                                                                                                                                                  
    --deletion-loss-weight 0.1 \                                                                                                                                                                  
    --freeze-transformer-steps 999999 \                                                                                                                                                           
    --snapshot-freq 5000 \                                                                                                                                                                        
    --max-steps 20000 \                                                                                                                                                                           
    --eval-batch-size 16 \                                                                                                                                                                        
    --wandb-run-name soft-gate-frozen-transformer-20k                                                                                                                                             
                                                                                                                                                                                                  
  This is a meaningful experiment — it answers whether the pretrained transformer representations are already sufficient for the gate to learn useful deletions, or whether joint finetuning is   
  necessary for the transformer to adapt to the gate's masking. If eval/loss and eval/generative_perplexity are competitive with soft-gate-layer3-30pct-20k (joint training), it suggests the gate
   is learning to exploit existing structure rather than requiring the model to co-adapt.                                                                                                         
                  
  One thing to watch: the score entropy loss still backpropagates through the frozen transformer blocks during phase 2 in the current code — gradients are computed but requires_grad=False means 
  no weight update. That's correct behavior, just slightly wasteful on compute. If you want to also skip the backward through frozen layers you'd need torch.no_grad() around those blocks, but
  that's an optimization, not a correctness issue.
  ---
  ROPE

  /Users/aronimadass/Desktop/projects/stanford/CS224N-project/diffusion/Score-Entropy-Discrete-Diffusion/model/rotary.py 
  It's imported directly at modeling_mrdiffusion.py:45
  Four usages, all resolving to diffusion/Score-Entropy-Discrete-Diffusion/model/rotary.py via the sys.path injection at lines 32–35:                                                             
                                     
  ┌─────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐                                             
  │            Line             │                                                       Usage                                                       │                                             
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                             
  │ modeling_mrdiffusion.py:452 │ _attn_with_mask — rotary.apply_rotary_pos_emb (FlashAttn path, soft deletion)                                     │                                             
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                             
  │ modeling_mrdiffusion.py:494 │ _attn_with_positional_rotary — rotary._apply_rotary_pos_emb_torchscript (per-batch, position-aware hard deletion) │                                             
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                             
  │ modeling_mrdiffusion.py:522 │ _attn_flash — rotary.apply_rotary_pos_emb (normal FlashAttn path)                                                 │                                             
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                             
  │ modeling_mrdiffusion.py:604 │ MrSEDD.__init__ — rotary.Rotary(...) instantiation                                                                │                                           
  └─────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘      

  mrdiffusion-sedd uses ROPE at every layer (as opposed to mrBert which uses absolute positional embeddings only at input layer). Does this mean we'll need to modify ROPE implementation to be   
  position-aware during hard-deletion? if modification is needed, I would like it to be controlled via flag so I am able to run experiments with and without the modification at inference time 
  
  The bug: Rotary.forward(x) generates t = torch.arange(seq_len) — always contiguous [0,1,2,...,L_kept-1]. After hard deletion, a token that was at position 5 gets RoPE encoding for position 2  
  (if it's the 3rd survivor). This is wrong — the compressed phase blocks see incorrect relative positions between survivors.                                                                     
                                                                                                                                                                                                  
  The fix: After hard deletion, generate RoPE using the original positions of survivors ([0,2,5,7] etc.) rather than re-indexing from 0. This requires:                                           
  1. Extracting survivor positions from keep_mask in _apply_hard_deletion                                                                                                                         
  2. A new _recompute_rotary_at_positions(positions) method that bypasses the arange cache                                                                                                        
  3. A config flag to control it so both modes are experimentally comparable

  Here's the plan:
                                                     
  1. configuration_mrdiffusion.py: add rope_original_positions: bool = False
  2. modeling_mrdiffusion.py:                                                                                                                                                                     
    - _apply_hard_deletion → return 4th value kept_positions [B, L_kept]
    - MrSEDD._compute_rotary_for_positions → new method, per-batch cos/sin from original positions using _apply_rotary_pos_emb_torchscript (bypasses FlashAttn's shared-position-only limitation) 
    - MrDDiTBlock._attn_with_positional_rotary → new SDPA path using per-batch rotary                                                                                                             
    - MrDDiTBlock.forward → accept positional_rotary_cos_sin optional param                                                                                                                       
    - MrSEDD.forward → wire it up: set after hard deletion, clear at restoration                                                                                                                  
  3. train_mrdiffusion.py: add --rope_original_positions flag 

  What changed and why                                                                                                                                                                            
                                                                                                                                                                                                  
  configuration_mrdiffusion.py                                                                                                                                                                    
  - Added rope_original_positions: bool = False and included it in to_dict                                                                                                                        
                                                                                                                                                                                                  
  modeling_mrdiffusion.py                                                                                                                                                                         
                                                                                                                                                                                                  
  _apply_hard_deletion — now returns 4 values; kept_positions [B, L_kept] stores the original integer positions of surviving tokens.                                                              
                                                                                                                                                                                                  
  _compute_rotary_for_positions(positions) — new method on MrSEDD. Computes cos/sin [B, L_kept, 3, 1, D_rot] from arbitrary non-contiguous positions using inv_freq directly. Must use            
  _apply_rotary_pos_emb_torchscript downstream because FlashAttention's rotary apply takes cos[L, D/2] (single shared sequence, no batch dimension) — it cannot encode different positions per  
  batch item.                                                                                                                                                                                     
                                                                                                                                                                                                
  MrDDiTBlock._attn_with_positional_rotary — new SDPA attention path that calls _apply_rotary_pos_emb_torchscript with per-batch [B, L, 3, 1, D_rot] cos/sin, then runs                           
  F.scaled_dot_product_attention without any mask.
                                                                                                                                                                                                  
  MrDDiTBlock.forward — new positional_rotary_cos_sin optional parameter; takes priority over both delete_gate_mask and _attn_flash.                                                              
   
  MrSEDD.forward:                                                                                                                                                                                 
  - positional_rotary_cos_sin = None added to init vars                                                                                                                                         
  - After hard deletion: unpacks 4 values; if rope_original_positions=True sets positional_rotary_cos_sin, otherwise falls back to existing contiguous _recompute_rotary(x)                       
  - At restore_gate_layer: clears positional_rotary_cos_sin = None (phase 3 is full sequence, contiguous again)                                                                                 
                                                                                                                                                                                                  
  train_mrdiffusion.py                                                                                                                                                                            
  - --rope_original_positions flag (default False, only meaningful with --deletion_mode hard)                                                                                                     
  - Passed through to MrDiffusionConfig                                                                                                                                                           
                                                                                                                                                                                                  
  Experiment command                                                                                                                                                                              
                                                                                                                                                                                                  
  modal run --detach train_modal_mrdiffusion.py::main \                                                                                                                                           
    --pretrained-from louaaron/sedd-small \                                                                                                                                                       
    --deletion-mode hard \                                                                                                                                                                        
    --deletion-type scaled_sigmoid \                                                                                                                                                              
    --delete-gate-layer 3 \                                                                                                                                                                       
    --target-deletion-rate 0.3 \                                                                                                                                                                  
    --deletion-loss-weight 0.1 \                                                                                                                                                                  
    --rope-original-positions \                                                                                                                                                                   
    --snapshot-freq 5000 \                                                                                                                                                                        
    --max-steps 20000 \                                                                                                                                                                           
    --eval-batch-size 16 \                                                                                                                                                                        
    --wandb-run-name hard-gate-rope-original-pos-20k                                                                                                                                              
                                                                                                                                                                                                  
  Compare against hard-gate-layer3-30pct-20k (contiguous re-indexing) to measure the impact of position-aware RoPE. 

  Will this impact training?                                                                                                                                                                      
                                                                                                                                                                                                  
  Yes and no — depends on what "impact" means:                                                                                                                                                    
                                                                                                                                                                                                
  Semantically yes (that's the point): The model now sees correct relative distances between survivors during the compressed phase. Token at position 5 and token at position 7 have a relative   
  gap of 2 in the RoPE encoding, rather than whatever the contiguous re-indexing happened to assign them. The model learns with more faithful positional signal, which may improve hard-deletion
  quality.                                                                                                                                                                                        
                                                                                                                                                                                                  
  Mechanically no:                                                                                                                                                                                
  - _apply_rotary_pos_emb_torchscript is just (qkv * cos) + (rotate_half(qkv) * sin) — fully differentiable, gradients flow identically                                                           
  - SDPA is numerically equivalent to FlashAttention for the forward pass (same operation, different kernel)                                                                                      
  - Position values are in [0, L-1] — same range the model already saw in phase 1, so no frequency blow-up  
                                                                                                                                                                                                  
  One subtle thing to verify: when rope_original_positions=True, the code skips _recompute_rotary(x) after hard deletion, leaving rotary_cos_sin holding the stale full-length [1, L, 3, 1, D]    
  tensor. This is safe because positional_rotary_cos_sin is not None takes priority in the block dispatch, so rotary_cos_sin is never touched during the compressed phase. At restoration         
  (restore_gate_layer or end of loop), rotary_cos_sin is either recomputed or not used again. But worth a quick sanity check if you see unexpected behavior.                                      
                                                                                                                                                                                                  
  Default runs are completely unaffected — rope_original_positions=False by default, the new code path is never entered.                                                                          

  given that it only impacts hard deletion, will I need to retrain the deletion gate if I only used soft deletion during training?                                                                
                                                                                                                                   
  No. The soft-deletion gate is directly usable for hard deletion at inference — no retraining needed. This is the intended pattern and the README even calls it out explicitly ("soft-train /    
  hard-inference").                                                                                                                                                                             
                                                                                                                                                                                                  
  Here's why it works: the gate learns to push suppressed tokens toward sigmoid_mask_scale (-30) and kept tokens toward 0. Hard deletion just binarizes that same signal at deletion_threshold    
  (-15, the midpoint). A token the gate wanted to suppress softly (value ≈ -28) will also be hard-deleted (value < -15). The threshold is a post-hoc decision on values the gate was already      
  trained to separate.                                                                                                                                                                            
                                                                                                                                                                                                  
  --rope-original-positions is irrelevant to this — it only affects what happens after the hard deletion decision has been made (how RoPE is computed for the survivors). The gate itself sees no 
  difference.                                                                                                                                                                                     
                                                                                                                                                                                                  
  So the workflow is:                                                                                                                                                                             
  # Train once with soft deletion
  modal run --detach ... --deletion-mode soft --wandb-run-name soft-gate-layer3-30pct-20k                                                                                                         
                                                                                                                                                                                                  
  # Eval with hard deletion (no retraining)                                                                                                                                                       
  modal run train_modal_mrdiffusion.py::eval_zero_shot_main \                                                                                                                                     
    --run-name soft-gate-layer3-30pct-20k \                                                                                                                                                       
    --deletion-mode hard                                                                                                                                                                          
                                                                                                                                                                                                  
  # Eval with position-aware hard deletion (also no retraining)                                                                                                                                   
  modal run train_modal_mrdiffusion.py::eval_zero_shot_main \                                                                                                                                     
    --run-name soft-gate-layer3-30pct-20k \                                                                                                                                                       
    --deletion-mode hard --rope-original-positions                                                                                                                                                
                                                                                                                                                                                                  
  This also means hard-gate-rope-original-pos-20k is most useful as a training experiment to see if training under correct positional encodings changes what the gate learns — not just as an     
  inference-time switch on a soft-trained checkpoint. 

  Can you double-check the implementation of _compute_rotary_for_positions for correctness and tell me what the biggest challenge was in making ROPE position aware?                              
  
  The implementation is correct. Step-by-step comparison against Rotary.forward:
                                                                                                                                                                                                  
  ┌───────────────┬──────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┐                                                    
  │     Step      │                    Rotary.forward                    │                   _compute_rotary_for_positions                   │                                                    
  ├───────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤                                                    
  │ Positions     │ t = torch.arange(seq_len) → [L]                      │ positions → [B, L_kept] ✓                                         │                                                    
  ├───────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤                                                    
  │ Frequencies   │ einsum("i,j->ij", t, inv_freq) → [L, D/2]            │ positions.float().unsqueeze(-1) * inv_freq → [B, L_kept, D/2] ✓   │                                                    
  ├───────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤                                                    
  │ Embedding     │ cat((freqs, freqs), dim=-1) → [L, D]                 │ same → [B, L_kept, D] ✓                                           │                                                    
  ├───────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤                                                    
  │ cos/sin shape │ [None,:,None,None,:].repeat(1,1,3,1,1) → [1,L,3,1,D] │ .unsqueeze(2).unsqueeze(3).repeat(1,1,3,1,1) → [B,L_kept,3,1,D] ✓ │
  ├───────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤                                                    
  │ v-identity    │ cos[:,:,2,:,:].fill_(1.)                             │ cos[:,:,2,:,:] = 1.0 ✓                                            │
  └───────────────┴──────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┘                                                    
                  
  ---                                                                                                                                                                                             
  The biggest challenge: FlashAttention's rotary apply is not batch-aware for positions.
                                                                                                                                                                                                  
  apply_rotary_pos_emb tries the flash_attn path first:
  cos = cos[0,:,0,0,:cos.shape[-1]//2]  # takes batch index 0 → [L, D/2]                                                                                                                          
  sin = sin[0,:,0,0,:sin.shape[-1]//2]                                  
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)                                                                                                                            
                                                                                                                                                                                                  
  It unconditionally indexes cos[0] — batch dimension is thrown away. This is fine for the normal case where all batch items share the same contiguous positions [0,1,...,L-1]. But after hard    
  deletion, each batch item keeps different original positions — item 0 might keep [0,2,5,7] while item 1 keeps [1,3,4,6]. There is no single shared position sequence that's correct for both.   
                                                                                                                                                                                                  
  The solution was to bypass apply_rotary_pos_emb entirely and call _apply_rotary_pos_emb_torchscript directly — it just does (qkv * cos) + (rotate_half(qkv) * sin) as a tensor operation, which 
  broadcasts correctly over [B, L_kept, 3, n_heads, D] with per-batch [B, L_kept, 3, 1, D] cos/sin. SDPA is then used instead of FlashAttention for the same reason — FlashAttention's varlen
  interface also assumes a single position sequence. 
  ---
  ┌──────────────────────────┬────────────────────────────────────────────────────────────────────────┐                                                                                           
  │            W&B key             │                                 Source                                 │
  ├────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                                     
  │ train/loss                     │ total loss from training step (unchanged)                              │
  ├────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ train/score_entropy_loss       │ from compute_loss_components at log time                                                     │                                                               
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ train/gate_loss                │ from compute_loss_components at log time                                                     │                                                               
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ train/logit_reg_loss           │ from compute_loss_components at log time                                                     │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ train/deletion_rate            │ from dummy forward (unchanged)                                                               │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ train/gate_mean                │ gate_output.mean() — average gate value across all tokens                                    │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ train/gate_std                 │ gate_output.std() — spread of gate values (high = confident decisions)                       │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ train/learning_rate            │ unchanged                                                                                    │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ efficiency/time_per_forward_ms │ CUDA event timing (CPU fallback: perf_counter)                                               │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ efficiency/avg_sequence_length │ L × (1 − deletion_rate)                                                                      │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤                                                               
  │ efficiency/speedup_vs_baseline │ theoretical attention speedup: n_blocks / ((gate_layer+1) + compressed_blocks × kept_ratio²) │
  └────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘
  ---
  GATE COLLAPSE

  Gate collapse is when the gate converges to a trivial fixed output and stops learning meaningful deletions. There are two failure modes:                                                               
                                                                                                                                                                                                       
  Collapse to zero deletion (what you're seeing — deletion_rate ≈ 0.0007):                                                                                                                               
  - Gate outputs values near 0 for all tokens (no deletion)
  - Happens because the score entropy gradient dominates and the safest path for the model is to never delete anything                                                                                   
  - The gate "gives up" learning to discriminate between important and unimportant tokens                             
                                                                                                                                                                                                         
  Collapse to full deletion (opposite failure):                                                                                                                                                          
  - Gate outputs sigmoid_mask_scale (-30) for nearly all tokens                                                                                                                                          
  - Happens when deletion_loss_weight is too high and the model satisfies the rate target by deleting indiscriminately                                                                                   
                                                                                                                                                                                                         
  In both cases the gate loses its discriminative ability — it's no longer asking "is this token important?" and instead outputs the same value for every token. The goal is a gate that outputs a       
  bimodal distribution: ~0 (keep) for important tokens and ~-30 (delete) for unimportant ones.                                                                                                           
                                                                                                                                                                                                         
  Your specific case is the first type: gate_logit_reg_weight=0.001 pushes logits toward 0 (keep), combined with a weak deletion_loss_weight=0.1 drowned out by score entropy ~3500, the gate just keeps 
  everything.
  ---
  The cleanest fix: disable the freeze when stop_gate_grad=True. The freeze's purpose was to let the gate warm up before the transformer adapts — but with stop_gate_grad, the gate never influences the      
  transformer anyway, so the freeze serves no purpose and breaks the GradScaler.                                                                                                                               
                                                                                                                                                                                                              
  The freeze is now automatically disabled when --stop-gate-grad is passed.
  ---