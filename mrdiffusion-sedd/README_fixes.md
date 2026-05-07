FIXES

GATE COLLAPSE

given the current case of gate collapse (keep all), how should regularization be adjusted?                                                                                                             
                                                                                             
⏺ To fix collapse-to-keep, you need to push harder toward deletion and reduce the bias toward keeping:                                                                                                   
                                                                                                                                                                                                       
  1. Increase deletion_loss_weight (most impactful)                                                                                                                                                      
  - Current: 0.1 — overwhelmed by score entropy ~3500                                                                                                                                                    
  - Try: 1.0 or 5.0 — makes the rate MSE loss competitive                                                                                                                                                
  - The PI controller helps but starts at 0.1 and may ramp too slowly if warmup is 2500 steps
                                                                                                                                                                                                         
  2. Reduce gate_logit_reg_weight (currently working against you)                                                                                                                                        
  - Current: 0.001 — L2 on logits pushes them toward 0, which means "keep" (gate output = 0)                                                                                                             
  - This regularization is biased toward the collapsed state you're already in                                                                                                                           
  - Try: 0.0 or 0.0001 — remove the keep-bias during early training                                                                                                                                      
                                                                                                                                                                                                         
  3. Consider asymmetric initialization                                                                                                                                                                  
  - The gate linear layer initializes near zero → sigmoid(0) = 0.5 → scaled = -15 → near threshold                                                                                                       
  - A small negative bias init would push early gate outputs toward deletion, helping escape collapse                                                                                                    
                                                                                                                                                                                                         
  Recommended command change:                                                                                                                                                                            
  --deletion-loss-weight 1.0 \       # was 0.1                                                                                                                                                           
  --gate_logit_reg_weight 0.0 \      # was 0.001 (default)                                                                                                                                               
  --use-pi-controller \                                                                                                                                                                                  
                                                                                                                                                                                                         
  The core insight: gate_logit_reg_weight penalizes large logits symmetrically, but since the collapsed state has logits near 0, it provides zero restoring force — it actually stabilizes the collapsed 
  state rather than escaping it. 
  
---
OOM issues

* At step 5000, the training loop paused to compute generative perplexity. This loads GPT-2-Large (~3 GB) onto the same A100 GPU that's already running the diffusion model (~2.5 GB activations +
   optimizer states). The timeline:
                                                                                                                                                                                                  
  step 5000: training forward pass completes                                                                                                                                                      
           → save checkpoint                                                                                                                                                                      
           → generate samples                                                                                                                                                                     
           → load GPT-2-Large onto GPU  ← peaks at ~38 GB used                                                                                                                                    
           → compute perplexity                                                                                                                                                                   
           → del eval_model                                                                                                                                                                       
           → torch.cuda.empty_cache()                                                                                                                                                             
  step 5001: training forward pass → needs 1.82 GB contiguous → FAILS                                                                                                                             
                                                                                                                                                                                                  
  Why del + empty_cache() wasn't enough                                                                                                                                                           
                                                                                                                                                                                                  
  empty_cache() releases PyTorch's cache back to CUDA, but doesn't fix fragmentation. Think of it like this:                                                                                      
                  
  Before GPT-2:   [train_model=2.5GB][  free=37GB  ]                                                                                                                                              
  During GPT-2:   [train_model=2.5GB][GPT-2=3GB][activations=32GB][free=2GB]                                                                                                                      
  After del+cache:[train_model=2.5GB][ holes ][ holes ][ free=4.25GB scattered]                                                                                                                   
                                                                                                                                                                                                  
  The 4.25 GB of "reserved but unallocated" memory is spread across small non-contiguous blocks. CUDA can't hand out a single contiguous 1.82 GB chunk from scattered 50–200 MB pieces.           
                                                                                                                                                                                                  
  Why expandable_segments:True fixes it                                                                                                                                                           
                  
  Without it, PyTorch pre-allocates fixed-size memory segments. When GPT-2 is deleted, those segments sit as holes between other allocations.                                                     
                  
  With expandable_segments:True, PyTorch uses CUDA's virtual memory API to grow/shrink allocations in-place — no fixed holes, so fragmentation doesn't accumulate across the GPT-2 load/unload    
  cycle.          
                                                                                                                                                                                                  
  The underlying tension                                                                                                                                                                          
   
  Perplexity eval on the same GPU as training is inherently fragile on a 40 GB A100 with a large diffusion model. The real mitigation options in order of robustness:                             
                  
  1. expandable_segments:True — what we added, cheapest fix                                                                                                                                       
  2. Load GPT-2 to CPU, run perplexity on CPU (slow but zero GPU pressure)
  3. Run perplexity in a separate Modal function on a second GPU                                                                                                                                  
  4. Reduce GPT-2 from gpt2-large to gpt2 (~500 MB instead of 3 GB)  
---