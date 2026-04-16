This is a fascinating research direction that sits right at the intersection of your MrBERT work and your interest in diffusion models. Let me walk through the key challenges and potential solutions.
The Core Tension
The delete gate in MrT5/MrBERT learns to dynamically drop tokens from intermediate representations to speed up inference. It works well in discriminative encoder-only models (BERT, XLM-R) because you do a single forward pass and the final prediction (classification, span extraction) is relatively tolerant of reduced internal resolution.
Discrete diffusion models are fundamentally different in ways that make this non-trivial.
Challenge 1: Score Computation Requires Full Positional Coverage
In SEDD, the score network outputs ratios for every position simultaneously:
(s_θ(x₁...xᵢ...xₐ, t))ᵢ,x̂ᵢ ≈ pₜ(x₁...x̂ᵢ...xₐ) / pₜ(x₁...xᵢ...xₐ)
This is a seq-to-seq map {1,...,n}^d → R^{d×n}. If you delete tokens from intermediate layers, you lose the ability to produce scores for those positions at the output. But the reverse diffusion sampling (τ-leaping, Tweedie denoiser) needs a score for every token position at every timestep to decide transitions. There's no analog of a [CLS] token absorbing everything — you need spatially resolved outputs.
Similarly in BD3-LM, the denoiser p_θ(x^b | x^b_t, x^{<b}) must predict clean tokens for all L′ positions within a block. Deleting internal representations means you can't produce predictions for deleted positions.
Challenge 2: Iterative Multi-Step Sampling
In BERT/XLM-R, you run the network once. In diffusion, you run it many times (32–2048 steps in SEDD, T=5K in BD3-LM). This creates two sub-problems.
First, deletion decisions at one step propagate through subsequent steps. If you delete a token's representation at step t, and the reverse process decides to change that token at step t−Δt, you've lost the context needed to make that transition accurately. Second, the same network is called at different noise levels σ(t), and the information content per token varies dramatically. At high noise (t≈T), most tokens are masked/corrupted and carry little signal — deletion is "cheap." At low noise (t≈0), most tokens are clean and carry critical signal — deletion is expensive. The delete gate needs to be noise-level-aware, which MrBERT's gate wasn't designed for.
Challenge 3: BD3-LM's Block-Causal Structure
BD3-LM uses a specific attention mask structure: block-diagonal for noised tokens within a block, offset-block-causal for cross-attention to previous clean blocks, and block-causal for updating clean blocks. If you delete tokens within a block, you break the block-diagonal self-attention pattern. The FlexAttention kernel they designed (their Figure 4) assumes fixed block sizes — dynamic deletion would require restructuring this carefully optimized sparsity pattern.
Challenge 4: Training Objective Compatibility
SEDD's denoising score entropy (Theorem 3.4) requires computing p_{t|0}(y|x₀) / p_{t|0}(xₜ|x₀) for all positions. These ratios are defined per-token in the original sequence. If tokens are deleted/merged during the forward pass, you still need to produce loss terms for all original positions, or you need to reformulate the loss.
BD3-LM already suffers from high gradient variance due to the masking noise process (their Section 4.3 shows this is a key bottleneck). Adding stochastic deletion decisions introduces another source of variance. Their whole Section 5 is about reducing variance through clipped schedules — a deletion gate would work against this.
Challenge 5: The PI Controller Doesn't Transfer Directly
In MrBERT, the PI controller targets a fixed deletion rate (say 50%) across all inputs. In diffusion, the optimal deletion rate should vary with the noise level. At t≈1 (fully noised), maybe you can delete 70% of tokens. At t≈0 (nearly clean), maybe only 10%. A fixed-rate controller is wrong; you'd need a schedule-aware controller, which is a t → deletion_rate mapping that itself needs to be learned or designed.
Potential Solution Architecture
Here's how I'd approach it, combining ideas from both papers:
1. Internal merging with positional restoration. Don't delete tokens from the output — delete them from intermediate transformer layers and restore before the final projection. Concretely:
Input: x_t ∈ R^{L×d}          (L tokens, d-dim embeddings)
  → Layers 1..k: full attention (L tokens)
  → Delete gate: merge L tokens → L' tokens (L' < L)
  → Layers k+1..m: reduced attention (L' tokens)
  → Restore: expand L' → L using stored positional info
  → Layers m+1..N: full attention (L tokens)
  → Output: scores/predictions for all L positions
This is similar to how MrT5 uses pre-deletion blending — you blend information before deleting, then expand back. The key insight is that the outer layers see full resolution, preserving compatibility with the score entropy loss (SEDD) or the masked cross-entropy (BD3-LM).
2. Noise-conditioned deletion rate. Condition the delete gate on the noise level t:
gate_logit(token_i) = MLP(h_i, σ(t))
Then define a deletion schedule r(t) (analogous to BD3-LM's noise schedule) that specifies the target deletion rate at each noise level. You could optimize this schedule using the same variance-minimization framework from BD3-LM Section 5.3 — find the deletion schedule that minimizes Var_{X,t}[∇_θ L].
3. Apply within BD3-LM blocks. BD3-LM's block structure gives natural boundaries. Apply token merging within each block of L′ tokens during the diffusion layers, while keeping the cross-block attention (offset-block-causal mask) at full resolution. This preserves the KV caching advantage across blocks while reducing computation within blocks. For block size L′=16, even merging down to 8 internal tokens per block could give meaningful speedup on the attention bottleneck.
4. Soft deletion during training, hard during inference. Exactly as in MrBERT — use soft (differentiable) deletion with the Gumbel-sigmoid trick during training, and hard thresholding during inference. The PI controller would target r(t) rather than a fixed rate.
5. Speedup analysis differs from MrBERT. In MrBERT, speedup is measured per forward pass. In diffusion, the total cost is (number of steps) × (cost per step). Token merging reduces cost per step, but you might also need fewer steps if the merged representations are smoother. Conversely, if merging introduces approximation error, you might need more steps. The Pareto frontier from SEDD Figure 1 (generative perplexity vs. sampling iterations) would shift, and characterizing this tradeoff is important.
Where BD3-LM Is More Amenable Than SEDD
BD3-LM is probably the better starting point for this work because of its masking-based noise process. When many tokens in a block are masked, those masked tokens are all identical ([MASK]), which means they carry redundant information in intermediate layers — a perfect target for merging. The deletion gate can learn to aggressively merge masked tokens (which are all the same) while preserving unmasked tokens. This creates a natural curriculum: early in the reverse process (high mask rate), merge aggressively; late (low mask rate), merge less.
SEDD with uniform noise doesn't have this property — corrupted tokens take random values rather than a single mask token, so they're not obviously redundant internally.
Summary of the Research Roadmap
The core idea is sound — diffusion models run the network many times, so per-step speedup compounds. The main challenges are maintaining per-position output fidelity, handling noise-level-dependent information content, and avoiding additional gradient variance. The most promising approach is internal merging with positional restoration, conditioned on noise level, applied within BD3-LM's block structure. Want me to sketch out specific equations for any of these components, or dive deeper into the variance analysis?