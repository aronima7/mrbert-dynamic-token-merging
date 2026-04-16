# MrDiffusion: Dynamic Token Merging for Discrete Diffusion Language Models

## Implementation and testing plan

---

## 1. Project overview

### 1.1 Objective

Apply MrT5/MrBERT's dynamic token merging (delete gate) mechanism to discrete diffusion language models (SEDD and BD3-LM) to achieve computational speedup during iterative denoising without sacrificing generation quality. The key insight is that absorbing diffusion creates a large population of identical [MASK] token representations in intermediate transformer layers, which can be merged with minimal information loss.

### 1.2 Core hypothesis

At noise level t in absorbing diffusion, approximately (1 − αₜ) fraction of tokens are [MASK]. These tokens share identical embeddings at the input layer and highly correlated representations in intermediate layers. A learned delete gate can merge these redundant representations in middle transformer layers, reducing self-attention cost from O(L²) to O(L'²) where L' << L, while preserving full positional coverage at input and output layers for correct score/denoising computation.

### 1.3 Target models

- **MrDiffusion-SEDD**: Delete gate integrated into SEDD Absorb's score network
- **MrDiffusion-BD3**: Delete gate integrated into BD3-LM's block denoiser with within-block merging

### 1.4 Success criteria

- Perplexity (NELBO) within 2% of unmodified baselines
- Generative perplexity (GPT-2 Large evaluated) within 5% of baselines
- 30-50% reduction in FLOPs per denoising step
- 20-40% wall-clock speedup per denoising step on A100 GPU
- MAUVE score within 0.02 of baselines for conditional generation

---

## 2. Architecture design

### 2.1 Sandwich architecture (both models)

The delete gate operates in a "sandwich" configuration within the N-layer transformer:

```
Layer group          Sequence length    Purpose
─────────────────────────────────────────────────────
Layers 1..k          L (full)          Build contextual representations
Delete gate          L → L'            Merge redundant tokens
Layers k+1..m        L' (reduced)      Process compressed sequence
Restore              L' → L            Recover full positional coverage
Layers m+1..N        L (full)          Final contextualization
Output head          L                 Scores (SEDD) or distributions (BD3)
```

For the 12-layer architecture used by both papers, default split: k=2, m=10, giving 2 full-resolution early layers, 8 reduced-resolution middle layers, and 2 full-resolution late layers.

### 2.2 Delete gate module

```python
class DeleteGate(nn.Module):
    """
    Noise-conditioned delete gate for diffusion transformers.
    Produces per-token merge/keep decisions conditioned on both
    the token representation and the current noise level.
    """
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        # Input: token hidden state (hidden_dim) + noise level embedding (hidden_dim)
        # Output: scalar gate logit per token
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1)
        )
        # Noise level projection (reuse the model's time embedding)
        self.noise_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, noise_embedding, temperature=1.0):
        """
        Args:
            hidden_states: (B, L, D) token representations after layer k
            noise_embedding: (B, D) noise level embedding from adaLN
            temperature: Gumbel-sigmoid temperature (anneal during training)

        Returns:
            gate_values: (B, L) soft gate values in [0, 1]
            gate_hard: (B, L) hard binary decisions (for inference)
        """
        # Broadcast noise embedding to all positions
        noise_expanded = self.noise_proj(noise_embedding).unsqueeze(1)  # (B, 1, D)
        noise_expanded = noise_expanded.expand_as(hidden_states)       # (B, L, D)

        # Concatenate and compute gate logits
        gate_input = torch.cat([hidden_states, noise_expanded], dim=-1)  # (B, L, 2D)
        logits = self.gate_mlp(gate_input).squeeze(-1)                  # (B, L)

        # Gumbel-sigmoid for differentiable sampling
        if self.training:
            # Sample Gumbel noise
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            gate_values = torch.sigmoid((logits + gumbel) / temperature)
        else:
            gate_values = torch.sigmoid(logits)

        # Hard decisions via straight-through estimator
        gate_hard = (gate_values > 0.5).float()
        gate_hard = gate_values + (gate_hard - gate_values).detach()

        return gate_values, gate_hard
```

### 2.3 Pre-deletion blending

```python
class PreDeletionBlend(nn.Module):
    """
    Before deleting tokens, blend each token's representation with its
    neighbor weighted by the gate value. This ensures the surviving
    token carries information from deleted neighbors.

    Implements: h'_i = g_i * h_i + (1 - g_i) * h_{next_surviving}
    """
    def forward(self, hidden_states, gate_values):
        """
        Args:
            hidden_states: (B, L, D)
            gate_values: (B, L) where 1 = keep, 0 = delete

        Returns:
            blended: (B, L, D) blended representations
        """
        B, L, D = hidden_states.shape

        # Shift hidden states to get "next token" representations
        h_next = torch.roll(hidden_states, -1, dims=1)
        h_next[:, -1, :] = hidden_states[:, -1, :]  # Last token keeps itself

        # Blend: kept tokens stay, deleted tokens blend into neighbors
        gate_expanded = gate_values.unsqueeze(-1)  # (B, L, 1)
        blended = gate_expanded * hidden_states + (1 - gate_expanded) * h_next

        return blended
```

### 2.4 Token merge and restore operations

```python
class TokenMerge(nn.Module):
    """
    Remove deleted tokens from the sequence, producing a shorter sequence
    for the reduced-resolution transformer layers.
    """
    def forward(self, hidden_states, gate_hard, position_ids=None):
        """
        Args:
            hidden_states: (B, L, D) blended representations
            gate_hard: (B, L) binary keep/delete decisions
            position_ids: (B, L) original position IDs for RoPE

        Returns:
            merged: (B, L', D) compressed sequence (L' <= L)
            merge_info: dict with indices needed for restore
        """
        B, L, D = hidden_states.shape

        # Find maximum number of kept tokens across the batch
        keep_counts = gate_hard.sum(dim=1).int()  # (B,)
        L_prime = keep_counts.max().item()

        # Gather kept tokens (pad shorter sequences)
        merged = torch.zeros(B, L_prime, D, device=hidden_states.device)
        merged_positions = torch.zeros(B, L_prime, dtype=torch.long,
                                        device=hidden_states.device)
        keep_mask = torch.zeros(B, L_prime, dtype=torch.bool,
                                 device=hidden_states.device)

        for b in range(B):
            kept_indices = gate_hard[b].nonzero(as_tuple=True)[0]
            n_kept = len(kept_indices)
            merged[b, :n_kept] = hidden_states[b, kept_indices]
            if position_ids is not None:
                merged_positions[b, :n_kept] = position_ids[b, kept_indices]
            keep_mask[b, :n_kept] = True

        merge_info = {
            'gate_hard': gate_hard,
            'keep_counts': keep_counts,
            'keep_mask': keep_mask,
            'original_length': L,
            'merged_positions': merged_positions
        }

        return merged, merge_info


class TokenRestore(nn.Module):
    """
    Expand merged sequence back to full length by scattering kept tokens
    to their original positions and interpolating deleted positions
    from their nearest surviving neighbor.
    """
    def forward(self, merged, merge_info):
        """
        Args:
            merged: (B, L', D) compressed representations
            merge_info: dict from TokenMerge

        Returns:
            restored: (B, L, D) full-length representations
        """
        B = merged.shape[0]
        L = merge_info['original_length']
        D = merged.shape[-1]
        gate_hard = merge_info['gate_hard']

        restored = torch.zeros(B, L, D, device=merged.device)

        for b in range(B):
            kept_indices = gate_hard[b].nonzero(as_tuple=True)[0]
            n_kept = len(kept_indices)

            # Place kept tokens at their original positions
            restored[b, kept_indices] = merged[b, :n_kept]

            # For deleted positions, copy from nearest surviving neighbor
            deleted_indices = (gate_hard[b] == 0).nonzero(as_tuple=True)[0]
            if len(deleted_indices) > 0 and len(kept_indices) > 0:
                # Find nearest kept token for each deleted position
                dists = torch.abs(
                    deleted_indices.unsqueeze(1).float() -
                    kept_indices.unsqueeze(0).float()
                )
                nearest = kept_indices[dists.argmin(dim=1)]
                restored[b, deleted_indices] = restored[b, nearest]

        return restored
```

### 2.5 PI controller for deletion rate targeting

```python
class DeletionRateController:
    """
    PI controller that adjusts the gate bias to achieve a target deletion
    rate r(t) that varies with noise level.

    The target schedule r(t) should roughly track the masking probability
    (1 - alpha_t) with a safety margin.
    """
    def __init__(self, kp: float = 1.0, ki: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0

    def target_rate(self, t: float) -> float:
        """
        Noise-conditioned target deletion rate.
        At t=1 (fully masked): target ~70% deletion
        At t=0 (fully clean): target ~5% deletion
        """
        # Sigmoid-like schedule tracking mask rate with offset
        mask_rate = t  # Linear schedule: 1 - alpha_t = t
        return 0.05 + 0.65 * mask_rate  # Range: [0.05, 0.70]

    def compute_bias(self, actual_rate: float, t: float) -> float:
        """
        Compute bias adjustment for gate logits.
        """
        target = self.target_rate(t)
        error = target - actual_rate
        self.integral += error
        bias = self.kp * error + self.ki * self.integral
        return bias
```

### 2.6 Integration with SEDD

```python
class MrDiffusionSEDD(nn.Module):
    """
    SEDD score network with delete gate integration.
    Modifications from vanilla SEDD:
    1. Delete gate after layer k
    2. Reduced-resolution attention in layers k+1..m
    3. Restore before layer m+1
    4. Noise level conditions the gate
    """
    def __init__(self, config, gate_layer_k=2, restore_layer_m=10):
        super().__init__()
        self.embedding = TokenEmbedding(config)
        self.rope = RotaryPositionalEncoding(config)
        self.time_embed = NoiseEmbedding(config)

        # Full transformer layers
        self.early_layers = nn.ModuleList([
            DiTBlock(config) for _ in range(gate_layer_k)
        ])
        self.middle_layers = nn.ModuleList([
            DiTBlock(config) for _ in range(restore_layer_m - gate_layer_k)
        ])
        self.late_layers = nn.ModuleList([
            DiTBlock(config) for _ in range(config.num_layers - restore_layer_m)
        ])

        # Delete gate components
        self.delete_gate = DeleteGate(config.hidden_dim)
        self.pre_blend = PreDeletionBlend()
        self.token_merge = TokenMerge()
        self.token_restore = TokenRestore()

        # Output: exponentiate for positive ratios
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, x_t, sigma_t):
        """
        Args:
            x_t: (B, L) noised token indices
            sigma_t: (B,) noise levels

        Returns:
            scores: (B, L, V) predicted score ratios (positive)
        """
        # Embed tokens and noise
        h = self.embedding(x_t)            # (B, L, D)
        pos_ids = torch.arange(h.shape[1], device=h.device)
        t_emb = self.time_embed(sigma_t)   # (B, D)

        # Phase 1: Early layers at full resolution
        for layer in self.early_layers:
            h = layer(h, t_emb, pos_ids)

        # Delete gate
        gate_soft, gate_hard = self.delete_gate(h, t_emb)
        gate = gate_soft if self.training else gate_hard

        # Pre-deletion blending
        h = self.pre_blend(h, gate)

        # Merge
        h_merged, merge_info = self.token_merge(h, gate_hard, pos_ids)
        merged_pos = merge_info['merged_positions']

        # Phase 2: Middle layers at reduced resolution
        for layer in self.middle_layers:
            h_merged = layer(h_merged, t_emb, merged_pos)

        # Restore
        h = self.token_restore(h_merged, merge_info)

        # Phase 3: Late layers at full resolution
        for layer in self.late_layers:
            h = layer(h, t_emb, pos_ids)

        # Output: exponentiate for positive ratios
        logits = self.output_proj(h)      # (B, L, V)
        scores = torch.exp(logits)        # Positive ratios

        return scores
```

### 2.7 Integration with BD3-LM

```python
class MrDiffusionBD3(nn.Module):
    """
    BD3-LM with within-block delete gate.
    Key difference from SEDD integration:
    - Delete gate operates only within the current noised block
    - Cross-attention to cached clean blocks is at full resolution
    - KV cache from previous blocks is completely untouched
    """
    def __init__(self, config, block_size, gate_layer_k=2, restore_layer_m=10):
        super().__init__()
        self.block_size = block_size
        self.gate_layer_k = gate_layer_k

        # Same transformer backbone as BD3-LM
        self.transformer = BD3Transformer(config)

        # Delete gate (operates within blocks only)
        self.delete_gate = DeleteGate(config.hidden_dim)
        self.pre_blend = PreDeletionBlend()
        self.token_merge = TokenMerge()
        self.token_restore = TokenRestore()

    def forward_block(self, x_b_t, kv_cache_prev, t_b):
        """
        Denoise a single block conditioned on previous clean blocks.

        Args:
            x_b_t: (B, L') noised block tokens
            kv_cache_prev: cached K,V from blocks 1..b-1 (full resolution)
            t_b: (B,) noise level for this block

        Returns:
            logits: (B, L', V) clean token predictions for all positions
            kv_new: K,V for this block (after denoising, for caching)
        """
        h = self.transformer.embed(x_b_t)   # (B, L', D)
        t_emb = self.transformer.time_embed(t_b)

        # Phase 1: Early layers with full block + cross-attention to cache
        for i in range(self.gate_layer_k):
            h = self.transformer.layers[i](
                h, t_emb,
                kv_cache=kv_cache_prev,       # Cross-attn at full resolution
                attention_mask='block_self'     # Self-attn within block only
            )

        # Delete gate (within block only)
        gate_soft, gate_hard = self.delete_gate(h, t_emb)
        gate = gate_soft if self.training else gate_hard
        h = self.pre_blend(h, gate)
        h_merged, merge_info = self.token_merge(h, gate_hard)

        # Phase 2: Middle layers with reduced block
        for i in range(self.gate_layer_k, self.restore_layer_m):
            h_merged = self.transformer.layers[i](
                h_merged, t_emb,
                kv_cache=kv_cache_prev,       # Cross-attn STILL full resolution
                attention_mask='block_self'     # Self-attn now on merged tokens
            )

        # Restore
        h = self.token_restore(h_merged, merge_info)

        # Phase 3: Late layers at full block resolution
        for i in range(self.restore_layer_m, len(self.transformer.layers)):
            h = self.transformer.layers[i](h, t_emb, kv_cache=kv_cache_prev)

        logits = self.transformer.output_head(h)  # (B, L', V)
        return logits
```

---

## 3. Training plan

### 3.1 Datasets

| Dataset | Tokenizer | Context length | Tokens trained | Purpose |
|---------|-----------|----------------|----------------|---------|
| LM1B | bert-base-uncased | 128 | 65B | Primary benchmark (SEDD + BD3-LM) |
| OpenWebText | GPT-2 BPE | 1024 | 524B | Scale benchmark (BD3-LM) |

Preprocessing follows both papers exactly: concatenate and wrap sequences (no padding), no BOS/EOS injection for generation experiments.

### 3.2 Training strategy

**Phase 1: Pretrain base diffusion model (without gate)**

Train vanilla SEDD Absorb and BD3-LM to convergence, reproducing baseline numbers from both papers. This gives us the baselines to compare against and a checkpoint to initialize from.

- Architecture: 12 layers, 768 hidden, 12 heads (110M parameters)
- Optimizer: AdamW, lr=3e-4, linear warmup 2500 steps, gradient clip 1.0
- Batch size: 512
- EMA: 0.9999
- Hardware: 8x A100-80GB

**Phase 2: Add delete gate (warm-start from Phase 1)**

Initialize the transformer from Phase 1 checkpoint. Add the delete gate module (randomly initialized) and fine-tune the full model.

- Gate warm-up: First 5K steps, set target deletion rate to 0 (gate learns representations but doesn't delete). Gradually increase target to r(t) over steps 5K-15K.
- Gumbel temperature schedule: Start at τ=2.0, anneal to τ=0.5 over 50K steps (linear)
- Learning rate for gate parameters: 3x base lr (gate needs to learn faster than the frozen-then-unfrozen transformer)
- Freeze transformer layers for first 2K steps, then unfreeze all

**Phase 3: Joint optimization of gate + noise schedule**

For BD3-LM, jointly optimize the deletion rate schedule r(t) and the clipped noise schedule (β, ω). Every 5K gradient steps, run a grid search over:
- β ∈ {0.0, 0.15, 0.3, 0.45}
- ω ∈ {0.7, 0.8, 0.9, 1.0}
- r_max ∈ {0.5, 0.6, 0.7, 0.8} (maximum deletion rate at t=1)

Minimize: Var_{X,t}[L_BD(X; θ, β, ω)] including the gate's stochasticity.

### 3.3 Training budget

| Phase | Dataset | Steps | Tokens | GPU-hours (8xA100) |
|-------|---------|-------|--------|---------------------|
| 1a: SEDD baseline (LM1B) | LM1B | 850K | 65B | ~120 |
| 1b: BD3-LM baseline (LM1B) | LM1B | 1M | 65B | ~160 |
| 1c: BD3-LM baseline (OWT) | OWT | 1M | 524B | ~800 |
| 2a: MrDiff-SEDD (LM1B) | LM1B | 150K | 10B | ~25 |
| 2b: MrDiff-BD3 (LM1B) | LM1B | 150K | 10B | ~30 |
| 2c: MrDiff-BD3 (OWT) | OWT | 150K | 80B | ~130 |
| 3: Joint optimization | Both | 50K | 3B | ~15 |

Total: approximately 1,280 GPU-hours. With 8x A100 node, roughly 7 days wall-clock.

### 3.4 Loss function

The loss is identical to the baseline (denoising score entropy for SEDD, masked cross-entropy NELBO for BD3-LM). The delete gate adds no auxiliary loss — it's trained purely through the gradients flowing back from the main loss through the Gumbel-sigmoid and straight-through estimator.

Optional regularization: add a small L2 penalty on gate logit magnitudes to prevent the gate from becoming overconfident too early:

```python
loss = main_loss + lambda_gate * gate_logits.pow(2).mean()
```

where lambda_gate = 0.001. This keeps the gate "soft" during training, improving the Gumbel approximation quality.

---

## 4. Evaluation plan

### 4.1 Primary metrics table

For every model configuration, report:

| Metric | How computed | Success threshold |
|--------|-------------|-------------------|
| **NELBO perplexity** | exp(NELBO / L), 1000 timestep MC estimate | Within 2% of baseline |
| **Generative perplexity** | GPT-2 Large scoring 1000 un-annealed samples (len 1024) | Within 5% of baseline |
| **MAUVE score** | 5000 gen vs 1000 ref, 50-token prompt, 50-token gen | Within 0.02 of baseline |
| **FLOPs/step** | Analytical: k·L²d + (N-k)·L'²d + gate_overhead | 30-50% reduction |
| **Wall-clock/step** | Mean over 100 steps on A100, fixed batch size | 20-40% reduction |
| **Peak GPU memory** | torch.cuda.max_memory_allocated during training | 15-30% reduction |
| **Throughput** | Samples/second at inference | 25-45% improvement |
| **Sample entropy** | Token-level entropy of generated corpus | Within 0.1 of baseline |

### 4.2 Experiment matrix

**Experiment 1: SEDD on LM1B (proof of concept)**

| Model | LM1B PPL | Gen. PPL (1024 steps) | FLOPs/step |
|-------|----------|-----------------------|------------|
| SEDD Absorb (baseline) | ≤32.79 | ~26 | 1.0x |
| MrDiff-SEDD r_max=50% | ≤?? | ?? | ~0.65x |
| MrDiff-SEDD r_max=70% | ≤?? | ?? | ~0.50x |

**Experiment 2: BD3-LM on LM1B (block sizes)**

| Model | L' | LM1B PPL | Gen. PPL | FLOPs/step |
|-------|-----|----------|----------|------------|
| BD3-LM (baseline) | 4 | ≤28.23 | — | 1.0x |
| MrDiff-BD3 | 4 | ≤?? | — | ~0.60x |
| BD3-LM (baseline) | 8 | ≤29.83 | — | 1.0x |
| MrDiff-BD3 | 8 | ≤?? | — | ~0.55x |
| BD3-LM (baseline) | 16 | ≤30.60 | — | 1.0x |
| MrDiff-BD3 | 16 | ≤?? | — | ~0.50x |

**Experiment 3: BD3-LM on OWT (full scale)**

| Model | OWT PPL | Gen. PPL (L=1024) | Gen. PPL (L=2048) | NFEs |
|-------|---------|-------------------|-------------------|------|
| AR baseline | 17.54 | 14.1 | 13.2 | L |
| BD3-LM L'=4 | ≤20.73 | 25.7 | 23.6 | 1K |
| MrDiff-BD3 L'=4 | ≤?? | ?? | ?? | 1K |

**Experiment 4: Zero-shot transfer (OWT → held-out)**

Evaluate OWT-trained models on PTB, WikiText, LM1B, LAMBADA, AG News, Pubmed, Arxiv. Compare BD3-LM L'=4 baseline vs. MrDiff-BD3 L'=4.

**Experiment 5: Conditional generation / infilling**

| Model | Standard MAUVE | Infill MAUVE | Annealing |
|-------|---------------|--------------|-----------|
| GPT-2 Medium | 0.955 | — | Nucleus-0.95 |
| SEDD Standard | 0.957 | 0.942 | None |
| MrDiff-SEDD Standard | ?? | ?? | None |
| MrDiff-SEDD Infill | — | ?? | None |

**Experiment 6: Pareto frontier (main result)**

Plot generative perplexity vs. total FLOPs for:
- SEDD Absorb at 32, 64, 128, 256, 512, 1024, 2048 steps
- MrDiff-SEDD at same step counts
- BD3-LM L'=4 at same step counts (with NFE matching)
- MrDiff-BD3 L'=4 at same step counts

This produces the key figure: MrDiffusion's Pareto frontier should dominate the baseline frontier.

### 4.3 Ablation studies

**Ablation A: Sandwich depth split**

Fix r_max=60%, vary (k, m) on SEDD/LM1B:

| k (early) | m (restore) | Reduced layers | PPL | FLOPs | Wall-clock |
|-----------|-------------|----------------|-----|-------|------------|
| 1 | 11 | 10 | ?? | ?? | ?? |
| 2 | 10 | 8 | ?? | ?? | ?? |
| 3 | 9 | 6 | ?? | ?? | ?? |
| 4 | 8 | 4 | ?? | ?? | ?? |
| 6 | 6 | 0 (no gate) | baseline | 1.0x | 1.0x |

**Ablation B: Deletion rate schedule**

Fix architecture k=2, m=10, vary r_max:

| r_max | Avg deletion | PPL | FLOPs/step | ΔPPL per ΔFLOP |
|-------|-------------|-----|------------|----------------|
| 0.3 | ~18% | ?? | ?? | ?? |
| 0.5 | ~30% | ?? | ?? | ?? |
| 0.7 | ~42% | ?? | ?? | ?? |
| 0.9 | ~55% | ?? | ?? | ?? |

**Ablation C: Gate conditioning**

| Gate variant | PPL | FLOPs | Notes |
|-------------|-----|-------|-------|
| Noise-conditioned (proposed) | ?? | ?? | Full MLP(h, σ(t)) |
| Token-only (no noise) | ?? | ?? | MLP(h) only |
| Noise-only (no token) | ?? | ?? | MLP(σ(t)) broadcast |
| Static (always delete [MASK]) | ?? | ?? | Oracle baseline |
| Random (coin flip at r(t)) | ?? | ?? | Ablation baseline |

**Ablation D: Soft vs. hard gate gap**

| Evaluation mode | PPL | Notes |
|----------------|-----|-------|
| Soft gate (training mode) | ?? | Upper bound on quality |
| Hard gate (inference mode) | ?? | Actual deployment quality |
| Gap | ?? | Should be < 0.5 PPL |

**Ablation E: Interaction with BD3-LM clipped schedules**

| Noise schedule | With gate PPL | Without gate PPL | Var. NELBO (with) | Var. NELBO (without) |
|---------------|---------------|------------------|-------------------|---------------------|
| U[0, 1] | ?? | 31.33 | ?? | 7.39 |
| U[0.3, 0.8] | ?? | 31.19 | ?? | 3.62 |
| U[0.5, 1.0] | ?? | 31.29 | ?? | 3.63 |
| Optimized | ?? | ?? | ?? | ?? |

### 4.4 Gate behavior analysis

**Analysis 1: Deletion rate vs. noise level**

For the trained model, at each noise level t ∈ {0.0, 0.1, ..., 1.0}, run 100 batches and record:
- Overall deletion rate r(t)
- Deletion rate for [MASK] tokens: r_mask(t)
- Deletion rate for clean tokens: r_clean(t)

Plot all three curves. Expected: r_mask(t) >> r_clean(t) at all noise levels.

**Analysis 2: Gate value distribution**

At t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, plot histograms of gate logits (before sigmoid). Expected: bimodal distribution that becomes more skewed toward "delete" at higher noise levels.

**Analysis 3: Token-level heatmap**

For 5 example sequences, create a 2D heatmap (position × noise level → gate value). Clean tokens should appear as bright vertical stripes; masked tokens should transition from dark (deleted) to bright (kept).

**Analysis 4: PI controller convergence**

Plot |r_target(t) - r_actual(t)| averaged over t and batches vs. training step. Should converge to < 0.05 within 20K steps.

---

## 5. Implementation phases and timeline

### Phase 1: Baseline reproduction (weeks 1-3)

**Week 1: Infrastructure setup**
- Set up training pipeline with PyTorch + FlashAttention
- Implement SEDD score entropy loss (Eq. 7 + Eq. 10 from SEDD paper)
- Implement BD3-LM masked cross-entropy loss (Eq. 8 from BD3 paper)
- Implement data loading for LM1B (concat+wrap, bert-base-uncased tokenizer)
- W&B experiment tracking

**Week 2: Baseline training**
- Train SEDD Absorb on LM1B (110M params, 850K steps)
- Train BD3-LM on LM1B (110M params, pretrain 850K + finetune 150K per block size)
- Validate: match Table 3 (SEDD) and Table 3 (BD3-LM) perplexities

**Week 3: Sampling and evaluation**
- Implement Tweedie denoiser sampling (Algorithm 2 from SEDD)
- Implement BD3-LM block sampling (Algorithm 2 from BD3)
- Implement first-hitting sampler from Zheng et al. for BD3-LM
- Implement generative perplexity evaluation (GPT-2 Large scorer)
- Implement MAUVE score evaluation
- Validate: match Figure 1 Pareto frontier (SEDD) and Table 7 gen. perplexity (BD3)

### Phase 2: Delete gate implementation (weeks 4-6)

**Week 4: Core gate modules**
- Implement DeleteGate, PreDeletionBlend, TokenMerge, TokenRestore
- Implement PI controller for deletion rate targeting
- Unit tests: verify merge/restore roundtrip preserves tensor values for kept tokens
- Unit tests: verify gate output shapes, gradient flow through Gumbel-sigmoid
- Profile: measure gate overhead FLOPs and latency

**Week 5: SEDD integration**
- Integrate gate into SEDD score network (MrDiffusionSEDD class)
- Implement noise-conditioned gate with adaLN embedding reuse
- Modify training loop: add gate warm-up schedule, temperature annealing
- Modify sampling loop: hard gate at inference time
- Initial training run on LM1B: verify loss converges, gate learns non-trivial decisions

**Week 6: BD3-LM integration**
- Integrate gate into BD3-LM block denoiser (MrDiffusionBD3 class)
- Handle within-block merging with cross-block attention at full resolution
- Modify BD3-LM's FlexAttention mask to support variable-length blocks
- Verify KV cache is unaffected by gate
- Initial training run on LM1B with block sizes L'=4, 8, 16

### Phase 3: Evaluation and ablations (weeks 7-9)

**Week 7: Primary experiments**
- Experiments 1-3: Perplexity evaluation on LM1B and OWT
- Experiment 6: Pareto frontier plots
- Gate behavior analysis 1-3: deletion rate curves, histograms, heatmaps

**Week 8: Extended experiments**
- Experiment 4: Zero-shot transfer evaluation
- Experiment 5: Conditional generation / infilling MAUVE scores
- Ablations A-B: sandwich depth and deletion rate sweeps

**Week 9: Final ablations and analysis**
- Ablations C-E: gate conditioning, soft/hard gap, schedule interaction
- Wall-clock timing measurements on A100
- Memory profiling
- Gate behavior analysis 4: PI controller convergence
- Compile all results

### Phase 4: Writing and iteration (weeks 10-12)

**Week 10: Draft results section**
- Create all tables and figures
- Write experiment methodology section
- Write results and analysis

**Week 11: Draft full paper**
- Introduction, related work, method sections
- LaTeX formatting with Overleaf

**Week 12: Revision and submission**
- Internal review and iteration
- Code cleanup and open-source preparation
- Submit to venue (ICML/NeurIPS/ICLR)

---

## 6. Code structure

```
mrdiffusion/
├── configs/
│   ├── sedd_lm1b.yaml           # SEDD baseline config
│   ├── bd3_lm1b.yaml            # BD3-LM baseline config
│   ├── bd3_owt.yaml             # BD3-LM OWT config
│   ├── mrdiff_sedd_lm1b.yaml   # MrDiffusion-SEDD config
│   └── mrdiff_bd3_lm1b.yaml    # MrDiffusion-BD3 config
├── models/
│   ├── transformer.py            # DiT transformer backbone
│   ├── sedd.py                   # Vanilla SEDD score network
│   ├── bd3lm.py                  # Vanilla BD3-LM block denoiser
│   ├── delete_gate.py            # DeleteGate, PreDeletionBlend
│   ├── token_merge.py            # TokenMerge, TokenRestore
│   ├── pi_controller.py          # DeletionRateController
│   ├── mr_sedd.py                # MrDiffusion-SEDD integration
│   └── mr_bd3lm.py               # MrDiffusion-BD3 integration
├── losses/
│   ├── score_entropy.py          # SEDD denoising score entropy
│   └── masked_crossent.py        # BD3-LM masked cross-entropy NELBO
├── sampling/
│   ├── tau_leaping.py            # τ-leaping sampler
│   ├── tweedie.py                # Tweedie denoiser sampler
│   ├── first_hitting.py          # First-hitting sampler (BD3-LM)
│   └── conditional.py            # Infilling / conditional sampling
├── noise/
│   ├── schedules.py              # Linear, geometric, log-linear, clipped
│   ├── transition.py             # Q matrices (absorb, uniform)
│   └── schedule_optimizer.py     # Grid search for optimal (β, ω, r_max)
├── data/
│   ├── lm1b.py                   # LM1B data loading (concat+wrap)
│   └── openwebtext.py            # OWT data loading (concat+wrap)
├── evaluation/
│   ├── perplexity.py             # NELBO perplexity computation
│   ├── generative_ppl.py         # GPT-2 Large generative perplexity
│   ├── mauve_eval.py             # MAUVE score computation
│   ├── sample_entropy.py         # Token-level sample entropy
│   └── zero_shot.py              # Zero-shot evaluation on held-out sets
├── analysis/
│   ├── gate_behavior.py          # Deletion rate vs. noise level plots
│   ├── gate_heatmap.py           # Token-level gate value heatmaps
│   ├── pareto.py                 # Pareto frontier plotting
│   ├── flops.py                  # Analytical FLOPs computation
│   └── profiler.py               # Wall-clock and memory profiling
├── scripts/
│   ├── train_baseline.py         # Train vanilla SEDD / BD3-LM
│   ├── train_mrdiffusion.py      # Train MrDiffusion (Phase 2)
│   ├── evaluate.py               # Run all evaluations
│   ├── ablate.py                 # Run ablation sweep
│   └── generate_samples.py       # Generate and save samples
├── tests/
│   ├── test_merge_restore.py     # Roundtrip tests for merge/restore
│   ├── test_gate_gradient.py     # Gradient flow through Gumbel-sigmoid
│   ├── test_loss_equivalence.py  # Verify loss matches baseline when gate=1
│   └── test_pi_controller.py     # PI controller convergence test
└── requirements.txt
```

---

## 7. Risk mitigation

### 7.1 Risk: Gate destabilizes training

**Symptom:** Loss diverges or oscillates after adding gate.
**Mitigation:** Gate warm-up (5K steps with zero deletion target), lower gate learning rate, larger Gumbel temperature. Fallback: freeze transformer entirely and only train gate on top of frozen features.

### 7.2 Risk: Merge/restore introduces positional errors

**Symptom:** Perplexity degrades disproportionately for longer sequences.
**Mitigation:** Ensure RoPE positions are correctly tracked through merge and scatter back during restore. Test with position-sensitive tasks (e.g., copy task, position-dependent classification). Fallback: add a lightweight position correction MLP after restore.

### 7.3 Risk: Wall-clock speedup is marginal despite FLOPs reduction

**Symptom:** Scatter/gather overhead and irregular tensor shapes negate attention savings.
**Mitigation:** Batch all sequences to same L' via padding (losing some efficiency but enabling regular tensor ops). Use torch.compile for fused kernels. Fallback: implement custom CUDA kernel for merge/restore.

### 7.4 Risk: Gate doesn't learn to differentiate masked vs. clean tokens

**Symptom:** r_mask(t) ≈ r_clean(t) — gate deletes randomly.
**Mitigation:** Add an explicit mask-token indicator feature to gate input (binary flag). This gives the gate a "hint" that masked tokens are merge candidates. Ablate with and without to show the gate can also learn this from representations alone.

### 7.5 Risk: BD3-LM's FlexAttention breaks with variable-length blocks

**Symptom:** FlexAttention mask doesn't support blocks of different sizes within a batch.
**Mitigation:** Pad all merged blocks to the same L' (max across the batch) with a separate padding mask. This loses some efficiency but maintains compatibility with the optimized kernel. Long-term: extend FlexAttention to support per-example block sizes.

### 7.6 Risk: Gradient variance increases too much with gate stochasticity

**Symptom:** Training variance with gate > 2x baseline, slow convergence.
**Mitigation:** Use deterministic gate decisions during loss computation (hard gate, no Gumbel noise) and only use Gumbel for gradient estimation. Alternatively, increase batch size to compensate. Fallback: use a fixed deletion schedule r(t) instead of learned gate, removing one source of stochasticity.

---

## 8. Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPUs | 4x A100-40GB | 8x A100-80GB |
| GPU memory | 40GB per GPU | 80GB per GPU |
| CPU RAM | 128GB | 256GB |
| Storage | 200GB (datasets + checkpoints) | 500GB |
| Training time | ~14 days (4 GPU) | ~7 days (8 GPU) |

---

## 9. Key dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.2 | Core framework |
| FlashAttention 2 | ≥ 2.5 | Efficient attention |
| FlexAttention | PyTorch ≥ 2.5 | BD3-LM block-causal masks |
| transformers | ≥ 4.36 | GPT-2 Large for eval |
| wandb | latest | Experiment tracking |
| mauve-text | latest | MAUVE score computation |
| einops | latest | Tensor manipulation |

---

## 10. Deliverables

1. **Trained model checkpoints**: MrDiff-SEDD and MrDiff-BD3 at 110M scale on LM1B and OWT
2. **Evaluation results**: All tables and figures from Section 4
3. **Analysis artifacts**: Gate behavior plots, heatmaps, Pareto frontiers
4. **Paper draft**: Full submission-ready manuscript (LaTeX/Overleaf)
5. **Open-source code**: Complete codebase on GitHub with README, configs, and reproduction scripts
6. **Generated samples**: 1000 unconditional samples from each model configuration for qualitative inspection
