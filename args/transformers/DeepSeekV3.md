# DeepseekV3Config Quick Reference (MoE & LoRA Additions)

Focuses on the parameters that are **specific to DeepSeek‑V3** compared to GPT‑style configs. Defaults and behaviour are based on `transformers/models/deepseek_v3/configuration_deepseek_v3.py`, with code snippets adapted from `transformers/models/deepseek_v3/modeling_deepseek_v3.py`.

---

## Mixture-of-Experts (MoE)

### `n_shared_experts` *(default 1)*
- Number of experts that always process every token. Configured inside the MoE block.

```python
class DeepseekV3MoE(nn.Module):
    def __init__(self, config):
        # shared experts: dense backbone present for every token
        self.shared_experts = DeepseekV3MLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def forward(self, hidden_states):
        residual = hidden_states
        flat_states = hidden_states.view(-1, hidden_states.shape[-1])  # hidden_states: (batch, seq_len, hidden_size) -> flat_states: (batch*seq_len, hidden_size)
        # topk_indices / topk_weights returned by the router (see sections below)
        routed = self.experts(flat_states, topk_indices, topk_weights).view_as(hidden_states)
        return routed + self.shared_experts(residual)    # shared experts always add capacity
```

**Guidance**
- Leave at `1` for DeepSeek‑V3 behaviour.
- Increase (2‑4) to stabilise very small models or noisy data.
- Set to `0` only if you want a purely routed (fully sparse) model.

---

### `n_routed_experts` *(default 256)*
- Total specialist experts owned by each MoE layer.
- Router logits are shaped `(batch·seq_len, n_routed_experts)`; larger counts increase sparsity but require more data.

```python
self.gate = DeepseekV3TopkRouter(config)        # routing MLP
self.experts = DeepseekV3NaiveMoe(config)       # ModuleList; config.num_local_experts == n_routed_experts

router_logits = self.gate(hidden_states)        # (batch*seq_len, n_routed_experts)
topk_idx, topk_w = self.route_tokens_to_experts(router_logits)
flat_states = hidden_states.view(-1, hidden_states.shape[-1])      # hidden_states: (batch, seq_len, hidden_size) -> (batch*seq_len, hidden_size)
routed_output = self.experts(flat_states, topk_idx, topk_w).view_as(hidden_states)
```

**Rule of thumb**: match model size to expert count (e.g. 8‑32 experts for small research models, 128‑256 for large runs).

---

### `num_experts_per_tok` *(default 8)*
- How many routed experts each token uses (`self.top_k` in the MoE module).
- Controls activation sparsity: smaller values → faster inference.

After masking out inactive groups:

```python
self.top_k = config.num_experts_per_tok
scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]  # (tokens, top_k)
topk_weights = router_logits.gather(1, topk_indices)                                  # (tokens, top_k)
```

**Best practice**
- Keep `num_experts_per_tok` ≪ `n_routed_experts` (1‑10%).
- Ensure it is divisible by `topk_group` (see below) to keep groups balanced.

---

### `n_group` *(default 8)* and `topk_group` *(default 4)*
- Experts are evenly split into `n_group` groups. At routing time, only `topk_group` groups are considered per token.

```python
group_scores = (
    router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
    .topk(2, dim=-1)[0]
    .sum(dim=-1)
)  # (tokens, n_group)
group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)
score_mask = (
    group_mask.unsqueeze(-1)
    .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
    .reshape(-1, self.n_routed_experts)
)
```

**Constraints**
- `n_routed_experts % n_group == 0` (integer experts per group).
- `topk_group ≤ n_group`.
- `num_experts_per_tok % topk_group == 0` ensures equal experts per chosen group.

---

### `norm_topk_prob` *(default True)*
- Normalises routing weights so chosen experts receive probabilities that sum to 1.

```python
topk_weights = router_logits.gather(1, topk_indices)    # raw sigmoid weights
if self.norm_topk_prob:
    denominator = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    topk_weights = topk_weights / denominator
topk_weights = topk_weights * self.routed_scaling_factor
```

Leave enabled unless experimenting with unnormalised routing.

---

### `routed_scaling_factor` *(default 2.5)*
- Scales routed expert contributions before they are added back to the residual stream.
- Higher values emphasise routed experts; lower values lean on shared experts.

```python
topk_weights = topk_weights * config.routed_scaling_factor         # scale routed experts
shared_output = self.shared_experts(residual)                      # always-on experts
hidden_states = routed_output + shared_output                      # combine routed + shared
```

Typical range: 1.5–3.0. The DeepSeek checkpoints use 2.5.

---

### `first_k_dense_replace` *(default 3)*
- For the first `k` decoder layers, use a dense MLP instead of MoE (helps stability near the input).

```python
class DeepseekV3DecoderLayer(...):
    if layer_idx >= config.first_k_dense_replace:
        self.mlp = DeepseekV3MoE(config)
    else:
        self.mlp = DeepseekV3MLP(config)   # dense FFN
```

Set to `0` for fully sparse models or to `num_hidden_layers` to disable MoE entirely.

---

### `moe_intermediate_size` *(default 2048)*
- Hidden width of each routed/shared expert MLP (`DeepseekV3MLP` inside the MoE).
- Total active capacity ≈ `moe_intermediate_size × num_experts_per_tok`.

```python
class DeepseekV3NaiveMoe(nn.ModuleList):
    for _ in range(config.num_local_experts):
        self.append(
            DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
        )
```

(`config.num_local_experts` is an alias for `n_routed_experts` supplied via the configuration attribute map.)

Use larger values when you decrease the number of experts, so each expert retains enough capacity.

---

## LoRA-based Attention

### `num_key_value_heads` *(default = `num_attention_heads`)*
- Enables grouped-query attention by decoupling the number of key/value heads from total attention heads.

```python
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
key_states = repeat_kv(key_states, self.num_key_value_groups)
value_states = repeat_kv(value_states, self.num_key_value_groups)
```

- When `num_key_value_heads == num_attention_heads`, the layer falls back to standard multi-head attention.
- Setting it to `1` yields multi-query attention (single KV head shared by all queries).
- Intermediate values provide GQA with `num_key_value_groups` query heads per KV head.

### `kv_lora_rank` *(default 512)*
- Rank used to factorise key/value projections. If `None`, the model falls back to full-rank linear layers.

```python
self.kv_a_proj_with_mqa = nn.Linear(
    hidden_size, kv_lora_rank + qk_rope_head_dim, bias=config.attention_bias
)  # down projection
self.kv_a_layernorm = DeepseekV3RMSNorm(kv_lora_rank)
self.kv_b_proj = nn.Linear(
    kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
)  # up projection
```

Pick a rank that is ~5‑10% of the full KV dimension (`num_key_value_heads × (qk_rope_head_dim + v_head_dim)`).

---

### `q_lora_rank` *(default 1536)*
- Low-rank factor for queries. Typically larger than the KV rank because queries drive attention scores.

```python
if config.q_lora_rank is None:
    self.q_proj = nn.Linear(hidden_size, num_heads * qk_head_dim, bias=False)
else:
    self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=config.attention_bias)
    self.q_a_layernorm = DeepseekV3RMSNorm(q_lora_rank)
    self.q_b_proj = nn.Linear(q_lora_rank, num_heads * qk_head_dim, bias=False)
```

Common ratio: `q_lora_rank ≈ 2–3 × kv_lora_rank`.

---

### `qk_rope_head_dim` *(default 64)* and `qk_nope_head_dim` *(default 128)*
- Split query/key dimensions into RoPE and non-RoPE portions. Total head dimension is their sum.

```python
q_states = q_states.view(query_shape).transpose(1, 2)
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

cos, sin = position_embeddings
if config.rope_interleave:
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
else:
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

query_states = torch.cat((q_pass, q_rot), dim=-1)
key_states = torch.cat((k_pass, k_rot), dim=-1)
```

RoPE dims handle positional awareness; non-RoPE dims emphasise semantic similarity. Adjust the split to bias one or the other.

---

### `v_head_dim` *(default 128)*
- Value head width; can differ from the query/key head dimension.
- When using flash attention, values are padded/truncated to match query/key width before softmax and then trimmed back.

```python
if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
    value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])
...
attn_output, attn_weights = attention_interface(..., dropout=self.attention_dropout, scaling=self.scaling)
if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
    attn_output = attn_output[:, :, :, : self.v_head_dim]
attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
attn_output = self.o_proj(attn_output)
```

Tune `v_head_dim` to adjust output expressiveness without touching the attention scoring dimensions.

---

### `kv_lora_rank`, `q_lora_rank`, `qk_*`, `v_head_dim` Interaction
- Effective attention shape per head:
  - Query/key: `qk_rope_head_dim + qk_nope_head_dim`.
  - Value: `v_head_dim`.
- LoRA ranks factorise the heavy input projections (`hidden_size → num_heads × head_dim`).
- Keep LoRA ranks below the full rank; if you set them ≥ full dimension you lose memory savings.

---

## Additional Decoder Settings

### `rope_interleave` *(default True)*
- Switches to the interleaved rotary helper for efficiency when enabled. Set to `False` if you prefer the standard RoPE kernel.

```python
if config.rope_interleave:
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
else:
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
```

### `hidden_act` *(default `"silu"`)*
- Activation used by the dense and expert MLPs.

```python
self.act_fn = ACT2FN[config.hidden_act]
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### `rms_norm_eps` *(default 1e-6)*
- Epsilon for the RMSNorm layers that wrap self-attention and MLP blocks.

```python
self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### `attention_dropout` *(default 0.0)* and `attention_bias` *(default False)*
- Dropout probability applied to attention weights during training, and a flag controlling whether the LoRA projections include bias terms.

```python
attn_output, attn_weights = attention_interface(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,
)

self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=config.attention_bias)
```

### `pretraining_tp` *(default 1)*
- Tensor-parallel partition count used for weight initialisation or sharding-aware loading. Leave at `1` unless aligning with a tensor-parallel checkpoint.

### `rope_scaling` / `rope_parameters`
- Optional dictionary that tweaks RoPE behaviour (e.g. YaRN extrapolation). The configuration converts legacy keys to `self.rope_parameters` and validates them before constructing rotary embeddings.

```python
rope_scaling = kwargs.pop("rope_scaling", None)
self.rope_parameters = rope_scaling or rope_parameters
standardize_rope_params(self, rope_theta=kwargs.get("rope_theta", 10000.0))
rope_config_validation(self)
```

### `max_position_embeddings` *(default 4096)*
- Sets the maximum sequence length used to initialise rotary embeddings and caches.

```python
self.max_position_embeddings = max_position_embeddings
rotary_emb = DeepseekV3RotaryEmbedding(config)  # uses config.max_position_embeddings
```

Internally, the rotary module keeps two trackers for dynamic context growth. During `forward`, the decorator reroutes control to `dynamic_frequency_update`, which compares the current sequence length to the cached limits and resizes the RoPE tables when needed:

```python
class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

    @dynamic_rope_update               # keep decorator in the real code
    def forward(self, x, position_ids):
        # pseudo-expansion of the decorator for illustration:
        seq_len = torch.max(position_ids) + 1  # current request length
        if seq_len > self.max_seq_len_cached:  
            # longer than what we've cached so far → recompute RoPE tables and extend cache
            inv_freq, self.attention_scaling = rope_init_fn(self.config, x.device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            # cache was previously extended beyond the pretrained length but we're back in-range → restore original tables to avoid precision drift
            original_inv_freq = self.original_inv_freq.to(x.device)
            self.register_buffer("inv_freq", original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

        cos, sin = compute_rope_tables(...)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

Here:
- `seq_len > self.max_seq_len_cached` checks whether the current request is longer than anything we have cached; if so we regenerate the RoPE frequencies for the new, longer length and remember that size.
- `seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len` detects the opposite situation: after having grown the cache past its original size, we are now back below the pretrained maximum, so we restore the original frequencies to avoid accumulating numerical error on short sequences.
The decorator automatically upsizes the cached cosine/sine tables when you exceed the configured limit and resets them when you fall back below the original length, using the two stored values.

### `initializer_range` *(default 0.02)*
- Standard deviation used when initialising linear layers throughout the model.

```python
module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
```

### `use_cache`, `pad_token_id`, `bos_token_id`, `eos_token_id`, `tie_word_embeddings`
- Inherit the usual Hugging Face semantics (KV caching, special token IDs, tied embeddings). They mirror the GPT stack and are supported by the shared `PreTrainedModel` utilities.

---

## Routing & Cache Notes

- The router treats every token independently with logits shaped `(batch·seq_len, n_routed_experts)`.
- `past_key_values` (if caching) stores tensors shaped `(batch, num_heads, cached_seq_len, head_dim)`; concatenated when `use_cache=True`.
- MoE layers appear in decoder blocks where `layer_idx ≥ first_k_dense_replace`.

---

## Example Setups

### Compact Sparse Model
```python
config = DeepseekV3Config(
    hidden_size=1024,
    num_hidden_layers=12,
    num_attention_heads=16,

    n_shared_experts=1,
    n_routed_experts=16,
    num_experts_per_tok=4,
    n_group=4,
    topk_group=2,
    routed_scaling_factor=2.0,
    first_k_dense_replace=2,

    kv_lora_rank=128,
    q_lora_rank=256,
    qk_rope_head_dim=32,
    qk_nope_head_dim=64,
    v_head_dim=64,
)
```

### DeepSeek‑V3 Reference
```python
config = DeepseekV3Config(
    hidden_size=7168,
    intermediate_size=18432,
    moe_intermediate_size=2048,
    num_hidden_layers=61,
    num_attention_heads=128,
    num_key_value_heads=128,

    n_shared_experts=1,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_group=8,
    topk_group=4,
    routed_scaling_factor=2.5,
    first_k_dense_replace=3,
    norm_topk_prob=True,

    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
)
```

### Dense Baseline (MoE Disabled)
```python
config = DeepseekV3Config(
    num_hidden_layers=24,
    n_shared_experts=0,
    n_routed_experts=1,
    num_experts_per_tok=1,
    first_k_dense_replace=24,     # matches num_hidden_layers for a fully dense stack

    kv_lora_rank=None,
    q_lora_rank=None,
)
```

---

## Key Reminders

1. **MoE hierarchy**: `n_routed_experts` → split via `n_group` → pruned by `topk_group` → final `num_experts_per_tok`.
2. **Routing stability**: keep `norm_topk_prob=True` and use a sane `routed_scaling_factor`.
3. **Dense warm-up**: use `first_k_dense_replace` to stabilise early layers.
4. **LoRA ranks**: choose ranks well below full dimension; keep `q_lora_rank ≥ kv_lora_rank`.
5. **Hybrid attention**: `qk_rope_head_dim` vs `qk_nope_head_dim` lets you balance positional vs content signals.
6. **Expert capacity**: product `moe_intermediate_size × num_experts_per_tok` should stay comparable to a dense FFN of similar size.
