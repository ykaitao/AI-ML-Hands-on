# GPT2Model Quick Reference (Attention & Forward API)

Focused notes for reading the GPT‑2 PyTorch implementation in `transformers/models/gpt2/modeling_gpt2.py`. Shapes and behaviours reference the eager attention path (`eager_attention_forward`) and the public `forward` arguments used across GPT‑2 variants.

---

## Attention Tensor Map

| Symbol | Meaning | Example shape |
| --- | --- | --- |
| `b` | batch size | 2 |
| `s_q`, `s_k` | query/token length and key/value length | 32, 32 |
| `nh` | number of attention heads | 12 |
| `dh` | dimension per head (hidden_size / nh) | 64 |

All tensors in the snippets below follow `[b, nh, seq, dim]` unless stated otherwise.

---

## `eager_attention_forward`

Code reference: `transformers/models/gpt2/modeling_gpt2.py` (function defined near the top of the file).

```python
def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    # query: [b, nh, s_q, dh]
    # key/value: [b, nh, s_k, dh]

    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    # -> [b, nh, s_q, s_k]

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full([], value.size(-1) ** 0.5,
                                                 dtype=attn_weights.dtype,
                                                 device=attn_weights.device)

    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.full([], torch.finfo(attn_weights.dtype).min,
                                dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask  # padding mask carries 0 / -inf

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # along s_k
    attn_weights = module.attn_dropout(attn_weights.type(value.dtype))

    attn_output = torch.matmul(attn_weights, value).transpose(1, 2)
    # -> attn_output: [b, s_q, nh, dh]
    #    attn_weights: [b, nh, s_q, s_k]

    return attn_output, attn_weights
```

### Shape Summary

| Tensor | Shape | Notes |
| --- | --- | --- |
| `query`, `key`, `value` | `[b, nh, s_q/s_k, dh]` | Produced by `GPT2Attention._split_heads` |
| `module.bias` | `[1, 1, max_pos, max_pos]` | Causal mask buffer registered at init |
| `attention_mask` | `[b, 1, 1, s_k]` or `[b, nh, s_q, s_k]` | Broadcast-friendly padding mask |
| `attn_weights` | `[b, nh, s_q, s_k]` | Probabilities after softmax |
| `attn_output` | `[b, s_q, nh, dh]` | Transposed layout ready for merge |

---

## `_upcast_and_reordered_attn`

When `config.reorder_and_upcast_attn=True` (used for mixed precision stability) the attention path reuses the allocations above but enforces fp32 matmul. This function lives immediately under the GPT‑2 attention class definition in the same file.

```python
def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
    bsz, num_heads, q_seq_len, dh = query.size()
    k_seq_len = key.size(-2)

    # 1) Allocate fp32 scores with flattened batch/head dimension
    attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len,
                               dtype=torch.float32, device=query.device)

    scale = 1.0
    if self.scale_attn_weights:
        scale /= float(value.size(-1)) ** 0.5
    if self.scale_attn_by_inverse_layer_idx:
        scale /= float(self.layer_idx + 1)

    # 2) Compute scaled dot-product in fp32
    with torch.autocast(query.device.type, enabled=False):
        q = query.reshape(-1, q_seq_len, dh).float()            # [b*nh, s_q, dh]
        k = key.transpose(-1, -2).reshape(-1, dh, k_seq_len).float()  # [b*nh, dh, s_k]
        # torch.baddbmm(dst, mat1, mat2, beta, alpha) computes: dst = beta * dst + alpha * (mat1 @ mat2)
        # Here beta=0 ensures attn_weights is overwritten with the fp32 result of q @ k (scaled in one kernel).
        attn_weights = torch.baddbmm(attn_weights, q, k, beta=0, alpha=scale)
        attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

    # 3) Apply causal/padding masks exactly as in eager path
    # ... identical logic omitted for brevity ...

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = module.attn_dropout(attn_weights.type(value.dtype))
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2)
    return attn_output, attn_weights
```

**Why pre-allocate?**
- Guarantees fp32 storage (`torch.empty(..., dtype=torch.float32)`).
- Fuses scaling into `torch.baddbmm` (single kernel, no extra allocations).
- Avoids relying on autocast for the matmul result type.

---

## KV Cache Workflow

| Step | Operation | Shape / Note |
| --- | --- | --- |
| 1 | First call to `model(input_ids, use_cache=True)` | Returns `CausalLMOutput(past_key_values=Cache)` |
| 2 | Token-by-token loop | Pass only `input_ids[:, -1:]` plus `past_key_values` |
| 3 | Cache update (`DynamicCache.update`) | Stores keys/values per layer: `[b, nh, cached_len, dh]` |
| 4 | Generation APIs (`generate`) | Wrap the loop automatically; ensure `pad_token_id` set when padding |

Disable caching (`use_cache=False` or `config.use_cache=False`) during training or when using gradient checkpointing to save memory.

---

## Forward Argument Cheat Sheet

| Argument | Default | Use When | Skip When | Notes |
| --- | --- | --- | --- | --- |
| `input_ids` | **Required** | Standard tokenised text | Using `inputs_embeds` directly | `[b, s]` of token IDs |
| `attention_mask` | `None` | Variable-length batches, padding | Single fixed-length sample | Mask with `1` valid / `0` pad |
| `past_key_values` | `None` | Autoregressive generation loops | One-off forward, training | Tuple of cached keys/values per layer |
| `labels` | `None` | Training (LM loss, classification) | Pure inference | Uses `-100` to ignore positions |
| `use_cache` | `True` in eval, `False` in train | Generation, inference | Training, gradient checkpointing | Mixed precision safe |
| `output_attentions` / `output_hidden_states` | `False` | Analysis, probing | Production inference | Returns tuples per layer |
| `return_dict` | `True` | Modern code | Legacy tuple-based code | Access via `outputs.logits`, etc. |
| `position_ids`, `token_type_ids` | `None` | Custom positioning, segments | Standard GPT‑2 usage | Rare for GPT‑2; auto-generated |

### Quick Usage Patterns

- **Batch inference (with padding)**  
  ```python
  enc = tokenizer(texts, padding=True, return_tensors="pt")
  outputs = model(
      input_ids=enc["input_ids"],
      attention_mask=enc["attention_mask"],
  )
  ```

- **Training language model**  
  ```python
  outputs = model(
      input_ids=batch["input_ids"],
      attention_mask=batch["attention_mask"],
      labels=batch["input_ids"].clone(),  # model shifts targets internally
      use_cache=False,
  )
  loss = outputs.loss
  ```

- **Fast generation with cache**  
  ```python
  generated = model.generate(
      input_ids,
      max_length=100,
      use_cache=True,
      pad_token_id=tokenizer.eos_token_id,
  )
  ```

---

## Multiple Choice & QA Heads (Specialised Models)

| Model | Extra Inputs | Purpose |
| --- | --- | --- |
| `GPT2DoubleHeadsModel` | `mc_token_ids`, `mc_labels` | Multiple-choice classification alongside LM loss |
| `GPT2ForQuestionAnswering` | `start_positions`, `end_positions` | Span extraction tasks |

Example (multiple choice):
```python
choices = tokenizer(["Paris ... <CLS>", "London ... <CLS>"], padding=True, return_tensors="pt")
cls_positions = (choices["input_ids"] == tokenizer.cls_token_id).nonzero(as_tuple=False)[:, -1]

outputs = model(
    input_ids=choices["input_ids"].unsqueeze(0),   # [batch, num_choices, seq]
    mc_token_ids=cls_positions.unsqueeze(0),       # [batch, num_choices]
    mc_labels=torch.tensor([0]),
)
```

---

## Practical Reminders

1. **Mask padding when batching** – always pass `attention_mask` if padding tokens are present.  
2. **Disable cache for training** – `use_cache=False` reduces memory usage and is required for gradient checkpointing.  
3. **Inspect attention/hidden states sparingly** – enable only when needed; they add CPU/GPU overhead.  
4. **Set `pad_token_id` for generation** – GPT‑2 uses EOS as pad token; update tokenizer/model config accordingly.  
5. **Gradient checkpointing** – call `model.gradient_checkpointing_enable()` and set `model.config.use_cache = False`.  
6. **Autoregressive loop** – pass only the newest token plus `past_key_values`; speeds up generation dramatically.  
7. **Stay within `max_position_embeddings`** – GPT‑2 base supports 1024 tokens; RoPE-based variants differ.  
8. **Use `return_dict=True`** – dataclass outputs (`outputs.logits`, `outputs.loss`) are clearer than tuple indexing.  
9. **Mask loss on padding** – set padded positions in `labels` to `-100` to ignore them in the LM loss.  
10. **Monitor dtype transitions** – attention weights may be upcast to fp32; final tensors match `value.dtype`.

---

Use this as a compact map back to the source code while working with GPT‑2 attention internals or advanced forward arguments. For the full API surface, defer to the Hugging Face model docs and docstrings in `modeling_gpt2.py`.
