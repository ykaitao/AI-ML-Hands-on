# GPT2Config Quick Reference

Concise guidance for the most common GPT-2 configuration flags. Defaults are lifted from `transformers/models/gpt2/configuration_gpt2.py`; illustrative snippets mirror the logic in `transformers/models/gpt2/modeling_gpt2.py` but use simplified PyTorch layers for readability.

---

## Core Architecture

### `vocab_size` (default 50257)
- Controls how many token embeddings are learned; must match the tokenizer’s vocabulary.
- Used when constructing the embedding matrix.

```python
self.wte = nn.Embedding(config.vocab_size, config.hidden_size)  # word token embeddings
token_vectors = self.wte(input_ids)  # shape (batch, seq_len, hidden_size)
```

### `n_positions` / `max_position_embeddings` (default 1024)
- Maximum sequence length the model can represent; sets the size of the learned positional embeddings.

```python
self.wpe = nn.Embedding(config.n_positions, config.hidden_size)  # word positional embeddings
position_ids = torch.arange(seq_len, device=input_ids.device)
position_vectors = self.wpe(position_ids)
```

### `n_embd` / `hidden_size` (default 768)
- Width of every hidden vector: token embeddings, attention projections, and feed-forward layers.

```python
self.embed_dim = config.hidden_size
hidden_states = torch.zeros(batch, seq_len, self.embed_dim, device=input_ids.device)
```

### `n_layer` / `num_hidden_layers` (default 12)
- Number of stacked transformer decoder blocks.

```python
self.h = nn.ModuleList(GPT2Block(config, layer_idx=i) for i in range(config.n_layer))  # decoder blocks
for block in self.h:
    hidden_states = block(hidden_states)[0]
```

### `n_head` / `num_attention_heads` (default 12)
- How many attention heads each block uses; `n_embd` must be divisible by this value.

```python
self.num_heads = config.n_head
self.head_dim = config.hidden_size // self.num_heads
query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
```

### `n_inner` (default 4 × `n_embd`)
- Intermediate dimension inside the block MLP; controls feed-forward capacity.

```python
class GPT2MLP(nn.Module):
    def __init__(self, config):  # MLP = feed-forward network inside each block
        super().__init__()
        inner = config.n_inner or 4 * config.hidden_size
        self.c_fc = nn.Linear(config.hidden_size, inner)  # "fully connected" expansion (Conv1D in HF)
        self.act = ACT2FN[config.activation_function]
        self.c_proj = nn.Linear(inner, config.hidden_size)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)
```

---

## Regularization & Activation

### `resid_pdrop` (default 0.1)
- Dropout applied to the outputs of attention and MLP blocks before adding the residual connection.

```python
attn_output = self.resid_dropout(attn_output)  # dropout on residual branch
hidden_states = residual + attn_output
```

### `embd_pdrop` (default 0.1)
- Dropout applied once token and position embeddings are summed.

```python
inputs_embeds = self.drop(token_vectors + position_vectors)
```

### `attn_pdrop` (default 0.1)
- Dropout on the attention probability matrix to regularise self-attention.

```python
attn_probs = self.attn_dropout(attn_probs)
context = torch.matmul(attn_probs, value)
```

### `activation_function` (default `"gelu_new"`)
- Choice of non-linearity in the MLP (options include `gelu_new`, `gelu`, `relu`, `silu`, `tanh`).

```python
self.act = ACT2FN[config.activation_function]
hidden_states = self.act(self.c_fc(hidden_states))
```

### `layer_norm_epsilon` (default 1e-5)
- Epsilon for layer normalisation to avoid division-by-zero.

```python
self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
hidden_states = self.ln_1(hidden_states)
```

### `initializer_range` (default 0.02)
- Standard deviation used when initialising linear and embedding weights.

```python
module.weight.data.normal_(mean=0.0, std=config.initializer_range)
```

> **Tuning tips:** raise dropout (≈0.2) for tiny datasets, lower it (≤0.05) when fine-tuning large checkpoints, and increase `layer_norm_epsilon` (1e-4) when training with fp16/bf16.

---

## Attention Behaviour

### `scale_attn_weights` (default `True`)
- Divides attention logits by `sqrt(head_dim)` for numerical stability.

```python
if config.scale_attn_weights:
    attn_scores = attn_scores / math.sqrt(self.head_dim)
```

### `scale_attn_by_inverse_layer_idx` (default `False`)
- Optionally scales attention logits by `1 / (layer_idx + 1)` to stabilise very deep stacks.

```python
if config.scale_attn_by_inverse_layer_idx:
    attn_scores = attn_scores / float(layer_idx + 1)
```

### `reorder_and_upcast_attn` (default `False`)
- Forces eager attention to compute the dot-product in float32 when using mixed precision to avoid NaNs.

```python
if config.reorder_and_upcast_attn:
    # Upcast to float32 for the dot product, then cast back
    attn_scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
else:
    # Stay in the original precision (e.g., fp16) for faster but less stable math
    attn_scores = torch.matmul(query, key.transpose(-1, -2))
```

### `use_cache` (default `True`)
- Enables key/value caching for autoregressive decoding; usually disabled during training to save memory.

```python
# hidden_states arrives as (batch, seq_len, hidden_size); during generation seq_len is usually 1
batch, seq_len, _ = hidden_states.shape
query_states, key_states, value_states = self.c_attn(hidden_states).split(self.embed_dim, dim=-1)

# reshape to (batch, num_heads, seq_len, head_dim)
query_states = query_states.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

if use_cache and past_key_values is not None:
    # past_key_values stores cached tensors shaped (batch, num_heads, cached_seq_len, head_dim)
    past_k, past_v = past_key_values
    key_states = torch.cat([past_k, key_states], dim=-2)
    value_states = torch.cat([past_v, value_states], dim=-2)

# standard scaled dot-product attention
attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
attn_probs = nn.functional.softmax(attn_scores, dim=-1)
attn_output = torch.matmul(attn_probs, value_states)

# return/update cache for next decoding step
next_past_key_values = (key_states, value_states) if use_cache else None
```

---

## Token IDs & Sequence Summary

### `bos_token_id` / `eos_token_id` (default 50256)
- Special tokens for begin-of-sequence/end-of-sequence; also used as the padding token in vanilla GPT-2.

```python
config = GPT2Config(bos_token_id=50256, eos_token_id=50256)
generated = model.generate(input_ids, bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id)
```

### `summary_*` parameters
- Configure the optional sequence-summary head in `GPT2DoubleHeadsModel` (e.g., classification tasks).
- `summary_type="last"` takes the final token’s hidden state.
- `summary_type="first"` takes the first token’s hidden state.
- `summary_type="mean"` averages all token states.
- `summary_type="cls_index"` looks up the token position you pass in via the `cls_index` argument (e.g., the location of a special `<cls>` classification token) and uses that hidden state—handy when your dataset stores the classification token at a variable position.

```python
self.summary_type = config.summary_type  # options: "last", "first", "mean", "cls_index"
pooled_output = self.summary(hidden_states)
```
> One concrete example of **summary_type="cls_index"** is the GPT‑2 “double heads” multiple-choice setup (used in the original paper and in Hugging Face’s GPT2DoubleHeadsModel). Each candidate answer is appended to the prompt and ended with its own special classification token. Because every candidate can be a different length, the <cls> token lands at a different position for each option. The dataloader therefore supplies a cls_index tensor that marks the exact offset of the classification token for each choice, and summary_type="cls_index" tells the model to grab the hidden state at those positions instead of assuming a fixed location.
---

- `summary_use_proj=True` appends a learnable linear layer to the pooled vector.
- `summary_proj_to_labels=True` makes that projection output `num_labels` features instead of `hidden_size`.
- `summary_activation="tanh"` applies an activation after the projection (set to `None` to skip it).
- `summary_first_dropout` controls dropout applied *before* the projection/activation block (name inherited from the shared `XLMSequenceSummary` helper).

```python
pooled_output = nn.Dropout(config.summary_first_dropout)(pooled_output)
if config.summary_use_proj:
    out_dim = config.num_labels if config.summary_proj_to_labels and config.num_labels > 0 else config.hidden_size
    pooled_output = nn.Linear(config.hidden_size, out_dim)(pooled_output)
if config.summary_activation:
    pooled_output = ACT2FN[config.summary_activation](pooled_output)
```


## Minimal Recipes

```python
# Efficient fine-tuning of the small GPT-2 checkpoint
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config.from_pretrained("openai-community/gpt2")
config.resid_pdrop = config.embd_pdrop = config.attn_pdrop = 0.05
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", config=config)
```

```python
# Training a compact model from scratch
config = GPT2Config(
    vocab_size=24000,
    n_positions=512,
    n_embd=512,
    n_layer=8,
    n_head=8,
    n_inner=2048,
    resid_pdrop=0.2,
    embd_pdrop=0.2,
    attn_pdrop=0.2,
)
```

---

## Key Reminders

- Match `vocab_size` to your tokenizer **before** instantiating the model.
- Maintain `n_embd % n_head == 0`; the implied head dimension is typically 64 in GPT-2 families.
- Increasing `n_positions` roughly quadratically increases attention memory cost; pre-trained checkpoints cannot exceed their original limit without retraining.
- Dropout settings (`resid_pdrop`, `embd_pdrop`, `attn_pdrop`) govern overfitting vs. convergence speed—tune them together.
- Leave `scale_attn_weights=True` unless you are intentionally experimenting with alternative attention formulations.
