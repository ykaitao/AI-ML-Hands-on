# Tensor Dimension Notation for `eager_attention_forward`

We notate [batch_size, sequence_length, number_heads, dimension_head] as [b, s, nh, dh], then notate the code with tensor dimensions using `[b, s, nh, dh]`:

```python
def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    # Input dimensions:
    # query: [b, nh, s_q, dh]  - query sequence length s_q
    # key:   [b, nh, s_k, dh]  - key sequence length s_k (often s_k == s_q)
    # value: [b, nh, s_k, dh]  - value has same length as key
    # attention_mask: [b, 1, s_q, s_k] or [b, nh, s_q, s_k] - broadcastable
    
    # Compute attention scores: Q @ K^T
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    # query: [b, nh, s_q, dh] @ key.transpose(-1, -2): [b, nh, dh, s_k]
    # → attn_weights: [b, nh, s_q, s_k]

    if module.scale_attn_weights:
        # Scale by sqrt(dimension_head)
        # value.size(-1) = dh
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )
        # attn_weights: [b, nh, s_q, s_k] (scaled)

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)
        # attn_weights: [b, nh, s_q, s_k] (further scaled)

    if not module.is_cross_attention:
        # Apply causal mask for autoregressive generation
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        # query_length = s_q, key_length = s_k
        
        # module.bias: [1, 1, max_positions, max_positions] - pre-registered causal mask
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        # causal_mask: [1, 1, s_q, s_k] - lower triangular mask (True for allowed positions)
        
        mask_value = torch.finfo(attn_weights.dtype).min
        # mask_value: scalar (e.g., -3.4e38 for float32)
        
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        # mask_value: [] (scalar tensor)
        
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        # causal_mask: [1, 1, s_q, s_k] broadcasts to [b, nh, s_q, s_k]
        # Where mask is True, keep attn_weights; where False, set to -inf
        # attn_weights: [b, nh, s_q, s_k] (with -inf for masked positions)

    if attention_mask is not None:
        # Apply the attention mask (e.g., padding mask)
        # attention_mask typically: [b, 1, 1, s_k] or [b, 1, s_q, s_k]
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        # causal_mask: [b, 1, s_q, s_k] or [b, nh, s_q, s_k] (sliced to key length)
        
        attn_weights = attn_weights + causal_mask
        # Addition (not masking): attention_mask contains 0 for valid, -inf for masked
        # attn_weights: [b, nh, s_q, s_k] (masked positions have -inf)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    # Softmax over key dimension (dim=-1 = s_k dimension)
    # attn_weights: [b, nh, s_q, s_k] (probabilities, masked positions → 0 after softmax)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    # attn_weights: [b, nh, s_q, s_k] (potentially cast to value's dtype)
    
    attn_weights = module.attn_dropout(attn_weights)
    # attn_weights: [b, nh, s_q, s_k] (with dropout applied)

    attn_output = torch.matmul(attn_weights, value)
    # attn_weights: [b, nh, s_q, s_k] @ value: [b, nh, s_k, dh]
    # → attn_output: [b, nh, s_q, dh]
    
    attn_output = attn_output.transpose(1, 2)
    # Transpose nh and s_q dimensions
    # attn_output: [b, nh, s_q, dh] → [b, s_q, nh, dh]

    return attn_output, attn_weights
    # attn_output: [b, s_q, nh, dh]
    # attn_weights: [b, nh, s_q, s_k]
```

# Clarifying the `attn_weights` Pre-allocation Confusion

You're absolutely right to be confused! Let me clarify the timeline and scope of when `attn_weights` is allocated and used.

---

## The Timeline: When Things Happen

### **Function Call Hierarchy**

```python
# 1. Model forward pass is called (from outside)
outputs = model(input_ids=input_ids)

# 2. This calls GPT2Attention.forward() for each layer
# Inside GPT2Attention.forward():
def forward(self, hidden_states, ...):
    # ... prepare query, key, value ...
    
    # 3. Decide which attention function to use
    if using_eager and self.reorder_and_upcast_attn:
        # 4. Call _upcast_and_reordered_attn
        attn_output, attn_weights = self._upcast_and_reordered_attn(
            query_states, key_states, value_states, attention_mask
        )
    # 5. Function returns, attn_weights goes out of scope
    
    return attn_output, attn_weights
```

---

## Inside `_upcast_and_reordered_attn` - Step by Step

```python
def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
    """
    This function is called ONCE per attention layer per forward pass.
    Each call has its own local attn_weights variable.
    """
    
    # STEP 1: Get dimensions
    bsz, num_heads, q_seq_len, dk = query.size()
    _, _, k_seq_len, _ = key.size()

    # STEP 2: Pre-allocate attn_weights (LOCAL variable)
    # This happens FRESH every time this function is called
    attn_weights = torch.empty(
        bsz * num_heads, 
        q_seq_len, 
        k_seq_len, 
        dtype=torch.float32,  # ← Always fp32
        device=query.device
    )
    # attn_weights is now: [b*nh, s_q, s_k] with GARBAGE values
    # This is a NEW tensor, allocated just now
    
    # STEP 3: Compute scale factor
    scale_factor = 1.0
    if self.scale_attn_weights:
        scale_factor /= float(value.size(-1)) ** 0.5
    if self.scale_attn_by_inverse_layer_idx:
        scale_factor /= float(self.layer_idx + 1)

    # STEP 4: Reshape query and key
    with torch.autocast(query.device.type, enabled=False):
        q = query.reshape(-1, q_seq_len, dk)           # [b*nh, s_q, dh]
        k = key.transpose(-1, -2).reshape(-1, dk, k_seq_len)  # [b*nh, dh, s_k]
        
        # STEP 5: Compute attention scores
        # This OVERWRITES the garbage values in attn_weights
        attn_weights = torch.baddbmm(
            attn_weights,      # ← Same variable as STEP 2 (with garbage)
            q.float(),         # ← Query in fp32
            k.float(),         # ← Key in fp32
            beta=0,            # ← "Ignore the garbage values"
            alpha=scale_factor # ← Scale the result
        )
        # Now attn_weights contains: scale_factor * (q @ k)
        # The garbage values are gone, replaced with real attention scores
        
        attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        # attn_weights: [b, nh, s_q, s_k]

    # STEP 6: Apply masks (causal, padding)
    # ... (modifies attn_weights in place)
    
    # STEP 7: Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    
    # STEP 8: Dropout and compute output
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    # STEP 9: Return (attn_weights is still in scope here)
    return attn_output, attn_weights
    # After return, attn_weights goes out of scope and will be garbage collected
```

---

## Why do we need _upcast_and_reordered_attn?

Let me show you the alternatives to make it clearer:

### **Alternative 1: Let PyTorch Allocate Automatically**

```python
# ❌ Without pre-allocation
def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
    # ... setup ...
    
    with torch.autocast(query.device.type, enabled=False):
        q = query.reshape(-1, q_seq_len, dk)
        k = key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
        
        # Let PyTorch allocate automatically
        attn_weights = torch.matmul(q.float(), k.float())
        # Problem: Result might not be fp32 if inputs are fp16!
        
        attn_weights = attn_weights * scale_factor
        # Problem: Extra operation, extra memory allocation
```

**Issues:**
- No guarantee output is fp32
- Extra memory allocation for scaling
- Separate operations (slower)

---

### **Alternative 2: Pre-allocate with Correct Dtype**

```python
# ✅ With pre-allocation (what's actually done)
def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
    # ... setup ...
    
    # Pre-allocate with EXACT dtype we want (fp32)
    attn_weights = torch.empty(
        bsz * num_heads, q_seq_len, k_seq_len,
        dtype=torch.float32,  # ← Guarantee fp32
        device=query.device
    )
    
    with torch.autocast(query.device.type, enabled=False):
        q = query.reshape(-1, q_seq_len, dk)
        k = key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
        
        # Overwrite the pre-allocated tensor
        # baddbmm = "batch add batch matrix-matrix multiplication"
        # Formula: output = beta * input + alpha * (batch1 @ batch2)
        attn_weights = torch.baddbmm(
            attn_weights,  # ← Use the memory we allocated
            q.float(),
            k.float(),
            beta=0,         # ← Ignore garbage values
            alpha=scale_factor  # ← Apply scaling in one shot
        )
        # Result is guaranteed fp32, single operation
```

**Benefits:**
- ✅ Guaranteed fp32 output
- ✅ Single fused operation
- ✅ Control over memory layout

---

## Summary Table of Tensor Dimensions

| Variable | Dimension | Description |
|----------|-----------|-------------|
| **Input** | | |
| `query` | `[b, nh, s_q, dh]` | Query tensor |
| `key` | `[b, nh, s_k, dh]` | Key tensor |
| `value` | `[b, nh, s_k, dh]` | Value tensor |
| `attention_mask` | `[b, 1, s_q, s_k]` | Attention mask (broadcastable) |
| **Intermediate** | | |
| `key.transpose(-1, -2)` | `[b, nh, dh, s_k]` | Transposed key |
| `attn_weights` (after matmul) | `[b, nh, s_q, s_k]` | Raw attention scores |
| `module.bias` | `[1, 1, max_pos, max_pos]` | Pre-computed causal mask buffer |
| `causal_mask` (from bias) | `[1, 1, s_q, s_k]` | Sliced causal mask |
| `mask_value` | `[]` | Scalar tensor (-inf) |
| `attn_weights` (after softmax) | `[b, nh, s_q, s_k]` | Attention probabilities |
| **Output** | | |
| `attn_output` (before transpose) | `[b, nh, s_q, dh]` | Weighted sum of values |
| `attn_output` (after transpose) | `[b, s_q, nh, dh]` | Final output |
| `attn_weights` (returned) | `[b, nh, s_q, s_k]` | Attention weights |

---

## Key Notes

1. **Multi-head dimension ordering**: Input uses `[b, nh, s, dh]` (heads dimension before sequence), output transposes to `[b, s, nh, dh]` (sequence before heads)

2. **Query and key lengths**: 
   - Usually `s_q == s_k` for self-attention
   - Can differ in cross-attention or when using cached keys/values

3. **Causal mask broadcasting**: `[1, 1, s_q, s_k]` → `[b, nh, s_q, s_k]`

4. **Attention mask format**: Contains `0` for valid positions, `-inf` (or large negative) for masked positions (added, not multiplied)

5. **Matmul operations**:
   - `Q @ K^T`: `[b, nh, s_q, dh] @ [b, nh, dh, s_k] → [b, nh, s_q, s_k]`
   - `attn @ V`: `[b, nh, s_q, s_k] @ [b, nh, s_k, dh] → [b, nh, s_q, dh]`


# Best Practices for GPT2 Model Forward Arguments

This is a comprehensive guide for the forward pass arguments across all GPT-2 model variants. I'll focus on the most commonly used arguments with practical examples.

---

## Core Input Arguments

### 1. **`input_ids`** (Required in most cases)

Token IDs representing the input text.

#### ✅ When to use:
- **99% of use cases** - standard text input
- **Training** - provide tokenized sequences
- **Inference** - text generation, classification
- **Batch processing** - multiple sequences

#### ❌ When NOT to use:
- When using `inputs_embeds` directly (rare, advanced use cases)

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# ✅ GOOD: Standard usage
text = "Hello, how are you?"
input_ids = tokenizer.encode(text, return_tensors="pt")
outputs = model(input_ids=input_ids)

# ✅ GOOD: Batch processing
texts = ["Hello world", "How are you?", "Nice to meet you"]
encodings = tokenizer(texts, padding=True, return_tensors="pt")
outputs = model(input_ids=encodings['input_ids'])

# ✅ GOOD: Generation with past_key_values (incremental)
input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
# First pass - full sequence
outputs = model(input_ids=input_ids, use_cache=True)
past_key_values = outputs.past_key_values

# Second pass - only new token
next_token_id = torch.tensor([[123]])  # Predicted next token
outputs = model(
    input_ids=next_token_id,  # Only 1 token
    past_key_values=past_key_values  # Reuse previous computation
)

# ❌ BAD: Providing both input_ids and inputs_embeds
input_ids = tokenizer.encode("Hello", return_tensors="pt")
inputs_embeds = model.transformer.wte(input_ids)
outputs = model(input_ids=input_ids, inputs_embeds=inputs_embeds)
# ValueError: You cannot specify both input_ids and inputs_embeds

# ❌ BAD: Exceeding max position embeddings
long_text = "word " * 2000  # 2000 tokens
input_ids = tokenizer.encode(long_text, return_tensors="pt")
outputs = model(input_ids=input_ids)  # Error: max is 1024
```

---

### 2. **`attention_mask`**

Binary mask indicating which tokens should be attended to (1) and which are padding (0).

#### ✅ When to use:
- **Batch processing with variable lengths** - essential!
- **Padding** - when sequences have different lengths
- **Ignoring special tokens** - mask out certain positions

#### ❌ When NOT to use:
- **Single sequences without padding** - unnecessary overhead
- **All sequences same length, no padding** - not needed

```python
# ✅ GOOD: Batch with padding (ESSENTIAL)
texts = ["Short", "This is a much longer sentence"]
encodings = tokenizer(texts, padding=True, return_tensors="pt")

print(encodings['input_ids'])
# tensor([[   21,  50256, 50256, 50256, 50256, 50256],  # Padded
#         [  198,   318,   257, ...]])                  # Original

print(encodings['attention_mask'])
# tensor([[1, 0, 0, 0, 0, 0],  # Only attend to first token
#         [1, 1, 1, 1, 1, 1]]) # Attend to all

outputs = model(
    input_ids=encodings['input_ids'],
    attention_mask=encodings['attention_mask']  # CRITICAL!
)

# ❌ BAD: Forgetting attention_mask with padding
texts = ["Short", "This is a much longer sentence"]
encodings = tokenizer(texts, padding=True, return_tensors="pt")
outputs = model(input_ids=encodings['input_ids'])
# Model will attend to padding tokens - wrong results!

# ✅ GOOD: Single sequence (no attention_mask needed)
text = "Hello world"
input_ids = tokenizer.encode(text, return_tensors="pt")
outputs = model(input_ids=input_ids)  # No attention_mask - OK

# ✅ GOOD: Custom masking (e.g., ignore first token)
input_ids = tokenizer.encode("Hello world", return_tensors="pt")
attention_mask = torch.ones_like(input_ids)
attention_mask[0, 0] = 0  # Ignore first token
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# ✅ GOOD: Left-padding for generation
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
texts = ["Generate", "Generate some text"]
encodings = tokenizer(texts, padding=True, return_tensors="pt")
outputs = model.generate(
    encodings['input_ids'],
    attention_mask=encodings['attention_mask'],  # Important!
    max_length=50
)
```

---

### 3. **`past_key_values`** (KV Cache)

Cached key-value states from previous forward passes for faster generation.

#### ✅ When to use:
- **Autoregressive generation** - dramatically faster (10-100x)
- **Interactive chat** - cache conversation history
- **Long sequence generation** - essential for efficiency

#### ❌ When NOT to use:
- **Training** - use `use_cache=False` to save memory
- **Single forward pass** - no benefit
- **Parallel processing of full sequences** - not applicable

```python
# ✅ GOOD: Fast generation with caching (automatic in generate())
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")

# generate() automatically uses caching
output = model.generate(
    input_ids,
    max_length=50,
    use_cache=True,  # Default, but explicit
    do_sample=True
)

# ✅ GOOD: Manual incremental generation with caching
input_ids = tokenizer.encode("Hello", return_tensors="pt")
past_key_values = None

generated = input_ids
for _ in range(10):  # Generate 10 tokens
    if past_key_values is None:
        # First pass - full sequence
        outputs = model(input_ids=generated, use_cache=True)
    else:
        # Subsequent passes - only new token
        outputs = model(
            input_ids=generated[:, -1:],  # Only last token
            past_key_values=past_key_values,
            use_cache=True
        )
    
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

print(tokenizer.decode(generated[0]))

# ✅ GOOD: Chatbot with conversation caching
class ChatBot:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.past_key_values = None
        self.conversation = []
    
    def chat(self, user_input):
        # Encode new input
        new_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        
        if self.past_key_values is None:
            # First message
            input_ids = new_ids
        else:
            # Continue conversation - only pass new tokens
            input_ids = new_ids
        
        # Generate response using cached context
        outputs = self.model.generate(
            input_ids,
            max_length=50,
            past_key_values=self.past_key_values,
            use_cache=True
        )
        
        # Update cache (need to get from forward pass, not generate)
        # For real chatbot, you'd manage this more carefully
        return self.tokenizer.decode(outputs[0])

# ❌ BAD: Using cache during training (wastes memory)
model.train()
for batch in dataloader:
    outputs = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        use_cache=True  # BAD! Wastes memory during training
    )
    loss = outputs.loss
    loss.backward()

# ✅ GOOD: Training without cache
model.train()
for batch in dataloader:
    outputs = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        use_cache=False  # Saves memory
    )
    loss = outputs.loss
    loss.backward()

# 💡 Performance comparison
import time

input_ids = tokenizer.encode("Hello world", return_tensors="pt")

# Without caching
start = time.time()
for _ in range(50):
    full_ids = torch.cat([input_ids, torch.randint(0, 50000, (1, 1))], dim=1)
    outputs = model(full_ids, use_cache=False)
print(f"Without cache: {time.time() - start:.2f}s")

# With caching
start = time.time()
past = None
current_ids = input_ids
for _ in range(50):
    if past is None:
        outputs = model(current_ids, use_cache=True)
    else:
        outputs = model(current_ids[:, -1:], past_key_values=past, use_cache=True)
    past = outputs.past_key_values
    current_ids = torch.cat([current_ids, torch.randint(0, 50000, (1, 1))], dim=1)
print(f"With cache: {time.time() - start:.2f}s")
# Typically 10-100x faster!
```

---

### 4. **`labels`** (Training only)

Target labels for computing loss (language modeling, classification, etc.).

#### ✅ When to use:
- **Training** - compute loss automatically
- **Fine-tuning** - language modeling, classification
- **Validation** - compute metrics

#### ❌ When NOT to use:
- **Inference** - generation, prediction
- **When manually computing loss** - use logits instead

```python
# ✅ GOOD: Language modeling training
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
model.train()

text = "The quick brown fox jumps over the lazy dog"
encodings = tokenizer(text, return_tensors="pt")

# Labels are the same as input_ids for language modeling
outputs = model(
    input_ids=encodings['input_ids'],
    labels=encodings['input_ids']  # Model shifts internally
)

loss = outputs.loss  # Automatically computed
loss.backward()

# ✅ GOOD: Training loop
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

def collate_fn(examples):
    # Tokenizer returns input_ids
    # For LM, labels = input_ids
    encodings = tokenizer([ex['text'] for ex in examples], 
                         padding=True, 
                         truncation=True, 
                         return_tensors="pt")
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': encodings['input_ids'].clone()  # Clone for labels
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)
trainer.train()

# ✅ GOOD: Sequence classification with labels
from transformers import GPT2ForSequenceClassification

model = GPT2ForSequenceClassification.from_pretrained(
    "openai-community/gpt2",
    num_labels=2  # Binary classification
)

texts = ["I love this!", "I hate this!"]
labels = torch.tensor([1, 0])  # Positive, Negative

encodings = tokenizer(texts, padding=True, return_tensors="pt")
outputs = model(
    input_ids=encodings['input_ids'],
    attention_mask=encodings['attention_mask'],
    labels=labels  # Compute classification loss
)

loss = outputs.loss
logits = outputs.logits

# ✅ GOOD: Masking padding tokens in loss
texts = ["Short", "Much longer text here"]
encodings = tokenizer(texts, padding=True, return_tensors="pt")

# Create labels with -100 for padding (ignored in loss)
labels = encodings['input_ids'].clone()
labels[encodings['attention_mask'] == 0] = -100

outputs = model(
    input_ids=encodings['input_ids'],
    attention_mask=encodings['attention_mask'],
    labels=labels  # Padding tokens ignored
)

# ❌ BAD: Using labels during inference
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        labels=input_ids  # Unnecessary, wastes computation
    )
    # Just use: outputs = model(input_ids=input_ids)

# ✅ GOOD: Inference without labels
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    predictions = logits.argmax(dim=-1)
```

---

### 5. **`use_cache`**

Whether to return key-value cache for faster generation.

#### ✅ When to set `True`:
- **Generation** - default and recommended
- **Inference** - interactive applications
- **Long generation** - essential for speed

#### ✅ When to set `False`:
- **Training** - saves memory
- **Single forward pass** - no benefit
- **Gradient checkpointing** - incompatible

```python
# ✅ GOOD: Generation with cache (default)
model.eval()
outputs = model.generate(
    input_ids,
    max_length=100,
    use_cache=True  # Default, but explicit
)

# ✅ GOOD: Training without cache
model.train()
for batch in dataloader:
    outputs = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        use_cache=False  # Save memory
    )
    loss = outputs.loss
    loss.backward()

# ✅ GOOD: Gradient checkpointing (must disable cache)
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Required!

for batch in dataloader:
    outputs = model(
        input_ids=batch['input_ids'],
        labels=batch['labels']
        # use_cache automatically False with gradient checkpointing
    )
    loss = outputs.loss
    loss.backward()

# ❌ BAD: Cache with gradient checkpointing
model.gradient_checkpointing_enable()
outputs = model(
    input_ids=input_ids,
    use_cache=True  # Warning: incompatible!
)
# Warning: `use_cache=True` is incompatible with gradient checkpointing

# 💡 Memory comparison
import torch
torch.cuda.reset_peak_memory_stats()

# With cache (more memory)
outputs = model(input_ids, use_cache=True)
print(f"With cache: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

torch.cuda.reset_peak_memory_stats()

# Without cache (less memory)
outputs = model(input_ids, use_cache=False)
print(f"Without cache: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

### 6. **`output_attentions` and `output_hidden_states`**

Return attention weights and hidden states from all layers.

#### ✅ When to use:
- **Model interpretation** - visualize attention patterns
- **Feature extraction** - use intermediate representations
- **Debugging** - analyze model behavior
- **Research** - probing experiments

#### ❌ When NOT to use:
- **Production inference** - slower, uses more memory
- **Training** - unless needed for specific loss

```python
# ✅ GOOD: Attention visualization
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        output_attentions=True,
        output_hidden_states=True
    )

# Access all layer outputs
hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1)
attentions = outputs.attentions        # Tuple of num_layers

print(f"Number of hidden states: {len(hidden_states)}")  # 13 (embedding + 12 layers)
print(f"Number of attention layers: {len(attentions)}")  # 12 layers
print(f"Attention shape: {attentions[0].shape}")  # [batch, heads, seq, seq]

# ✅ GOOD: Visualizing attention
import matplotlib.pyplot as plt
import seaborn as sns

# Get attention from last layer
last_attention = attentions[-1][0].mean(dim=0)  # Average over heads

plt.figure(figsize=(10, 10))
sns.heatmap(last_attention.cpu().numpy(), cmap='viridis')
plt.title("Attention Patterns - Last Layer")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.show()

# ✅ GOOD: Feature extraction from specific layer
def get_layer_embeddings(text, layer_idx=-2):
    """Extract embeddings from specific layer (default: second-to-last)"""
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )
    
    # hidden_states[0] = embeddings
    # hidden_states[1] = layer 0
    # hidden_states[-1] = final layer
    layer_output = outputs.hidden_states[layer_idx]
    
    # Mean pooling over sequence
    embeddings = layer_output.mean(dim=1)
    return embeddings

embeddings = get_layer_embeddings("Hello world", layer_idx=-2)
print(f"Embedding shape: {embeddings.shape}")  # [1, 768]

# ✅ GOOD: Comparing attention across layers
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Analyze how attention to "cat" changes across layers
cat_token_idx = 2  # Assuming "cat" is at position 2

for layer_idx, attention in enumerate(outputs.attentions):
    # Average attention from all positions to "cat" token
    avg_attention_to_cat = attention[0, :, :, cat_token_idx].mean()
    print(f"Layer {layer_idx}: Avg attention to 'cat' = {avg_attention_to_cat:.4f}")

# ❌ BAD: Using in production without need
# Slower and uses more memory
for batch in production_dataloader:
    outputs = model(
        input_ids=batch['input_ids'],
        output_attentions=True,      # Unnecessary overhead
        output_hidden_states=True    # Unnecessary overhead
    )
    predictions = outputs.logits.argmax(dim=-1)

# ✅ GOOD: Production inference (no extra outputs)
for batch in production_dataloader:
    outputs = model(input_ids=batch['input_ids'])
    predictions = outputs.logits.argmax(dim=-1)

# ✅ GOOD: Probing classifier using hidden states
from torch import nn

class ProbingClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, hidden_states):
        # hidden_states from GPT-2 layer
        pooled = hidden_states.mean(dim=1)  # Mean pool
        return self.classifier(pooled)

# Train probing classifier on layer 6 representations
probe = ProbingClassifier()
optimizer = torch.optim.Adam(probe.parameters())

model.eval()  # Freeze GPT-2
for batch in train_loader:
    with torch.no_grad():
        outputs = model(
            input_ids=batch['input_ids'],
            output_hidden_states=True
        )
        layer_6_hidden = outputs.hidden_states[7]  # Index 7 = layer 6
    
    logits = probe(layer_6_hidden)
    loss = nn.CrossEntropyLoss()(logits, batch['labels'])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

### 7. **`return_dict`**

Return structured output (dataclass) vs tuple.

#### ✅ When to use `True` (default, recommended):
- **Modern code** - cleaner, more readable
- **Named access** - `outputs.logits` vs `outputs[0]`
- **Most use cases** - default behavior

#### ✅ When to use `False`:
- **Legacy code** - compatibility
- **Specific tuple unpacking needs** - rare

```python
# ✅ GOOD: Using return_dict=True (default, recommended)
outputs = model(input_ids=input_ids, return_dict=True)

# Clean named access
logits = outputs.logits
past_key_values = outputs.past_key_values
hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None

# ✅ GOOD: With labels (training)
outputs = model(
    input_ids=input_ids,
    labels=labels,
    return_dict=True
)
loss = outputs.loss
logits = outputs.logits

# ❌ UGLY: Using return_dict=False (legacy)
outputs = model(input_ids=input_ids, return_dict=False)
# Returns tuple - need to remember order!
# (logits, past_key_values, hidden_states, attentions, ...)
logits = outputs[0]
past_key_values = outputs[1]  # What index was this again?

# ✅ GOOD: Modern idiom with return_dict
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, return_dict=True)
    predictions = outputs.logits.argmax(dim=-1)
    return predictions

# ✅ GOOD: Type hints with return_dict
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

def generate_with_analysis(input_ids) -> CausalLMOutputWithCrossAttentions:
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        output_attentions=True,
        return_dict=True  # Returns proper dataclass
    )
    return outputs
```

---

### 8. **`position_ids` and `token_type_ids`**

Manual position and segment embeddings.

#### ✅ When to use `position_ids`:
- **Non-standard position encoding** - e.g., relative positions
- **Rotary embeddings experiments** - custom positions
- **Rare advanced use cases**

#### ✅ When to use `token_type_ids`:
- **Multi-segment inputs** - dialogue systems (but GPT-2 rarely uses this)
- **BERT-style segment embeddings** - not typical for GPT-2

#### ❌ When NOT to use (99% of cases):
- **Standard usage** - automatically computed
- **Normal text generation** - unnecessary

```python
# ✅ DEFAULT: Position IDs computed automatically
input_ids = tokenizer.encode("Hello world", return_tensors="pt")
outputs = model(input_ids=input_ids)
# Positions [0, 1] used automatically

# ✅ RARE: Custom position IDs (advanced)
input_ids = tokenizer.encode("Hello world", return_tensors="pt")
# Use same position for all tokens (experimental)
position_ids = torch.zeros_like(input_ids)
outputs = model(input_ids=input_ids, position_ids=position_ids)

# ✅ RARE: Relative positions (advanced)
# Simulate position wrapping
input_ids = tokenizer.encode("A" * 100, return_tensors="pt")
position_ids = torch.arange(100).unsqueeze(0) % 50  # Wrap at 50
outputs = model(input_ids=input_ids, position_ids=position_ids)

# ❌ TYPICALLY NOT NEEDED: token_type_ids for GPT-2
# GPT-2 doesn't typically use segment embeddings like BERT
# But you can add them if model is configured with token_type_ids
input_ids = tokenizer.encode("Segment 1. Segment 2.", return_tensors="pt")
token_type_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])  # Two segments
outputs = model(
    input_ids=input_ids,
    token_type_ids=token_type_ids  # Usually not used for GPT-2
)
```

---

## Model-Specific Arguments

### 9. **`mc_token_ids` and `mc_labels`** (GPT2DoubleHeadsModel only)

For multiple choice classification.

```python
from transformers import GPT2DoubleHeadsModel

model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")

# ✅ GOOD: Multiple choice task
# Add a [CLS] token
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
model.resize_token_embeddings(len(tokenizer))

choices = [
    "Paris is the capital of France [CLS]",
    "London is the capital of France [CLS]"
]

# Tokenize choices
encoded_choices = [tokenizer.encode(s) for s in choices]
cls_positions = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # [1, 2, seq_len]
mc_token_ids = torch.tensor([cls_positions])            # [1, 2]
mc_labels = torch.tensor([0])                           # Correct choice

outputs = model(
    input_ids=input_ids,
    mc_token_ids=mc_token_ids,
    mc_labels=mc_labels
)

lm_loss = outputs.loss      # Language modeling loss
mc_loss = outputs.mc_loss   # Multiple choice loss
mc_logits = outputs.mc_logits  # [batch_size, num_choices]
```

---

### 10. **`start_positions` and `end_positions`** (GPT2ForQuestionAnswering)

For span-based question answering.

```python
from transformers import GPT2ForQuestionAnswering

model = GPT2ForQuestionAnswering.from_pretrained("openai-community/gpt2")

# ✅ GOOD: Question answering
context = "Paris is the capital of France."
question = "What is the capital of France?"

inputs = tokenizer(question, context, return_tensors="pt")

# Suppose "Paris" spans positions [0, 1]
start_positions = torch.tensor([0])
end_positions = torch.tensor([1])

outputs = model(
    **inputs,
    start_positions=start_positions,
    end_positions=end_positions
)

loss = outputs.loss
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Prediction
start_idx = start_logits.argmax()
end_idx = end_logits.argmax()
answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
answer = tokenizer.decode(answer_tokens)
```

---

## 📋 Complete Usage Recipes

### **Recipe 1: Standard Text Generation**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

text = "Once upon a time"
input_ids = tokenizer.encode(text, return_tensors="pt")

# Generate
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=3,
    temperature=0.8,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    use_cache=True,  # Fast generation
    pad_token_id=tokenizer.eos_token_id
)

for i, seq in enumerate(output):
    print(f"Generated {i+1}: {tokenizer.decode(seq, skip_special_tokens=True)}")
```

### **Recipe 2: Training Language Model**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Prepare data
def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

dataset = load_dataset("your_dataset")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    eval_steps=500,
    fp16=True,  # Mixed precision
    gradient_checkpointing=True,  # Save memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()
```

### **Recipe 3: Sequence Classification**
```python
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from torch.utils.data import DataLoader

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2ForSequenceClassification.from_pretrained(
    "openai-community/gpt2",
    num_labels=3  # 3-class classification
)
model.config.pad_token_id = tokenizer.pad_token_id

# Prepare data
texts = ["Great product!", "Terrible service.", "It's okay."]
labels = torch.tensor([2, 0, 1])  # Positive, Negative, Neutral

# Tokenize with padding
encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Forward pass
outputs = model(
    input_ids=encodings['input_ids'],
    attention_mask=encodings['attention_mask'],  # IMPORTANT for padding
    labels=labels
)

loss = outputs.loss
logits = outputs.logits
predictions = logits.argmax(dim=-1)

print(f"Loss: {loss.item():.4f}")
print(f"Predictions: {predictions}")
```

### **Recipe 4: Interactive Chatbot with Context**
```python
class GPT2Chatbot:
    def __init__(self, model_name="openai-community/gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.conversation_history = ""
    
    def chat(self, user_input, max_length=100):
        # Add user input to history
        self.conversation_history += f"User: {user_input}\nBot: "
        
        # Encode
        input_ids = self.tokenizer.encode(
            self.conversation_history,
            return_tensors="pt"
        )
        
        # Truncate if too long
        if input_ids.shape[1] > 512:
            input_ids = input_ids[:, -512:]
        
        # Generate response
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True  # Faster
        )
        
        # Decode response
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        bot_response = full_response[len(self.conversation_history):].split('\n')[0]
        
        # Update history
        self.conversation_history += bot_response + "\n"
        
        return bot_response
    
    def reset(self):
        self.conversation_history = ""

# Usage
chatbot = GPT2Chatbot()
print(chatbot.chat("Hello!"))
print(chatbot.chat("How are you?"))
chatbot.reset()
```

---

## 🎯 Key Takeaways

1. **`input_ids` is required** - main input for all tasks
2. **`attention_mask` with padding** - critical for batching
3. **`past_key_values` for speed** - essential for generation (10-100x faster)
4. **`labels` for training** - automatic loss computation
5. **`use_cache=True` for generation**, `False` for training
6. **`output_attentions/hidden_states`** - only for analysis, not production
7. **`return_dict=True`** - modern, readable (default)
8. **`position_ids/token_type_ids`** - rarely needed, auto-computed
9. **Always use `attention_mask`** with variable-length batches
10. **Disable cache** with gradient checkpointing
