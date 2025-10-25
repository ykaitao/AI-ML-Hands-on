# Best Practices for GPT2Config Arguments

This is a comprehensive guide for configuring GPT-2 models. I'll cover each parameter group with practical examples and use cases.

---

## 1. **Model Architecture Parameters**

### **`vocab_size`** (default: 50257)

Number of tokens in the vocabulary.

#### ✅ When to modify:
- **Custom tokenizer** - you trained your own with different vocab size
- **Domain-specific vocabulary** - medical, legal, code
- **Multilingual models** - need more tokens for multiple languages
- **Memory optimization** - smaller vocab for constrained devices

#### ❌ When NOT to modify (keep 50257):
- **Loading pre-trained GPT-2** - must match checkpoint
- **Fine-tuning** - keep original vocab size
- **Transfer learning** - maintain compatibility

```python
from transformers import GPT2Config, GPT2LMHeadModel

# ✅ GOOD: Loading pre-trained (don't specify vocab_size)
config = GPT2Config.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# ✅ GOOD: Training from scratch with custom tokenizer
config = GPT2Config(
    vocab_size=30000,  # Your custom tokenizer size
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)
model = GPT2LMHeadModel(config)

# ✅ GOOD: Smaller model for resource-constrained deployment
config = GPT2Config(
    vocab_size=10000,  # Reduced vocabulary
    n_embd=512,        # Smaller model
    n_layer=6
)

# ❌ BAD: Changing vocab_size when loading pre-trained
config = GPT2Config.from_pretrained("openai-community/gpt2")
config.vocab_size = 30000  # Mismatch! Will crash
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", config=config)

# ✅ GOOD: Expanding vocab for fine-tuning with new tokens
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# Add new tokens
new_tokens = ['<user>', '<assistant>', '<code>']
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))
print(f"New vocab size: {len(tokenizer)}")  # 50257 + 3 = 50260
```

---

### **`n_positions`** (default: 1024)

Maximum sequence length the model can handle.

#### ✅ When to increase (2048, 4096, 8192):
- **Long documents** - research papers, legal documents
- **Code generation** - long functions/files
- **Conversation history** - chat applications
- **Book/article generation** - long-form content

#### ✅ When to decrease (512, 256):
- **Short texts** - tweets, product reviews
- **Memory constraints** - limited GPU/RAM
- **Faster inference** - smaller attention matrices

#### ⚠️ Caution:
- Attention complexity is O(n²), so 2x length = 4x memory
- Pre-trained models can't handle longer sequences without retraining

```python
# ✅ GOOD: Training for long documents
config = GPT2Config(
    vocab_size=50257,
    n_positions=2048,  # 2x longer than GPT-2
    n_embd=768,
    n_layer=12,
    n_head=12
)
model = GPT2LMHeadModel(config)

# ✅ GOOD: Efficient model for short text classification
config = GPT2Config(
    vocab_size=50257,
    n_positions=256,   # Shorter sequences
    n_embd=512,        # Smaller embeddings
    n_layer=6          # Fewer layers
)

# ❌ BAD: Using pre-trained model beyond its position limit
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
# model.config.n_positions = 1024
long_text = "word " * 2000  # 2000 tokens - will crash!
tokens = tokenizer(long_text, return_tensors="pt")
output = model(**tokens)  # Error: position embeddings only go up to 1024

# ✅ GOOD: Proper handling of long sequences
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
long_text = "word " * 2000

# Option 1: Truncate
tokens = tokenizer(
    long_text,
    max_length=1024,
    truncation=True,
    return_tensors="pt"
)

# Option 2: Sliding window
def process_long_text(text, max_length=1024, stride=512):
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    
    results = []
    for i in range(0, len(input_ids), stride):
        chunk = input_ids[i:i+max_length]
        output = model(chunk.unsqueeze(0))
        results.append(output)
    return results

# Option 3: Use a model designed for long sequences
# from transformers import GPTNeoForCausalLM
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
# # GPT-Neo supports 2048 positions
```

---

### **`n_embd`** (default: 768)

Hidden size / embedding dimension.

#### ✅ Size guidelines:
- **Tiny (128-256)**: Experimental, mobile devices
- **Small (384-512)**: Edge devices, fast inference
- **Medium (768)**: GPT-2 base, good balance
- **Large (1024-1280)**: GPT-2 medium/large
- **XL (1600+)**: GPT-2 XL, research

```python
# ✅ GOOD: Small efficient model
config = GPT2Config(
    vocab_size=50257,
    n_positions=512,
    n_embd=384,      # Small
    n_layer=6,
    n_head=6         # n_head must divide n_embd
)

# ✅ GOOD: Scaling up (must scale n_head proportionally)
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=1280,     # Larger
    n_layer=36,
    n_head=20        # 1280 / 20 = 64 (head dimension)
)

# ❌ BAD: n_head doesn't divide n_embd
config = GPT2Config(
    n_embd=768,
    n_head=10        # 768 / 10 = 76.8 - not an integer!
)
# Will raise error during model initialization

# ✅ GOOD: Valid combinations
# n_embd=768, n_head=12 → head_dim=64
# n_embd=1024, n_head=16 → head_dim=64
# n_embd=1536, n_head=24 → head_dim=64
```

---

### **`n_layer`** (default: 12)

Number of transformer layers.

#### ✅ Guidelines:
- **Shallow (4-6)**: Fast, simple tasks, distillation
- **Medium (12)**: GPT-2 base, balanced
- **Deep (24-36)**: Complex reasoning, large models
- **Very deep (48+)**: Research, massive compute

```python
# ✅ GOOD: Shallow model for simple classification
config = GPT2Config(
    vocab_size=50257,
    n_positions=512,
    n_embd=512,
    n_layer=6,       # Shallow
    n_head=8
)

# ✅ GOOD: Deep model for complex generation
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=1280,
    n_layer=36,      # Deep
    n_head=20
)

# 💡 TIP: Layer scaling impacts
print(f"Parameters ∝ n_layer")
# 12 layers: ~117M parameters (GPT-2 base)
# 24 layers: ~345M parameters (GPT-2 medium)
# 36 layers: ~762M parameters (GPT-2 large)
```

---

### **`n_head`** (default: 12)

Number of attention heads.

#### ✅ Best practices:
- Must evenly divide `n_embd`
- Common head dimension: 64 (e.g., 768/12=64, 1024/16=64)
- More heads = more diverse attention patterns

```python
# ✅ GOOD: Standard configurations
configs_valid = [
    (768, 12),    # GPT-2 base
    (1024, 16),   # GPT-2 medium
    (1280, 20),   # GPT-2 large
    (1600, 25),   # GPT-2 XL
]

for n_embd, n_head in configs_valid:
    head_dim = n_embd // n_head
    print(f"n_embd={n_embd}, n_head={n_head}, head_dim={head_dim}")
    assert n_embd % n_head == 0

# ❌ BAD: Invalid configurations
configs_invalid = [
    (768, 10),    # 768/10 = 76.8
    (1024, 15),   # 1024/15 = 68.27
]
```

---

### **`n_inner`** (default: None → 4 * n_embd)

FFN (feed-forward network) intermediate dimension.

#### ✅ When to modify:
- **Increase (5-8x n_embd)**: More model capacity
- **Decrease (2-3x n_embd)**: Parameter efficiency
- **None**: Standard 4x multiplier (recommended)

```python
# ✅ GOOD: Default (4x)
config = GPT2Config(
    n_embd=768,
    n_inner=None     # Automatically becomes 3072 (4 * 768)
)

# ✅ GOOD: Increase capacity
config = GPT2Config(
    n_embd=768,
    n_inner=4096     # 5.33x instead of 4x
)

# ✅ GOOD: Parameter-efficient model
config = GPT2Config(
    n_embd=768,
    n_inner=2048     # 2.67x for efficiency
)

# 💡 Impact on parameters
# n_inner is in 2 linear layers per layer
# Parameters per layer ≈ 2 * n_embd * n_inner
# For n_embd=768:
#   n_inner=3072 → ~4.7M params/layer
#   n_inner=2048 → ~3.1M params/layer
#   n_inner=4096 → ~6.3M params/layer
```

---

## 2. **Dropout Parameters**

### **`resid_pdrop`, `embd_pdrop`, `attn_pdrop`** (default: 0.1)

Dropout rates for residual connections, embeddings, and attention.

#### ✅ When to increase (0.2-0.3):
- **Small datasets** - prevent overfitting
- **Complex models** - large n_embd/n_layer
- **Training from scratch** - more regularization

#### ✅ When to decrease (0.05) or disable (0.0):
- **Large datasets** - less overfitting risk
- **Inference** - dropout is automatically disabled
- **Fine-tuning pre-trained** - already regularized

```python
# ✅ GOOD: Training from scratch on small dataset
config = GPT2Config(
    vocab_size=50257,
    n_embd=768,
    n_layer=12,
    resid_pdrop=0.2,   # Higher dropout
    embd_pdrop=0.2,
    attn_pdrop=0.2
)

# ✅ GOOD: Fine-tuning pre-trained model
config = GPT2Config.from_pretrained("openai-community/gpt2")
config.resid_pdrop = 0.05  # Lower dropout for fine-tuning
config.embd_pdrop = 0.05
config.attn_pdrop = 0.05

# ✅ GOOD: Large dataset training
config = GPT2Config(
    vocab_size=50257,
    n_embd=1024,
    n_layer=24,
    resid_pdrop=0.1,   # Standard dropout
    embd_pdrop=0.1,
    attn_pdrop=0.1
)

# ✅ GOOD: Inference (dropout auto-disabled)
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
model.eval()  # Disables dropout automatically
with torch.no_grad():
    output = model(**inputs)

# ❌ BAD: Extremely high dropout
config = GPT2Config(
    resid_pdrop=0.5,   # Too high! Model won't learn well
    embd_pdrop=0.5,
    attn_pdrop=0.5
)
```

---

## 3. **Activation and Normalization**

### **`activation_function`** (default: "gelu_new")

Activation function in FFN.

#### ✅ Options:
- **"gelu_new"**: GPT-2 default, smooth gradient
- **"gelu"**: Original GELU, slightly different
- **"relu"**: Faster but less expressive
- **"silu"** (Swish): Modern alternative
- **"tanh"**: Legacy, rarely used

```python
# ✅ GOOD: Standard GPT-2 (default)
config = GPT2Config(activation_function="gelu_new")

# ✅ GOOD: Fast inference with ReLU
config = GPT2Config(activation_function="relu")

# ✅ GOOD: Modern SiLU/Swish
config = GPT2Config(activation_function="silu")

# ⚠️ CAUTION: Don't change for pre-trained models
config = GPT2Config.from_pretrained("openai-community/gpt2")
config.activation_function = "relu"  # Will break compatibility!
```

### **`layer_norm_epsilon`** (default: 1e-5)

Epsilon for numerical stability in layer normalization.

#### ✅ When to modify:
- **Mixed precision training**: Increase to 1e-4 or 1e-3
- **Numerical instability**: Increase if seeing NaNs
- **Default**: Usually leave at 1e-5

```python
# ✅ GOOD: Mixed precision training (fp16)
config = GPT2Config(
    layer_norm_epsilon=1e-4  # More stable for fp16
)

# ✅ GOOD: Default for fp32
config = GPT2Config(
    layer_norm_epsilon=1e-5  # Standard
)
```

---

## 4. **Initialization**

### **`initializer_range`** (default: 0.02)

Standard deviation for weight initialization.

#### ✅ When to modify:
- **Larger models**: Decrease (0.01-0.015) for stability
- **Smaller models**: Can increase slightly (0.025-0.03)
- **Default**: Usually 0.02 works well

```python
# ✅ GOOD: Large model initialization
config = GPT2Config(
    n_embd=1600,
    n_layer=48,
    initializer_range=0.01  # Smaller for stability
)

# ✅ GOOD: Small model
config = GPT2Config(
    n_embd=384,
    n_layer=6,
    initializer_range=0.025  # Slightly larger
)

# ✅ GOOD: Standard (default)
config = GPT2Config(initializer_range=0.02)
```

---

## 5. **Attention Optimization**

### **`scale_attn_weights`** (default: True)

Scale attention by sqrt(head_dim).

#### ✅ When to use:
- **True (default)**: Always recommended, prevents attention saturation
- **False**: Only for research/experimental purposes

```python
# ✅ GOOD: Keep enabled (default)
config = GPT2Config(scale_attn_weights=True)

# ❌ BAD: Disabling (not recommended)
config = GPT2Config(scale_attn_weights=False)
```

### **`scale_attn_by_inverse_layer_idx`** (default: False)

Scale attention by 1/(layer_idx + 1).

#### ✅ When to use:
- **Deep models (24+ layers)**: Can help training stability
- **Experimental**: Research purposes
- **False (default)**: Standard for most uses

```python
# ✅ GOOD: Very deep model
config = GPT2Config(
    n_layer=48,
    scale_attn_by_inverse_layer_idx=True  # Helps deep models
)

# ✅ GOOD: Standard model (default)
config = GPT2Config(
    n_layer=12,
    scale_attn_by_inverse_layer_idx=False
)
```

### **`reorder_and_upcast_attn`** (default: False)

Upcast attention to fp32 for mixed precision training.

#### ✅ When to use:
- **Mixed precision (fp16/bf16)**: Prevents numerical issues
- **False**: Standard fp32 training

```python
# ✅ GOOD: Mixed precision training
config = GPT2Config(
    reorder_and_upcast_attn=True  # Stability for fp16
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    fp16=True,  # Mixed precision
    # ... other args
)

# ✅ GOOD: Standard fp32 training
config = GPT2Config(
    reorder_and_upcast_attn=False  # Not needed for fp32
)
```

---

## 6. **Caching and Token IDs**

### **`use_cache`** (default: True)

Cache past key-values for faster generation.

#### ✅ When to enable (True):
- **Text generation**: Dramatically faster
- **Inference**: Always enable
- **Interactive chat**: Essential

#### ✅ When to disable (False):
- **Training**: Saves memory
- **Batch inference on short sequences**: Minimal benefit

```python
# ✅ GOOD: Generation with caching (fast)
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
model.config.use_cache = True

outputs = model.generate(
    input_ids,
    max_length=100,
    use_cache=True  # Reuses past key-values
)

# ✅ GOOD: Training without caching (saves memory)
config = GPT2Config.from_pretrained("openai-community/gpt2")
config.use_cache = False
model = GPT2LMHeadModel(config)

# Training loop
model.train()
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
```

### **`bos_token_id` and `eos_token_id`** (default: 50256)

Beginning and end of sequence token IDs.

```python
# ✅ GOOD: Standard GPT-2
config = GPT2Config(
    bos_token_id=50256,  # <|endoftext|>
    eos_token_id=50256   # Same token
)

# ✅ GOOD: Custom tokens
config = GPT2Config(
    vocab_size=50260,     # Extended vocab
    bos_token_id=50257,   # New <BOS>
    eos_token_id=50258    # New <EOS>
)

# ✅ GOOD: Generation with EOS
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
outputs = model.generate(
    input_ids,
    max_length=100,
    eos_token_id=tokenizer.eos_token_id,  # Stop at EOS
    pad_token_id=tokenizer.eos_token_id   # Use EOS as PAD
)
```

---

## 7. **Summary Parameters (for GPT2DoubleHeadsModel)**

These are only used for the double-heads variant (language modeling + classification).

### **`summary_type`**, **`summary_use_proj`**, etc.

```python
# ✅ GOOD: Using GPT2DoubleHeadsModel for multi-task
from transformers import GPT2DoubleHeadsModel

config = GPT2Config(
    summary_type="cls_index",      # Use specific position
    summary_use_proj=True,          # Add projection layer
    summary_activation=None,        # No activation
    summary_proj_to_labels=True,    # Project to num_labels
    summary_first_dropout=0.1       # Dropout before projection
)
model = GPT2DoubleHeadsModel(config)

# ❌ NOT NEEDED: Regular GPT2LMHeadModel ignores these
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
# summary_* parameters have no effect
```

---

## 📋 Complete Configuration Recipes

### **Recipe 1: Small Efficient Model**
```python
# For mobile/edge deployment
config = GPT2Config(
    vocab_size=30000,        # Smaller vocab
    n_positions=512,         # Shorter sequences
    n_embd=384,              # Small hidden size
    n_layer=6,               # Shallow
    n_head=6,                # 384/6 = 64
    n_inner=1536,            # 4 * 384
    activation_function="relu",  # Fast
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    use_cache=True           # Fast generation
)
model = GPT2LMHeadModel(config)
print(f"Parameters: {model.num_parameters():,}")  # ~40M
```

### **Recipe 2: Medium Balanced Model**
```python
# Good balance of quality and speed
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    use_cache=True
)
model = GPT2LMHeadModel(config)
print(f"Parameters: {model.num_parameters():,}")  # ~117M
```

### **Recipe 3: Large High-Quality Model**
```python
# For maximum quality (requires significant compute)
config = GPT2Config(
    vocab_size=50257,
    n_positions=2048,         # Longer context
    n_embd=1600,              # Larger
    n_layer=48,               # Deep
    n_head=25,                # 1600/25 = 64
    n_inner=6400,             # 4 * 1600
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-4,  # Stable for mixed precision
    initializer_range=0.01,   # Smaller for stability
    scale_attn_by_inverse_layer_idx=True,  # Help deep model
    reorder_and_upcast_attn=True,  # Mixed precision
    use_cache=True
)
model = GPT2LMHeadModel(config)
print(f"Parameters: {model.num_parameters():,}")  # ~1.5B
```

### **Recipe 4: Training from Scratch on Domain Data**
```python
# Medical/legal/code domain
config = GPT2Config(
    vocab_size=32000,         # Custom tokenizer
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    activation_function="gelu_new",
    resid_pdrop=0.15,         # Higher dropout (smaller domain)
    embd_pdrop=0.15,
    attn_pdrop=0.15,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=0,           # Custom special tokens
    eos_token_id=1,
    use_cache=True
)
model = GPT2LMHeadModel(config)
```

### **Recipe 5: Fine-tuning Pre-trained Model**
```python
# Load and modify config for fine-tuning
config = GPT2Config.from_pretrained("openai-community/gpt2")

# Adjust dropout (optional)
config.resid_pdrop = 0.05
config.embd_pdrop = 0.05
config.attn_pdrop = 0.05

# Load model with modified config
model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2",
    config=config
)

# Add new tokens if needed
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
model.resize_token_embeddings(len(tokenizer))
```

---

## 🎯 Key Takeaways

1. **Don't modify for pre-trained models** - breaks checkpoint compatibility
2. **`n_embd` must be divisible by `n_head`** - common head_dim is 64
3. **`n_positions` is max sequence length** - attention is O(n²)
4. **Higher dropout for small data** - prevents overfitting
5. **`use_cache=True` for generation** - dramatically faster
6. **`reorder_and_upcast_attn=True` for mixed precision** - numerical stability
7. **Scale model proportionally** - increase all dims together
8. **`vocab_size` must match tokenizer** - critical for compatibility
