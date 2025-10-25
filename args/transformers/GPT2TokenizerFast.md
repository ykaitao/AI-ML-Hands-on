# Best Practices for GPT2TokenizerFast `__init__()` Arguments

```python
from transformers import GPT2TokenizerFast
```

Based on the code, here's a comprehensive guide for the GPT2TokenizerFast initialization arguments:

---

## 1. **`vocab_file` and `merges_file`**

### ✅ When to use:
- **Loading custom-trained tokenizers** - you trained your own BPE model
- **Backwards compatibility** - working with older tokenizer formats
- **Explicit file management** - you want to manage vocab/merges separately
- **Debugging/inspection** - easier to examine plain JSON/text files

### ❌ When NOT to use (use `tokenizer_file` instead):
- **Loading pre-trained models** - `from_pretrained()` handles this automatically
- **Modern workflows** - `tokenizer_file` is more efficient and self-contained
- **Production deployment** - single file is simpler

```python
from transformers import GPT2TokenizerFast

# ✅ GOOD: Loading pre-trained (most common) - DON'T specify files
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
# Automatically loads vocab, merges, and config

# ✅ GOOD: Custom trained tokenizer with separate files
tokenizer = GPT2TokenizerFast(
    vocab_file="my_custom_vocab.json",
    merges_file="my_custom_merges.txt",
    unk_token="<|endoftext|>",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>"
)

# ✅ BETTER: Using tokenizer_file (modern approach)
tokenizer = GPT2TokenizerFast(
    tokenizer_file="tokenizer.json",  # All-in-one file
    unk_token="<|endoftext|>",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>"
)

# ❌ BAD: Manually specifying files for pre-trained models
tokenizer = GPT2TokenizerFast(
    vocab_file="path/to/gpt2/vocab.json",  # Unnecessary!
    merges_file="path/to/gpt2/merges.txt"
)
# Just use from_pretrained() instead

# ❌ BAD: Mixing tokenizer_file with vocab_file/merges_file
tokenizer = GPT2TokenizerFast(
    vocab_file="vocab.json",
    merges_file="merges.txt",
    tokenizer_file="tokenizer.json"  # Conflicting! Which should it use?
)
```

---

## 2. **`tokenizer_file`**

### ✅ When to use:
- **Modern tokenizer format** - single JSON file with everything
- **Faster loading** - pre-compiled tokenizer state
- **From `tokenizers` library** - you trained with HuggingFace tokenizers
- **Deployment** - single file to manage

### ❌ When NOT to use:
- **Legacy systems** - if you only have vocab.json + merges.txt
- **Debugging vocab** - harder to inspect than plain JSON/text files

```python
# ✅ GOOD: Loading from tokenizer.json (recommended)
tokenizer = GPT2TokenizerFast(
    tokenizer_file="path/to/tokenizer.json"
)

# ✅ GOOD: Save and reload with tokenizer.json
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.save_pretrained("my_tokenizer")
# Creates: tokenizer.json, tokenizer_config.json, special_tokens_map.json

# Reload
tokenizer = GPT2TokenizerFast.from_pretrained("my_tokenizer")

# ✅ GOOD: Training and saving
from tokenizers import ByteLevelBPETokenizer

# Train
trainer = ByteLevelBPETokenizer()
trainer.train(files=["corpus.txt"], vocab_size=30000)
trainer.save("my_tokenizer.json")

# Load in transformers
tokenizer = GPT2TokenizerFast(
    tokenizer_file="my_tokenizer.json",
    unk_token="<|endoftext|>",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>"
)
```

---

## 3. **`unk_token`, `bos_token`, `eos_token`**

### 💡 GPT-2 Default Behavior:
- All three use the same token: `"<|endoftext|>"`
- This is intentional in GPT-2's design

### ✅ When to use defaults (`"<|endoftext|>"`):
- **GPT-2 compatible models** - maintain compatibility
- **Text generation** - standard GPT-2 behavior
- **Fine-tuning GPT-2** - keep original special tokens

### ✅ When to customize:
- **Different model architecture** - BERT, T5, etc. use different tokens
- **Multi-task learning** - distinguish between tasks with different markers
- **Custom training** - you defined your own special tokens

```python
# ✅ GOOD: Standard GPT-2 (default)
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
# unk_token = bos_token = eos_token = "<|endoftext|>"

# ✅ GOOD: Custom tokens for specific use case
tokenizer = GPT2TokenizerFast(
    tokenizer_file="custom_tokenizer.json",
    unk_token="[UNK]",        # BERT-style
    bos_token="[CLS]",        # Beginning of sequence
    eos_token="[SEP]"         # End of sequence
)

# ✅ GOOD: Instruction-following model (ChatGPT-style)
tokenizer = GPT2TokenizerFast(
    tokenizer_file="tokenizer.json",
    unk_token="<|endoftext|>",
    bos_token="<|im_start|>",    # Instruction start
    eos_token="<|im_end|>"       # Instruction end
)

# ⚠️ CAUTION: Changing tokens for pre-trained GPT-2
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    unk_token="[UNK]",  # Model wasn't trained with this!
    bos_token="[BOS]",
    eos_token="[EOS]"
)
# This will break model performance - don't change for existing models

# ✅ GOOD: Adding NEW special tokens (not replacing)
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',        # GPT-2 doesn't have pad by default
    'additional_special_tokens': ['<|user|>', '<|assistant|>']
})
# Remember to resize model embeddings: model.resize_token_embeddings(len(tokenizer))
```

---

## 4. **`add_prefix_space`**

This is the **most critical** parameter for GPT2TokenizerFast.

### ✅ When to use `True`:
- **Pre-tokenized inputs** - `is_split_into_words=True` (REQUIRED!)
- **Consistent tokenization** - treat first word like all others
- **Word-level tasks** - NER, POS tagging on word lists
- **Batch processing with different lengths** - avoid position-dependent tokenization

### ❌ When to use `False` (default):
- **Standard text generation** - GPT-2's expected behavior
- **Loading pre-trained GPT-2** - model was trained without prefix space
- **Single continuous text** - not pre-tokenized
- **Maximum compatibility** - default behavior

```python
# ❌ DEFAULT: False - position matters (GPT-2 standard)
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
print(tokenizer("Hello world")["input_ids"])        # [15496, 995]
print(tokenizer(" Hello world")["input_ids"])       # [18435, 995]
# Different tokenization! First word gets different token

# ✅ GOOD: add_prefix_space=True for consistency
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_prefix_space=True
)
print(tokenizer("Hello world")["input_ids"])        # [15496, 995]
print(tokenizer(" Hello world")["input_ids"])       # [15496, 995]
# Same tokenization regardless of leading space

# ✅ REQUIRED: Pre-tokenized inputs (will crash without add_prefix_space=True)
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_prefix_space=True  # MUST set this!
)
words = ["Hello", "world", "!"]
encoding = tokenizer(
    words,
    is_split_into_words=True  # Requires add_prefix_space=True
)

# ❌ BAD: Will raise assertion error!
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")  # add_prefix_space=False
words = ["Hello", "world"]
encoding = tokenizer(words, is_split_into_words=True)
# AssertionError: You need to instantiate GPT2TokenizerFast with add_prefix_space=True

# ✅ GOOD: NER task with pre-tokenized text
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_prefix_space=True
)
words = ["John", "Smith", "lives", "in", "Paris"]
labels = ["B-PER", "I-PER", "O", "O", "B-LOC"]

encoding = tokenizer(
    words,
    is_split_into_words=True,
    return_tensors="pt"
)

# ⚠️ CAUTION: Performance impact
# GPT-2 was trained WITHOUT prefix space
# Using add_prefix_space=True may slightly degrade performance
# Only use when necessary (pre-tokenized inputs)

# ✅ WORKAROUND: If you need both behaviors
tokenizer_standard = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer_pretokenized = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_prefix_space=True
)

# Use standard for generation
text = "Once upon a time"
tokens = tokenizer_standard(text)

# Use pretokenized for NER
words = ["John", "lives", "here"]
tokens = tokenizer_pretokenized(words, is_split_into_words=True)
```

---

## 5. **`add_bos_token`** (Hidden parameter in kwargs)

### ✅ When to use `True`:
- **Sequence classification** - need explicit start marker
- **Encoder-decoder models** - decoder needs BOS
- **Chat/instruction models** - mark conversation start
- **Custom training** - if your model expects BOS at start

### ❌ When to use `False` (default):
- **Standard GPT-2 generation** - doesn't use BOS in practice
- **Pre-trained GPT-2** - model wasn't trained with BOS at start
- **Concatenating sequences** - don't want BOS in the middle

```python
# ❌ DEFAULT: False - no BOS added automatically
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokens = tokenizer("Hello world")
# No BOS token at start: [15496, 995]

# ✅ GOOD: Adding BOS for classification
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_bos_token=True
)
tokens = tokenizer("Hello world")
# BOS added: [50256, 15496, 995] where 50256 is <|endoftext|>

# ✅ GOOD: Manual control (more flexible)
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
text = "Hello world"
# Add BOS manually when needed
input_ids = [tokenizer.bos_token_id] + tokenizer.encode(text)

# ✅ GOOD: Chat template with explicit BOS
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_bos_token=True
)
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
# Each message gets BOS

# ❌ BAD: Using add_bos_token with concatenated texts
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_bos_token=True
)
texts = ["Part 1", "Part 2", "Part 3"]
all_tokens = []
for text in texts:
    all_tokens.extend(tokenizer.encode(text))
# Problem: BOS inserted at start of each part!
# Result: [BOS, "Part", "1", BOS, "Part", "2", BOS, "Part", "3"]

# ✅ BETTER: Control BOS placement
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
all_tokens = [tokenizer.bos_token_id]  # One BOS at start
for text in texts:
    all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
```

---

## 📋 Common Configuration Recipes

### **Recipe 1: Standard GPT-2 Inference**
```python
# Most common use case
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
# Defaults:
# - add_prefix_space=False
# - add_bos_token=False
# - unk/bos/eos_token="<|endoftext|>"

text = "Hello, how are you?"
tokens = tokenizer(text, return_tensors="pt")
```

### **Recipe 2: Fine-tuning GPT-2 for Classification**
```python
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

# Add padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Or add a new pad token
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
# Don't forget: model.resize_token_embeddings(len(tokenizer))

# Use for classification
texts = ["Text 1", "Text 2", "Text 3"]
encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

### **Recipe 3: NER / Token Classification**
```python
# MUST use add_prefix_space=True for pre-tokenized inputs
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_prefix_space=True  # Required!
)

# Pre-tokenized words
words = ["John", "Smith", "works", "at", "Google"]
labels = ["B-PER", "I-PER", "O", "O", "B-ORG"]

encoding = tokenizer(
    words,
    is_split_into_words=True,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

# Handle subword tokenization - align labels
word_ids = encoding.word_ids()
aligned_labels = []
for word_id in word_ids:
    if word_id is None:
        aligned_labels.append(-100)  # Ignore special tokens
    else:
        aligned_labels.append(labels[word_id])
```

### **Recipe 4: Custom Trained Tokenizer**
```python
from tokenizers import ByteLevelBPETokenizer

# Train custom tokenizer
trainer = ByteLevelBPETokenizer()
trainer.train(
    files=["my_corpus.txt"],
    vocab_size=30000,
    special_tokens=["<|endoftext|>", "<|pad|>"]
)
trainer.save_model("my_tokenizer")

# Load in transformers
tokenizer = GPT2TokenizerFast(
    vocab_file="my_tokenizer/vocab.json",
    merges_file="my_tokenizer/merges.txt",
    unk_token="<|endoftext|>",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
    add_prefix_space=False
)

# Save in transformers format
tokenizer.save_pretrained("my_tokenizer_transformers")
```

### **Recipe 5: Chat/Instruction Model**
```python
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

# Add special tokens for chat
special_tokens = {
    'pad_token': '<|pad|>',
    'additional_special_tokens': [
        '<|im_start|>',  # Instruction start
        '<|im_end|>',    # Instruction end
        '<|user|>',
        '<|assistant|>',
        '<|system|>'
    ]
}
tokenizer.add_special_tokens(special_tokens)
# Resize model: model.resize_token_embeddings(len(tokenizer))

# Format conversation
def format_chat(messages):
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"}
]

text = format_chat(messages)
tokens = tokenizer(text, return_tensors="pt")
```

---

## 🎯 Key Takeaways

1. **`from_pretrained()` is preferred** - automatically handles all file loading
2. **`add_prefix_space=True` is REQUIRED** for `is_split_into_words=True`
3. **Don't change special tokens** for pre-trained models (breaks performance)
4. **Add padding token** - GPT-2 doesn't have one by default, needed for batching
5. **`tokenizer_file` > `vocab_file` + `merges_file`** - more efficient, modern
6. **`add_bos_token`** - usually False for GPT-2, True for classification/chat
7. **Performance warning** - `add_prefix_space=True` may hurt generation quality
