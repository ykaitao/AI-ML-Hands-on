# ByteLevelBPETokenizer `__init__()` Parameters

```python
from tokenizers import ByteLevelBPETokenizer
```

Here's a detailed explanation of each parameter in the `__init__()` method:

## 1. **`vocab`** - Vocabulary mapping
- **Type**: `Optional[Union[str, Dict[str, int]]]`
- **Default**: `None`
- **Purpose**: Defines the token-to-ID mapping

**Examples**:
```python
# As a file path (string)
tokenizer = ByteLevelBPETokenizer(
    vocab="path/to/vocab.json",
    merges="path/to/merges.txt"
)

# As a dictionary
vocab_dict = {
    "hello": 0,
    "world": 1,
    "Ġthe": 2,  # Ġ represents a space in byte-level encoding
}
tokenizer = ByteLevelBPETokenizer(vocab=vocab_dict, merges=[("h", "e")])

# None - creates an untrained tokenizer
tokenizer = ByteLevelBPETokenizer()
```

## 2. **`merges`** - BPE merge rules
- **Type**: `Optional[Union[str, List[Tuple[str, str]]]]`
- **Default**: `None`
- **Purpose**: Defines the order in which byte pairs are merged

**Examples**:
```python
# As a file path
tokenizer = ByteLevelBPETokenizer(
    vocab="vocab.json",
    merges="merges.txt"
)

# As a list of tuples
merge_rules = [
    ("h", "e"),      # merge "h" and "e" into "he"
    ("l", "l"),      # merge "l" and "l" into "ll"
    ("he", "ll"),    # merge "he" and "ll" into "hell"
]
tokenizer = ByteLevelBPETokenizer(vocab=vocab_dict, merges=merge_rules)
```

## 3. **`add_prefix_space`** - Space handling
- **Type**: `bool`
- **Default**: `False`
- **Purpose**: Whether to add a space at the beginning of the first word

**Examples**:
```python
# Without prefix space (default)
tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
# "hello" → ["hello"]

# With prefix space (like GPT-2)
tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
# "hello" → [" hello"]
# Useful for treating first word same as subsequent words
```

## 4. **`lowercase`** - Case normalization
- **Type**: `bool`
- **Default**: `False`
- **Purpose**: Convert all text to lowercase before tokenization

**Examples**:
```python
# Case-sensitive (default)
tokenizer = ByteLevelBPETokenizer(lowercase=False)
# "Hello WORLD" → tokens preserve case

# Case-insensitive
tokenizer = ByteLevelBPETokenizer(lowercase=True)
# "Hello WORLD" → "hello world" → tokens
```

## 5. **`dropout`** - BPE dropout regularization
- **Type**: `Optional[float]`
- **Default**: `None`
- **Purpose**: Randomly skip merges during training for regularization (value between 0 and 1)

**Examples**:
```python
# No dropout (default)
tokenizer = ByteLevelBPETokenizer(dropout=None)

# With 10% dropout - randomly skip 10% of merges
tokenizer = ByteLevelBPETokenizer(dropout=0.1)
# Helps model generalize better by seeing different segmentations

# With 50% dropout
tokenizer = ByteLevelBPETokenizer(dropout=0.5)
```

## 6. **`unicode_normalizer`** - Unicode normalization
- **Type**: `Optional[str]`
- **Default**: `None`
- **Purpose**: Normalize unicode characters (e.g., "NFC", "NFD", "NFKC", "NFKD")

**Examples**:
```python
# No normalization (default)
tokenizer = ByteLevelBPETokenizer(unicode_normalizer=None)

# NFC normalization (canonical composition)
tokenizer = ByteLevelBPETokenizer(unicode_normalizer="NFC")
# "é" (U+00E9) stays as single character

# NFD normalization (canonical decomposition)
tokenizer = ByteLevelBPETokenizer(unicode_normalizer="NFD")
# "é" → "e" + combining accent (U+0065 + U+0301)

# NFKC (compatibility composition) - more aggressive
tokenizer = ByteLevelBPETokenizer(unicode_normalizer="NFKC")
# "ﬁ" (ligature) → "fi"
```

## 7. **`continuing_subword_prefix`** - Subword marker
- **Type**: `Optional[str]`
- **Default**: `None` (empty string)
- **Purpose**: Prefix added to continuation subwords (not first subword)

**Examples**:
```python
# No prefix (default for byte-level BPE)
tokenizer = ByteLevelBPETokenizer(continuing_subword_prefix=None)
# "playing" → ["play", "ing"]

# With "##" prefix (BERT-style)
tokenizer = ByteLevelBPETokenizer(continuing_subword_prefix="##")
# "playing" → ["play", "##ing"]

# With "@@" prefix
tokenizer = ByteLevelBPETokenizer(continuing_subword_prefix="@@")
# "playing" → ["play", "@@ing"]
```

## 8. **`end_of_word_suffix`** - Word boundary marker
- **Type**: `Optional[str]`
- **Default**: `None` (empty string)
- **Purpose**: Suffix added to the last subword of each word

**Examples**:
```python
# No suffix (default)
tokenizer = ByteLevelBPETokenizer(end_of_word_suffix=None)
# "playing" → ["play", "ing"]

# With "</w>" suffix (common in some BPE variants)
tokenizer = ByteLevelBPETokenizer(end_of_word_suffix="</w>")
# "playing" → ["play", "ing</w>"]

# With "_EOW" suffix
tokenizer = ByteLevelBPETokenizer(end_of_word_suffix="_EOW")
# "hello world" → ["hello_EOW", "world_EOW"]
```

## 9. **`trim_offsets`** - Offset trimming
- **Type**: `bool`
- **Default**: `False`
- **Purpose**: Whether to trim whitespace from token offsets in the output

**Examples**:
```python
# Don't trim offsets (default)
tokenizer = ByteLevelBPETokenizer(trim_offsets=False)
# " hello " → offsets include spaces [0, 7]

# Trim offsets
tokenizer = ByteLevelBPETokenizer(trim_offsets=True)
# " hello " → offsets exclude leading/trailing spaces [1, 6]
# Useful when you want character positions without whitespace
```

---

## Complete Example

```python
# GPT-2 style configuration
tokenizer = ByteLevelBPETokenizer(
    vocab="gpt2-vocab.json",
    merges="gpt2-merges.txt",
    add_prefix_space=True,      # Treat first word like others
    lowercase=False,             # Preserve case
    dropout=None,                # No dropout during inference
    unicode_normalizer=None,     # No normalization
    continuing_subword_prefix=None,  # No prefix
    end_of_word_suffix=None,     # No suffix
    trim_offsets=False           # Keep full offsets
)

# Custom configuration with normalization
tokenizer = ByteLevelBPETokenizer(
    add_prefix_space=False,
    lowercase=True,
    dropout=0.1,
    unicode_normalizer="NFKC",
    trim_offsets=True
)
# Then train it
tokenizer.train(files=["data.txt"], vocab_size=30000)
```

# Best Practices for ByteLevelBPETokenizer Arguments

## 1. **`vocab` and `merges`**

### ✅ When to use:
- **Loading pre-trained models** (e.g., GPT-2, RoBERTa)
- **Inference/production** - always provide both to use an existing tokenizer
- **Consistency required** - when you need reproducible tokenization

### ❌ When NOT to use:
- **Training from scratch** - leave both as `None` and call `train()` or `train_from_iterator()`
- **New domain/language** - better to train fresh vocabulary

```python
# ✅ GOOD: Loading existing tokenizer for inference
tokenizer = ByteLevelBPETokenizer(
    vocab="gpt2-vocab.json",
    merges="gpt2-merges.txt"
)

# ✅ GOOD: Training new tokenizer
tokenizer = ByteLevelBPETokenizer()  # No vocab/merges
tokenizer.train(files=["my_data.txt"], vocab_size=30000)

# ❌ BAD: Providing vocab but then training (training overwrites it)
tokenizer = ByteLevelBPETokenizer(vocab="old.json", merges="old.txt")
tokenizer.train(files=["new_data.txt"])  # Wastes time loading old vocab
```

---

## 2. **`add_prefix_space`**

### ✅ When to use (`True`):
- **GPT-2/GPT-3 style models** - ensures consistency between beginning and middle tokens
- **When position in sentence shouldn't matter** - "Hello" and " Hello" should tokenize similarly
- **Continuation tasks** - when joining generated text segments

### ❌ When NOT to use (`False`):
- **Position-aware models** - when first word should be treated differently
- **Memory-constrained** - adds slight overhead
- **Language-specific tokenization** - some languages don't use spaces

```python
# ✅ GOOD: GPT-2 style generation
tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
# "world" and " world" tokenize the same way
# Important for: "Hello world" + "world" = consistent

# ✅ GOOD: Sentence classification (position matters)
tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
# First word "Hello" vs middle word " Hello" can be different
```

---

## 3. **`lowercase`**

### ✅ When to use (`True`):
- **Case-insensitive tasks** - sentiment analysis, spam detection
- **Reduce vocabulary size** - "Hello"/"hello"/"HELLO" → one token
- **Noisy text** - social media, user-generated content with inconsistent casing
- **Smaller datasets** - reduces sparsity

### ❌ When NOT to use (`False`):
- **Named Entity Recognition (NER)** - "Apple" (company) vs "apple" (fruit)
- **Code generation** - case matters in programming
- **Proper nouns important** - "US" vs "us", "PM" vs "pm"
- **Large, clean datasets** - can learn case patterns

```python
# ✅ GOOD: Sentiment analysis on tweets
tokenizer = ByteLevelBPETokenizer(lowercase=True)
# "LOVE this!!!" and "love this" treated similarly

# ❌ BAD: Named entity extraction
tokenizer = ByteLevelBPETokenizer(lowercase=True)
# Loses distinction: "March" (month) vs "march" (walk)

# ✅ GOOD: Code or formal text
tokenizer = ByteLevelBPETokenizer(lowercase=False)
# Preserves: "String" (type) vs "string" (variable)
```

---

## 4. **`dropout`**

### ✅ When to use (`0.1` to `0.3`):
- **During training only** - improves model robustness
- **Overfitting prevention** - especially with smaller datasets
- **Research/experimentation** - exploring subword regularization
- **Multilingual models** - helps generalize across languages

### ❌ When NOT to use (`None`):
- **Inference/production** - introduces randomness, slower
- **Pre-trained model compatibility** - must match training configuration
- **Deterministic output required** - breaks reproducibility

```python
# ✅ GOOD: Training a new model with regularization
tokenizer = ByteLevelBPETokenizer(dropout=0.1)
tokenizer.train(files=["train.txt"], vocab_size=30000)
# Randomly skips 10% of merges → model sees varied segmentations

# ❌ BAD: Using dropout during inference
tokenizer = ByteLevelBPETokenizer(
    vocab="model-vocab.json",
    merges="model-merges.txt",
    dropout=0.1  # Will give different results each time!
)

# ✅ GOOD: Inference - no dropout
tokenizer = ByteLevelBPETokenizer(
    vocab="model-vocab.json",
    merges="model-merges.txt",
    dropout=None  # Deterministic
)
```

---

## 5. **`unicode_normalizer`**

### ✅ When to use:
- **`"NFC"`** (Canonical Composition) - **DEFAULT CHOICE** for most cases
  - Handles accented characters consistently: "é" stays as single char
  - Good for European languages, general text
  
- **`"NFKC"`** (Compatibility Composition) - **for noisy text**
  - Aggressive normalization: "ﬁ" → "fi", fullwidth "Ａ" → "A"
  - Good for: web scraping, OCR, social media
  
- **`"NFD"`** (Canonical Decomposition) - **rare, linguistic analysis**
  - Breaks accents: "é" → "e" + combining accent

### ❌ When NOT to use (`None`):
- **Already normalized data** - adds processing overhead
- **Emoji/special characters important** - normalization may alter them
- **Pre-trained model compatibility** - must match training config

```python
# ✅ GOOD: General multilingual text
tokenizer = ByteLevelBPETokenizer(unicode_normalizer="NFC")
# Handles: "café", "naïve", "résumé" consistently

# ✅ GOOD: Noisy web data
tokenizer = ByteLevelBPETokenizer(unicode_normalizer="NFKC")
# Normalizes: "𝐛𝐨𝐥𝐝 𝐭𝐞𝐱𝐭" → "bold text"
#            "①②③" → "123"

# ❌ BAD: Emoji-heavy social media (if preserving exact emojis)
tokenizer = ByteLevelBPETokenizer(unicode_normalizer="NFKC")
# Might alter: "❤️" or "🔥" unexpectedly

# ✅ GOOD: English-only, clean text
tokenizer = ByteLevelBPETokenizer(unicode_normalizer=None)
# Skip normalization overhead
```

---

## 6. **`continuing_subword_prefix`**

### ✅ When to use:
- **NEVER for byte-level BPE** - conflicts with byte-level encoding
- **Only if building non-byte-level tokenizer** - use regular BPE instead
- **Compatibility with other systems** - if downstream requires markers

### ❌ When NOT to use (`None`):
- **GPT-2/RoBERTa style** - byte-level encoding already handles word boundaries
- **Standard use cases** - keep default (None)

```python
# ✅ GOOD: Standard byte-level BPE (GPT-2/RoBERTa)
tokenizer = ByteLevelBPETokenizer(continuing_subword_prefix=None)
# Uses Ġ (byte-level space marker) instead

# ⚠️ RARE: Only if you have specific requirements
tokenizer = ByteLevelBPETokenizer(continuing_subword_prefix="##")
# Mixing styles - generally avoid this

# 💡 TIP: Use WordPiece or regular BPE for BERT-style "##" prefixes
# ByteLevelBPE isn't designed for this
```

---

## 7. **`end_of_word_suffix`**

### ✅ When to use:
- **Rare: explicit word boundary marking** - some linguistic tasks
- **Compatibility with legacy systems** - if required by downstream tools
- **Research purposes** - experimenting with boundary markers

### ❌ When NOT to use (`None`):
- **99% of use cases** - byte-level encoding already tracks boundaries
- **GPT-2/RoBERTa compatibility** - they don't use suffixes
- **Standard production** - unnecessary complexity

```python
# ✅ GOOD: Standard usage (default)
tokenizer = ByteLevelBPETokenizer(end_of_word_suffix=None)
# Byte-level encoding handles boundaries automatically

# ⚠️ RARE: Explicit boundary marking for special research
tokenizer = ByteLevelBPETokenizer(end_of_word_suffix="</w>")
# "hello" → "hello</w>", "world" → "world</w>"
# Only use if you have a specific research reason
```

---

## 8. **`trim_offsets`**

### ✅ When to use (`True`):
- **Token classification** - NER, POS tagging (want exact word positions)
- **Highlighting/annotations** - UI applications showing token boundaries
- **Question answering** - extracting answer spans from text
- **Character-level alignment** - when whitespace shouldn't count

### ❌ When NOT to use (`False`):
- **Text generation** - don't need character offsets
- **Classification tasks** - only care about tokens, not positions
- **Exact reconstruction needed** - want original spacing preserved

```python
# ✅ GOOD: Named Entity Recognition
tokenizer = ByteLevelBPETokenizer(trim_offsets=True)
encoding = tokenizer.encode("  John Smith  ")
# Offsets point to "John" and "Smith" without spaces
# Easier to highlight exact entities in UI

# ✅ GOOD: Question Answering (extracting answer spans)
tokenizer = ByteLevelBPETokenizer(trim_offsets=True)
# When extracting "Paris" from "  Paris  ", get clean boundaries

# ❌ BAD: Text generation
tokenizer = ByteLevelBPETokenizer(trim_offsets=True)
# Adds overhead for unused feature

# ✅ GOOD: Simple classification
tokenizer = ByteLevelBPETokenizer(trim_offsets=False)
# Faster, simpler
```

---

## 📋 Common Configuration Recipes

### **GPT-2 Compatible (Default)**
```python
tokenizer = ByteLevelBPETokenizer(
    vocab="gpt2-vocab.json",
    merges="gpt2-merges.txt",
    add_prefix_space=True,       # GPT-2 style
    lowercase=False,             # Case-sensitive
    dropout=None,                # No dropout for inference
    unicode_normalizer=None,     # No normalization
    continuing_subword_prefix=None,
    end_of_word_suffix=None,
    trim_offsets=False
)
```

### **Training a Robust New Model**
```python
tokenizer = ByteLevelBPETokenizer(
    add_prefix_space=True,
    lowercase=False,             # Keep case unless task-specific
    dropout=0.1,                 # Regularization during training
    unicode_normalizer="NFC",    # Normalize accents
    trim_offsets=False
)
tokenizer.train(files=["data.txt"], vocab_size=30000)
```

### **Case-Insensitive Sentiment Analysis**
```python
tokenizer = ByteLevelBPETokenizer(
    add_prefix_space=True,
    lowercase=True,              # Case doesn't matter
    dropout=None,                # Inference
    unicode_normalizer="NFC",
    trim_offsets=False
)
```

### **NER / Token Classification**
```python
tokenizer = ByteLevelBPETokenizer(
    vocab="ner-vocab.json",
    merges="ner-merges.txt",
    add_prefix_space=False,      # Position matters
    lowercase=False,             # "Apple" vs "apple"
    dropout=None,
    unicode_normalizer="NFC",
    trim_offsets=True            # Need accurate spans
)
```

### **Noisy Web/Social Media Data**
```python
tokenizer = ByteLevelBPETokenizer(
    lowercase=True,              # Normalize case
    dropout=0.2,                 # Higher regularization
    unicode_normalizer="NFKC",   # Aggressive normalization
    add_prefix_space=True,
    trim_offsets=False
)
```

---

## 🎯 Key Takeaways

1. **Start simple** - Most params can stay default (`None`/`False`)
2. **Match your task** - lowercase for sentiment, preserve case for NER
3. **Dropout only during training** - never in production
4. **Unicode normalization** - Use "NFC" unless you have noisy data (then "NFKC")
5. **Trim offsets** - Only when you need character-level alignment
6. **add_prefix_space=True** - For GPT-style models and consistency
7. **Avoid subword markers** - ByteLevelBPE handles boundaries automatically
