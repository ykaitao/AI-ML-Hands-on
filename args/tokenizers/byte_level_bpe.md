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

# Best Practices for `train()` and `train_from_iterator()` Arguments

Based on the code, here's a comprehensive guide for the training function arguments:

---

## 1. **`files`** (for `train()`) / **`iterator`** (for `train_from_iterator()`)

### ✅ When to use `train()` with files:
- **Large datasets on disk** - efficient memory usage, reads line by line
- **Multiple files** - can pass list of file paths
- **Static datasets** - data already saved as text files
- **Simple workflows** - straightforward file-based training
- **Reproducibility** - same files give same results

### ✅ When to use `train_from_iterator()`:
- **Streaming data** - data doesn't fit in memory
- **Dynamic preprocessing** - apply transformations on-the-fly
- **Database/API sources** - data not in files
- **Memory constraints** - process data in chunks
- **Real-time data** - training on live data streams

```python
# ✅ GOOD: Training from files (simple, most common)
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["corpus1.txt", "corpus2.txt", "corpus3.txt"],
    vocab_size=30000
)

# ✅ GOOD: Training from a single file
tokenizer.train(files="large_corpus.txt", vocab_size=30000)

# ✅ GOOD: Training from iterator (memory-efficient)
def data_generator():
    with open("huge_file.txt") as f:
        for line in f:
            yield line.strip().lower()  # Preprocess on-the-fly

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    iterator=data_generator(),
    vocab_size=30000,
    length=1000000  # Important: provide approximate length
)

# ✅ GOOD: Streaming from multiple sources
def multi_source_generator():
    # From file
    with open("file1.txt") as f:
        for line in f:
            yield line
    # From API/database
    for item in database.query():
        yield item.text
    # From processed data
    for doc in preprocessed_docs:
        yield doc

tokenizer.train_from_iterator(iterator=multi_source_generator())

# ❌ BAD: Loading entire huge file into memory
with open("huge_file.txt") as f:
    all_lines = f.readlines()  # Memory issue!
tokenizer.train_from_iterator(iter(all_lines))

# ✅ BETTER: Just use train() with files
tokenizer.train(files="huge_file.txt")
```

---

## 2. **`vocab_size`**

### ✅ When to use different sizes:

**Small (1,000 - 10,000):**
- Very limited domains (medical codes, product IDs)
- Resource-constrained deployment
- Simple/repetitive text

**Medium (10,000 - 32,000):**
- **Most common choice** (GPT-2 uses 50,257)
- Balanced between coverage and efficiency
- General-purpose applications
- Single language models

**Large (50,000 - 100,000+):**
- Multilingual models (cover many languages)
- Rich, diverse corpora
- Code + natural language
- Large model capacity available

### 💡 Guidelines:
```python
# ✅ GOOD: Small domain-specific corpus
tokenizer.train(
    files="medical_codes.txt",
    vocab_size=5000  # Limited, specialized vocabulary
)

# ✅ GOOD: Standard English text (most common)
tokenizer.train(
    files="english_corpus.txt",
    vocab_size=30000  # Good default
)

# ✅ GOOD: Large multilingual corpus
tokenizer.train(
    files=["en.txt", "fr.txt", "de.txt", "zh.txt"],
    vocab_size=64000  # Need more tokens for multiple languages
)

# ✅ GOOD: Code + documentation
tokenizer.train(
    files="github_repos.txt",
    vocab_size=50000  # Code has many unique tokens
)

# ❌ BAD: Tiny vocab for diverse data
tokenizer.train(
    files="wikipedia_dump.txt",
    vocab_size=1000  # Way too small! Most words will be UNK
)

# ❌ BAD: Huge vocab for small data
tokenizer.train(
    files="100_emails.txt",
    vocab_size=100000  # Overfitting, many single-character tokens
)

# 💡 RULE OF THUMB: 
# vocab_size should be ~1-5% of unique words in corpus
# For 1M unique words → vocab_size of 10k-50k is reasonable
```

---

## 3. **`min_frequency`**

### ✅ When to use different thresholds:

**Low (1-2):** Default, most inclusive
**Medium (3-5):** Filter rare typos/errors
**High (10+):** Very clean, common tokens only

### 💡 Best Practices:
```python
# ✅ GOOD: Large, clean corpus (default)
tokenizer.train(
    files="wikipedia.txt",
    vocab_size=30000,
    min_frequency=2  # Standard, filters hapax legomena
)

# ✅ GOOD: Noisy social media data
tokenizer.train(
    files="tweets.txt",
    vocab_size=30000,
    min_frequency=5  # Filter typos, rare username fragments
)

# ✅ GOOD: Small corpus (keep more tokens)
tokenizer.train(
    files="small_dataset.txt",
    vocab_size=10000,
    min_frequency=1  # Don't over-filter when data is limited
)

# ✅ GOOD: Very large corpus with noise
tokenizer.train(
    files="web_crawl.txt",
    vocab_size=50000,
    min_frequency=10  # Aggressive filtering for quality
)

# ❌ BAD: High threshold on small data
tokenizer.train(
    files="1000_sentences.txt",
    vocab_size=5000,
    min_frequency=50  # Too aggressive! Will have almost no vocab
)

# 💡 CONSIDERATIONS:
# - Higher min_frequency → cleaner vocab, better quality
# - Lower min_frequency → more coverage, handles rare words
# - Balance with corpus size: small data needs lower threshold
```

**Interactive calculation example:**
```python
# Estimate min_frequency based on corpus
import re
from collections import Counter

def suggest_min_frequency(file_path):
    words = []
    with open(file_path) as f:
        for line in f:
            words.extend(re.findall(r'\w+', line.lower()))
    
    freq = Counter(words)
    total_tokens = len(words)
    unique_tokens = len(freq)
    
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")
    
    # Suggest min_frequency
    if total_tokens < 100_000:
        return 1  # Small corpus
    elif total_tokens < 1_000_000:
        return 2  # Medium corpus
    elif total_tokens < 10_000_000:
        return 3  # Large corpus
    else:
        return 5  # Very large corpus

min_freq = suggest_min_frequency("my_corpus.txt")
tokenizer.train(files="my_corpus.txt", min_frequency=min_freq)
```

---

## 4. **`show_progress`**

### ✅ When to use (`True`):
- **Interactive training** - Jupyter notebooks, development
- **Long training runs** - want to monitor progress
- **Debugging** - see where training might stall
- **User-facing applications** - show progress to users

### ❌ When to use (`False`):
- **Production pipelines** - logs get cluttered
- **Automated scripts** - running in background
- **Testing/CI** - cleaner output
- **Batch processing** - training many tokenizers

```python
# ✅ GOOD: Interactive notebook
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files="large_corpus.txt",
    vocab_size=30000,
    show_progress=True  # See progress bar
)

# ✅ GOOD: Production pipeline
def train_tokenizer_batch(files_list):
    for i, files in enumerate(files_list):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=files,
            vocab_size=30000,
            show_progress=False  # Clean logs
        )
        tokenizer.save(f"tokenizer_{i}")
        print(f"Completed tokenizer {i}")

# ✅ GOOD: Custom progress tracking
def train_with_custom_logging(files):
    print(f"Starting training on {files}")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=30000,
        show_progress=False  # Use custom logging instead
    )
    print("Training complete!")
```

---

## 5. **`special_tokens`**

### ✅ Essential special tokens to include:

**Core tokens (always include):**
- `<pad>` - Padding token
- `<unk>` / `<|endoftext|>` - Unknown/end token
- `<s>` / `</s>` - Start/end of sequence (for seq2seq)

**Task-specific:**
- `<mask>` - Masked language modeling (BERT-style)
- `<cls>`, `<sep>` - Classification tasks
- `<|startoftext|>`, `<|endoftext|>` - GPT-style
- Custom tokens - `<user>`, `<bot>`, `<code>`, etc.

### 💡 Best Practices:

```python
from tokenizers import AddedToken

# ✅ GOOD: Minimal special tokens (GPT-2 style)
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files="corpus.txt",
    vocab_size=30000,
    special_tokens=["<|endoftext|>"]  # GPT-2 uses one special token
)

# ✅ GOOD: Standard NLP tasks
tokenizer.train(
    files="corpus.txt",
    vocab_size=30000,
    special_tokens=[
        "<pad>",    # Padding (always ID 0)
        "<unk>",    # Unknown tokens
        "<s>",      # Start of sequence
        "</s>"      # End of sequence
    ]
)

# ✅ GOOD: BERT-style masked language modeling
tokenizer.train(
    files="corpus.txt",
    vocab_size=30000,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<cls>",    # Classification token
        "<sep>",    # Separator
        "<mask>"    # Masking token
    ]
)

# ✅ GOOD: Custom application (chatbot)
tokenizer.train(
    files="conversations.txt",
    vocab_size=30000,
    special_tokens=[
        "<pad>",
        "<|endoftext|>",
        "<user>",       # User message marker
        "<assistant>",  # Assistant response marker
        "<system>"      # System message marker
    ]
)

# ✅ GOOD: Using AddedToken for fine control
special_tokens = [
    "<pad>",
    AddedToken("<user>", single_word=True, normalized=False),
    AddedToken("<assistant>", single_word=True, normalized=False),
]
tokenizer.train(
    files="corpus.txt",
    vocab_size=30000,
    special_tokens=special_tokens
)

# ❌ BAD: Too many special tokens
tokenizer.train(
    files="corpus.txt",
    vocab_size=1000,
    special_tokens=[f"<special_{i}>" for i in range(500)]  # 50% of vocab!
)

# ❌ BAD: Forgetting essential tokens
tokenizer.train(
    files="corpus.txt",
    vocab_size=30000,
    special_tokens=[]  # No padding token! Will cause issues
)

# 💡 ORDER MATTERS: Special tokens get lowest IDs
# special_tokens[0] gets ID 0, special_tokens[1] gets ID 1, etc.
# Put <pad> first so it gets ID 0 (expected by many frameworks)
```

### 📋 Common Special Token Patterns:

```python
# Pattern 1: GPT-2/GPT-3 (minimalist)
GPT2_SPECIAL = ["<|endoftext|>"]

# Pattern 2: BERT (classification & masking)
BERT_SPECIAL = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]

# Pattern 3: T5 (seq2seq)
T5_SPECIAL = ["<pad>", "<eos>", "<unk>"] + [f"<extra_id_{i}>" for i in range(100)]

# Pattern 4: RoBERTa (similar to GPT-2 with byte-level)
ROBERTA_SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

# Pattern 5: Custom chatbot
CHATBOT_SPECIAL = [
    "<pad>",
    "<|endoftext|>",
    "<|im_start|>",  # Instruction/message start
    "<|im_end|>",    # Instruction/message end
]
```

---

## 6. **`length`** (for `train_from_iterator()` only)

### ✅ When to provide:
- **Progress bar accuracy** - shows meaningful ETA
- **Memory planning** - tokenizer can optimize
- **Known dataset size** - you counted/estimated rows

### ❌ When to omit (`None`):
- **Unknown size** - streaming from API, database
- **Infinite generators** - continuous data streams
- **Quick scripts** - don't want to count

```python
# ✅ GOOD: Known length (best performance)
def get_data():
    with open("corpus.txt") as f:
        for line in f:
            yield line

# Count lines first
with open("corpus.txt") as f:
    num_lines = sum(1 for _ in f)

tokenizer.train_from_iterator(
    iterator=get_data(),
    vocab_size=30000,
    length=num_lines  # Accurate progress bar
)

# ✅ GOOD: Estimated length (close enough)
def estimate_lines(file_path, sample_size=1000):
    """Estimate total lines by sampling"""
    import os
    file_size = os.path.getsize(file_path)
    
    with open(file_path) as f:
        sample_bytes = 0
        sample_lines = 0
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            sample_bytes += len(line.encode())
            sample_lines += 1
    
    avg_bytes_per_line = sample_bytes / sample_lines
    estimated_lines = int(file_size / avg_bytes_per_line)
    return estimated_lines

length = estimate_lines("huge_corpus.txt")
tokenizer.train_from_iterator(
    iterator=get_data(),
    vocab_size=30000,
    length=length  # Good approximation
)

# ✅ GOOD: Database with count
def db_iterator():
    for record in database.query("SELECT text FROM corpus"):
        yield record.text

total_records = database.query("SELECT COUNT(*) FROM corpus")[0]

tokenizer.train_from_iterator(
    iterator=db_iterator(),
    vocab_size=30000,
    length=total_records
)

# ✅ ACCEPTABLE: Unknown length (no progress info)
def streaming_api():
    while True:
        data = api.get_next_batch()
        if not data:
            break
        for item in data:
            yield item

tokenizer.train_from_iterator(
    iterator=streaming_api(),
    vocab_size=30000,
    length=None  # Can't know size in advance
)

# ❌ BAD: Wrong length (misleading progress bar)
tokenizer.train_from_iterator(
    iterator=get_data(),
    vocab_size=30000,
    length=100  # Actual length is 1M - progress bar will be wrong
)
```

---

## 📋 Complete Training Recipes

### **Recipe 1: Standard Training (Most Common)**
```python
tokenizer = ByteLevelBPETokenizer(
    add_prefix_space=True,
    lowercase=False,
    dropout=0.1,  # Regularization during training
    unicode_normalizer="NFC"
)

tokenizer.train(
    files=["train_data.txt"],
    vocab_size=30000,      # Good default
    min_frequency=2,       # Filter rare tokens
    show_progress=True,    # See progress
    special_tokens=["<pad>", "<|endoftext|>"]
)

# For inference, remove dropout
tokenizer_inference = ByteLevelBPETokenizer(
    vocab="vocab.json",
    merges="merges.txt",
    add_prefix_space=True,
    dropout=None  # No dropout for inference
)
```

### **Recipe 2: Large-Scale Training (Memory-Efficient)**
```python
def data_stream():
    for file_path in glob.glob("data/*.txt"):
        with open(file_path) as f:
            for line in f:
                yield line.strip()

# Estimate total lines
import os
total_lines = 0
for file_path in glob.glob("data/*.txt"):
    with open(file_path) as f:
        total_lines += sum(1 for _ in f)

tokenizer = ByteLevelBPETokenizer(
    add_prefix_space=True,
    unicode_normalizer="NFC"
)

tokenizer.train_from_iterator(
    iterator=data_stream(),
    vocab_size=50000,      # Large vocab for diverse data
    min_frequency=3,       # Higher threshold for cleaner vocab
    show_progress=True,
    special_tokens=["<pad>", "<unk>", "<|endoftext|>"],
    length=total_lines
)
```

### **Recipe 3: Multilingual Training**
```python
tokenizer = ByteLevelBPETokenizer(
    add_prefix_space=True,
    lowercase=False,  # Preserve case across languages
    unicode_normalizer="NFC"  # Important for accented characters
)

tokenizer.train(
    files=[
        "corpus_en.txt",
        "corpus_fr.txt",
        "corpus_de.txt",
        "corpus_es.txt",
    ],
    vocab_size=64000,  # Larger vocab for multiple languages
    min_frequency=5,   # Higher threshold (more data)
    show_progress=True,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
        "<lang_en>",  # Language markers
        "<lang_fr>",
        "<lang_de>",
        "<lang_es>",
    ]
)
```

### **Recipe 4: Domain-Specific (Medical/Legal)**
```python
tokenizer = ByteLevelBPETokenizer(
    add_prefix_space=False,  # Preserve exact formatting
    lowercase=False,         # Medical codes are case-sensitive
    unicode_normalizer="NFC"
)

tokenizer.train(
    files="medical_records.txt",
    vocab_size=20000,      # Smaller, specialized vocab
    min_frequency=1,       # Keep rare medical terms
    show_progress=True,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<diagnosis>",     # Domain-specific markers
        "<treatment>",
        "<medication>",
    ]
)
```

### **Recipe 5: Noisy Social Media**
```python
tokenizer = ByteLevelBPETokenizer(
    add_prefix_space=True,
    lowercase=True,            # Normalize case
    unicode_normalizer="NFKC",  # Aggressive normalization
    dropout=0.2                # Higher regularization
)

tokenizer.train(
    files="tweets.txt",
    vocab_size=30000,
    min_frequency=10,  # Filter typos and rare usernames
    show_progress=True,
    special_tokens=[
        "<pad>",
        "<|endoftext|>",
        "<url>",       # Replace URLs
        "<mention>",   # Replace @mentions
        "<hashtag>",   # Replace #hashtags
    ]
)
```

---

## 🎯 Key Takeaways

1. **`vocab_size`**: 30k is standard; scale up for multilingual/diverse data
2. **`min_frequency`**: 2 is default; increase (5-10) for noisy data
3. **`special_tokens`**: Always include `<pad>` first (gets ID 0)
4. **`show_progress`**: True for interactive, False for production
5. **`files` vs `iterator`**: Use files when possible (simpler); iterator for streaming
6. **`length`**: Provide when known for better progress tracking
7. **Training vs Inference**: Use dropout during training, set to None for inference