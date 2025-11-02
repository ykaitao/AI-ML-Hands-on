# GPT2TokenizerFast Quick Reference (Init & Usage)

Practical guidance for initialising `GPT2TokenizerFast` (see `transformers/models/gpt2/tokenization_gpt2_fast.py`). The fast tokenizer is backed by the 🤗 `tokenizers` library and expects byte-level BPE files.

---

## Core Arguments (`__init__`)

| Argument | Default | Typical use | Skip when |
| --- | --- | --- | --- |
| `tokenizer_file` | `None` | Load modern one-file JSON tokenizers | You only have legacy `vocab/merges` |
| `vocab_file`, `merges_file` | `None` | Load BPE assets explicitly | Using `tokenizer_file` or `from_pretrained` |
| `unk_token`, `bos_token`, `eos_token` | `"<\|endoftext\|>"` | Modify only for custom models | Leave default for GPT‑2 compatibility |
| `add_prefix_space` | `False` | Set `True` for pre-tokenised inputs (`is_split_into_words=True`) | Leave `False` for standard generation |
| `add_bos_token` (kwarg) | `False` | Force BOS insertion at encode/start | GPT-2 pretraining didn’t use BOS |

### Quick Examples

- **Pretrained GPT‑2 (recommended)**  
  ```python
  tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
  ```

- **Single-file tokenizer**  
  ```python
  tokenizer = GPT2TokenizerFast(
      tokenizer_file="tokenizer.json",
      unk_token="<|endoftext|>",
      bos_token="<|endoftext|>",
      eos_token="<|endoftext|>",
  )
  ```

- **Legacy vocab/merges pair**  
  ```python
  tokenizer = GPT2TokenizerFast(
      vocab_file="vocab.json",
      merges_file="merges.txt",
      unk_token="<|endoftext|>",
      bos_token="<|endoftext|>",
      eos_token="<|endoftext|>",
  )
  ```

> **Avoid** mixing `tokenizer_file` with `vocab_file`/`merges_file`—only one source can be used at a time.

---

## Special Tokens

- GPT‑2 uses the same byte-level marker `"<|endoftext|>"` for unknown, BOS, and EOS. Keep this default when fine‑tuning GPT‑2 checkpoints.
- To add new specials (e.g. padding, chat roles), use `tokenizer.add_special_tokens({...})` and resize model embeddings accordingly.

```python
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "additional_special_tokens": ["<|user|>", "<|assistant|>"],
})
model.resize_token_embeddings(len(tokenizer))
```

---

## `add_prefix_space`

| Setting | Behaviour | Use when |
| --- | --- | --- |
| `False` (default) | First word is tokenised differently if no leading space | Standard GPT‑2 generation (matches pretraining) |
| `True` | Leading space added implicitly; consistent treatment across words | Required for `is_split_into_words=True`, NER, POS tagging |

```python
t_default = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
t_split = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", add_prefix_space=True)

print(t_default("Hello world")["input_ids"])    # [15496, 995]
print(t_default(" Hello world")["input_ids"])   # [18435, 995]
print(t_split("Hello world")["input_ids"])      # [15496, 995]
print(t_split(" Hello world")["input_ids"])     # [15496, 995]
```

Attempting to use pre-tokenised inputs (`is_split_into_words=True`) without `add_prefix_space=True` will raise an assertion.

---

## `add_bos_token`

Hidden kwarg consumed by the base class. When `True`, a BOS token is prepended during encoding.

```python
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", add_bos_token=True)
tokenized = tokenizer("Hello world")
# Adds BOS (token id 50256) before content
```

GPT‑2 was trained without an explicit BOS; enabling this may affect performance unless your model expects it.

---

## File I/O Patterns

| Task | Methods |
| --- | --- |
| Save fast tokenizer | `tokenizer.save_pretrained(path)` → produces `tokenizer.json`, `tokenizer_config.json`, etc. |
| Load from local dir | `GPT2TokenizerFast.from_pretrained(path)` |
| Save legacy vocab/merges | `tokenizer.save_vocabulary(save_dir)` → returns tuple of saved files |

---

## Workflow Recipes

### 1. **Standard Inference / Generation**
```python
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token  # add pad if batching
enc = tokenizer(["Hello", "How are you?"], padding=True, return_tensors="pt")
```

### 2. **NER / Token Classification**
```python
tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2",
    add_prefix_space=True,
)
encoding = tokenizer(
    ["John", "Doe", "works", "at", "Acme"],
    is_split_into_words=True,
    return_tensors="pt",
)
```

### 3. **Custom ByteLevel BPE**
```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(["corpus.txt"], vocab_size=30000, special_tokens=["<|endoftext|>", "<|pad|>"])
tokenizer.save_model("tokenizer_assets")

hf_tokenizer = GPT2TokenizerFast(
    vocab_file="tokenizer_assets/vocab.json",
    merges_file="tokenizer_assets/merges.txt",
    unk_token="<|endoftext|>",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    add_prefix_space=False,
)
hf_tokenizer.save_pretrained("tokenizer_hf")
```

### 4. **Instruction / Chat Formatting**
```python
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>"],
    "pad_token": "<|pad|>",
})
model.resize_token_embeddings(len(tokenizer))

def format_chat(messages):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
```

---

## Practical Reminders

1. Use `from_pretrained()` for pretrained assets; it resolves special tokens and config automatically.  
2. Choose `tokenizer_file` (single JSON) over separate vocab/merges when available for faster loading.  
3. Only toggle `add_prefix_space` when ingesting pre-tokenised word lists—keeping it `False` preserves GPT‑2 behaviour.  
4. Add a padding token before batching (`tokenizer.pad_token = tokenizer.eos_token` or add a new token).  
5. Resizing model embeddings is mandatory after adding new special tokens.  
6. Leave `unk/bos/eos` at GPT‑2 defaults unless retraining or adapting the model.  
7. `add_bos_token` inserts BOS automatically—handy for classification, but off by default.  
8. `save_pretrained` writes `tokenizer.json`; legacy `save_vocabulary` exports separate files.  

Use this as a quick map back to `tokenization_gpt2_fast.py` when configuring GPT‑2 tokenizers for different workflows.
