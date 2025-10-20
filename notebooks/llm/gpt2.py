# %% [markdown]
# # Hands-on GPT-2 Mini Workshop
#
# In this notebook, you'll learn step-by-step how to:
# 1. Load a text dataset
# 2. Train or load a tokenizer
# 3. Prepare data for a small GPT-2 model
# 4. Train and evaluate the model
# 5. Generate text
#
# This is designed for **educational purposes** — all sizes are small for fast experimentation.

# %% Install libraries
# !pip install -q transformers datasets accelerate --upgrade

# %% Imports
import os
import torch
from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

# %% Load dataset
dataset = load_dataset("text", data_files="./data_examples/the-verdict.txt")

# Split train/validation if validation not already in dataset
if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset["validation"] = dataset.pop("test")
print("Dataset loaded.")
print(dataset)

# %% Explore dataset (optional)
print("\nSample text:")
print(dataset["train"][0]["text"][:500])


# Count unique words
def count_unique_words(split):
    return len(set(word for ex in split for word in ex["text"].split()))


print("Unique words in training set:", count_unique_words(dataset["train"]))

# %% Device & optimizer detection
device_type, optim_type = (
    ("tpu", "adamw_torch")
    if "COLAB_TPU_ADDR" in os.environ
    else (
        ("mps", "adamw_torch")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else (
            ("gpu", "adamw_torch_fused")
            if torch.cuda.is_available()
            else ("cpu", "adamw_torch")
        )
    )
)
print(f"Device: {device_type}, Optimizer: {optim_type}")

# %% Tokenizer: Train or load
tokenizer_dir = "./tokenizers/verdict_tokenizer"

if not os.path.exists(tokenizer_dir):
    print("Training new tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files="./data_examples/the-verdict.txt",
        vocab_size=1500,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_model(tokenizer_dir)
    # Convert to Hugging Face tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
else:
    print("Loading existing tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)

# Set special tokens
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"
tokenizer.bos_token = "<s>"
tokenizer.unk_token = "<unk>"

print(f"\nTokenizer ready. Vocab size: {len(tokenizer)}")
sample = tokenizer.encode("I love large language models")
print("Encoded:", sample)
print("Decoded:", tokenizer.decode(sample))

# %% Model setup (small GPT-2)
# Note: small sizes are for educational purposes only
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=16,  # must be >= block_size
    n_embd=16,
    n_layer=2,
    n_head=2,
)
model = GPT2LMHeadModel(config)
print("\nModel initialized.")
print(model)


# %% Tokenize & group text into blocks
def tokenize_fn(examples):
    return tokenizer(examples["text"])


tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

# Chunk sequences into fixed-length blocks
block_size = 16


def group_texts(examples):
    concat = {k: sum(examples[k], []) for k in examples}
    total_len = (len(concat["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
        for k, t in concat.items()
    }
    # For causal LM, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized.map(group_texts, batched=True, batch_size=32)

# Verify dataset structure
for split in lm_datasets:
    for col in ["input_ids", "attention_mask", "labels"]:
        assert col in lm_datasets[split].features
print("\nDataset ready for training!")

# Optional: plot sequence length histogram
seq_lengths = [len(x) for x in lm_datasets["train"]["input_ids"]]
plt.hist(seq_lengths, bins=20)
plt.title("Token Sequence Length Distribution")
plt.show()

# %% Training
training_args = TrainingArguments(
    output_dir="./models/gpt2-small-verdict",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=50,  # reduce for fast hands-on demo
    weight_decay=0.01,
    optim=optim_type,
    report_to="none",
    dataloader_num_workers=0,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

print("\nStarting training...")
trainer.train()

results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(results["eval_loss"]))
print(f"\nEvaluation Perplexity: {perplexity.item():.2f}")

# %% Text generation
input_text = "He laughed slightly"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_length=40,
    num_return_sequences=3,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)

print("\nGenerated Texts:")
for o in outputs:
    print(tokenizer.decode(o, skip_special_tokens=True))

# %% Optional exercises for students
# 1. Modify `block_size` and `n_positions` and observe effects
# 2. Try different `top_k` or `top_p` for generation
# 3. Train on a small custom corpus (your own short stories)
# 4. Compare greedy vs. sampling vs. beam search generation
# 5. Plot train vs. validation loss curves for multiple epochs
