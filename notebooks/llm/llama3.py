# %% [markdown]
# # Hands-on LLaMA (Llama-3) Mini Workshop
#
# In this notebook, you'll learn step-by-step how to:
# 1. Load a text dataset
# 2. Train or load a tokenizer
# 3. Prepare data for a small LLaMA model
# 4. Train and evaluate the model
# 5. Generate text
#
# This is designed for **educational purposes** — all sizes are small for fast experimentation.

# %% Install libraries
# !pip install -q transformers datasets accelerate sentencepiece --upgrade

# %% Imports
import os
from pathlib import Path

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)
import sentencepiece as spm
from datasets import load_dataset
import matplotlib.pyplot as plt

# %% Load dataset
# Get the directory where this script is located
script_dir = Path(__file__).parent
data_file = (script_dir / "data_examples/the-verdict.txt").__str__()
tokenizer_dir = (script_dir / "tokenizers/verdict_llama_tokenizer").__str__()
output_dir = (script_dir / "models/llama3-small-verdict").__str__()

dataset = load_dataset("text", data_files=data_file)

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

spm_model_path = os.path.join(tokenizer_dir, "llama.model")

if not os.path.exists(spm_model_path):
    print("Training new SentencePiece tokenizer...")
    os.makedirs(tokenizer_dir, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=data_file,
        model_prefix=os.path.join(tokenizer_dir, "llama"),
        vocab_size=1187,
        character_coverage=1.0,
        model_type="unigram",
        user_defined_symbols=["<s>", "</s>", "<pad>", "<mask>"]
    )

# Load LLaMA tokenizer
print("Loading tokenizer...")
tokenizer = LlamaTokenizer(vocab_file=spm_model_path)

# Ensure special tokens are set
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"
tokenizer.bos_token = "<s>"
tokenizer.unk_token = "<unk>"

print(f"\nTokenizer ready. Vocab size: {len(tokenizer)}")
sample = tokenizer.encode("I love large language models")
print("Encoded:", sample)
print("Decoded:", tokenizer.decode(sample))

# %% Tokenize & group text into blocks
tokenized = dataset.map(
    lambda x: tokenizer(x["text"]),
    batched=True,
    remove_columns=["text"],
)
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

# Chunk sequences into fixed-length blocks
block_size = 32


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


# %% Model setup (small LLaMA)
# GPT and Llama diagrams: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama

config = LlamaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=block_size,  # config.max_position_embeddings >= block_size
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
)
model = LlamaForCausalLM(config)
print("\nModel initialized.")
print(model)


# %% Training
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,  # reduce for fast hands-on demo
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
# 1. Modify `block_size` and `max_position_embeddings` and observe effects
# 2. Try different `top_k` or `top_p` for generation
# 3. Train on a small custom corpus (your own short stories)
# 4. Compare greedy vs. sampling vs. beam search generation
# 5. Plot train vs. validation loss curves for multiple epochs
