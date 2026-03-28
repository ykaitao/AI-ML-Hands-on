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
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from workshop_common import (
    build_workshop_paths,
    count_unique_words,
    detect_device_and_optimizer,
    load_or_train_sentencepiece_tokenizer,
    load_text_dataset_with_validation,
    tokenize_and_group_texts,
    verify_causal_lm_dataset,
)

# %% Load dataset
paths = build_workshop_paths(
    __file__,
    tokenizer_subdir="tokenizers/verdict_llama_tokenizer",
    output_subdir="models/llama3-small-verdict",
)

dataset = load_text_dataset_with_validation(paths.data_file)
print("Dataset loaded.")
print(dataset)

# %% Explore dataset (optional)
print("\nSample text:")
print(dataset["train"][0]["text"][:500])


print("Unique words in training set:", count_unique_words(dataset["train"]))

# %% Device & optimizer detection
device_type, optim_type = detect_device_and_optimizer()
print(f"Device: {device_type}, Optimizer: {optim_type}")

# %% Tokenizer: Train or load

tokenizer = load_or_train_sentencepiece_tokenizer(
    data_file=paths.data_file,
    tokenizer_dir=paths.tokenizer_dir,
    vocab_size=1187,
)

print(f"\nTokenizer ready. Vocab size: {len(tokenizer)}")
sample = tokenizer.encode("I love large language models")
print("Encoded:", sample)
print("Decoded:", tokenizer.decode(sample))

# %% Tokenize & group text into blocks
block_size = 32
lm_datasets = tokenize_and_group_texts(dataset, tokenizer, block_size=block_size)
verify_causal_lm_dataset(lm_datasets)
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
    output_dir=paths.output_dir,
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
    max_length=model.config.max_position_embeddings,
    num_return_sequences=3,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
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
