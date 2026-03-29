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
from transformers import GPT2Config, GPT2LMHeadModel, Trainer

from workshop_common import (
    build_workshop_paths,
    build_training_args,
    detect_device_and_optimizer,
    generate_and_print_samples,
    load_or_train_byte_level_bpe_tokenizer,
    load_text_dataset_with_validation,
    prepare_causal_lm_datasets,
    print_dataset_overview,
    print_tokenizer_preview,
    train_and_report,
)

# %% Load dataset
paths = build_workshop_paths(
    __file__,
    tokenizer_subdir="tokenizers/verdict_tokenizer",
    output_subdir="models/gpt2-small-verdict",
)

dataset = load_text_dataset_with_validation(paths.data_file)
print_dataset_overview(dataset)

# %% Device & optimizer detection
device_type, optim_type = detect_device_and_optimizer()
print(f"Device: {device_type}, Optimizer: {optim_type}")

# %% Tokenizer: Train or load

tokenizer = load_or_train_byte_level_bpe_tokenizer(
    data_file=paths.data_file,
    tokenizer_dir=paths.tokenizer_dir,
)

print_tokenizer_preview(
    tokenizer,
    sample_text="I love large language models",
    extra_decode_examples=[
        (
            "Decoded one training entry:",
            [
                16,
                310,
                966,
                286,
                283,
                263,
                981,
                289,
                580,
                793,
                17,
                81,
                957,
                300,
                305,
                593,
            ],
        )
    ],
)

# %% Tokenize & group text into blocks
block_size = 32
lm_datasets = prepare_causal_lm_datasets(dataset, tokenizer, block_size=block_size)


# %% Model setup (small GPT-2)
# GPT and Llama diagrams: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=block_size,  # config.n_positions >= block_size
    n_embd=16,
    n_layer=2,
    n_head=2,
)
setattr(config, "_attn_implementation", "eager")
model = GPT2LMHeadModel(config)
print("\nModel initialized.")
print(model)


# %% Training
training_args = build_training_args(
    paths.output_dir,
    optim_type,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,  # reduce for fast hands-on demo
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# Stepwise debugging:
# 1. Open .venv/lib/python3.11/site-packages/transformers/models/gpt2/modeling_gpt2.py
# 2. Set breakpoints on GPT2Model.forward() and GPT2Block.forward()

# Loss verification sketch from transformers/loss/loss_utils.py::ForCausalLMLoss
# import numpy as np
#
# t = shift_labels.cpu().numpy()
# s = logits.cpu().detach().numpy()
# tt = t[t != ignore_index]
# ss = s[t != ignore_index]
# ss_e = np.exp(ss)
# ss_e_sum = ss_e.sum(axis=1, keepdims=True)
# p = ss_e / ss_e_sum
# sum([-np.log(p[i][j]) for i, j in enumerate(tt)]) / num_items_in_batch

results, perplexity = train_and_report(trainer)

# %% Text generation
generate_and_print_samples(
    model,
    tokenizer,
    input_text="He laughed slightly",
    max_length=model.config.max_position_embeddings,
    num_return_sequences=3,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)

# %% Optional exercises for students
# 1. Modify `block_size` and `n_positions` and observe effects
# 2. Try different `top_k` or `top_p` for generation
# 3. Train on a small custom corpus (your own short stories)
# 4. Compare greedy vs. sampling vs. beam search generation
# 5. Plot train vs. validation loss curves for multiple epochs
