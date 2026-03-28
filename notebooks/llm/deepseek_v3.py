# %% [markdown]
# # Hands-on DeepSeek-R1 Mini Workshop
#
# In this notebook-style script, you'll learn step-by-step how to:
# 1. Load a text dataset
# 2. Train or load a tokenizer
# 3. Prepare data for a tiny DeepSeek-style causal LM
# 4. Train and evaluate the model
# 5. Generate text
#
# This is a debugging-oriented teaching script. It does not load the public
# DeepSeek-R1 checkpoint. Instead, it builds a very small randomly initialized
# DeepSeek-V3-style model so you can step through the same core architecture:
# attention, RMSNorm, MoE routing, and causal LM loss.

# %% Install libraries
# !pip install -q transformers datasets accelerate --upgrade

# %% Imports
import torch
from transformers import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    Trainer,
    TrainingArguments,
)

from workshop_common import (
    build_workshop_paths,
    count_unique_words,
    detect_device_and_optimizer,
    load_or_train_byte_level_bpe_tokenizer,
    load_text_dataset_with_validation,
    tokenize_and_group_texts,
    verify_causal_lm_dataset,
)

# %% Load dataset
paths = build_workshop_paths(
    __file__,
    tokenizer_subdir="tokenizers/verdict_tokenizer",
    output_subdir="models/deepseek-r1-debug",
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
tokenizer = load_or_train_byte_level_bpe_tokenizer(
    data_file=paths.data_file,
    tokenizer_dir=paths.tokenizer_dir,
)

print(f"\nTokenizer ready. Vocab size: {len(tokenizer)}")
sample = tokenizer.encode("I love large language models")
print("Encoded:", sample)
print("Decoded:", tokenizer.decode(sample))

# %% Model setup (tiny DeepSeek-style model)
# Keep one dense layer and one MoE layer so the debugger can step through both.
config = DeepseekV3Config(
    vocab_size=len(tokenizer),
    max_position_embeddings=32,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    num_experts_per_tok=2,
    n_group=2,
    topk_group=1,
    routed_scaling_factor=2.0,
    first_k_dense_replace=1,
    q_lora_rank=32,
    kv_lora_rank=16,
    qk_rope_head_dim=8,
    qk_nope_head_dim=8,
    v_head_dim=16,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = DeepseekV3ForCausalLM(config)
print("\nModel initialized.")
print(model)

# Quick sanity check before Trainer moves the model to an accelerator.
debug_batch = tokenizer(
    "He laughed slightly at the verdict.",
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=16,
)
with torch.no_grad():
    debug_logits = model(**debug_batch).logits
print("Forward-pass logits shape:", tuple(debug_logits.shape))

# %% Tokenize & group text into blocks
block_size = 16
lm_datasets = tokenize_and_group_texts(dataset, tokenizer, block_size=block_size)
verify_causal_lm_dataset(lm_datasets)
print("\nDataset ready for training!")

# %% Training
# Keep this short so it is practical for interactive debugging.
training_args = TrainingArguments(
    output_dir=paths.output_dir,
    eval_strategy="no",
    learning_rate=5e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=3,
    weight_decay=0.01,
    optim=optim_type,
    report_to="none",
    dataloader_num_workers=0,
    fp16=False,
    save_strategy="no",
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

print("\nStarting training...")
# Stepwise debugging:
# 1. Open .venv/lib/python3.11/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py
# 2. Set breakpoints at DeepseekV3Model.forward(), DeepseekV3DecoderLayer.forward(),
#    DeepseekV3Attention.forward(), and DeepseekV3MoE.forward()
# 3. Start this script in the debugger and step into the forward pass

# Loss verification sketch:
# nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
# import numpy as np
#
# t = target.cpu().numpy()
# s = source.cpu().detach().numpy()
# tt = t[t != ignore_index]
# ss = s[t != ignore_index]
# ss_e = np.exp(ss)
# ss_e_sum = ss_e.sum(axis=1, keepdims=True)
# p = ss_e / ss_e_sum
# sum([-np.log(p[i][j]) for i, j in enumerate(tt)])

trainer.train()

results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(results["eval_loss"]))
print(f"\nEvaluation Perplexity: {perplexity.item():.2f}")

# %% Text generation
input_text = "He laughed slightly"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_length=32,
    num_return_sequences=3,
    do_sample=True,
    top_k=20,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

print("\nGenerated Texts:")
for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))

# %% Optional exercises for students
# 1. Increase `n_routed_experts` and inspect router behavior
# 2. Change `first_k_dense_replace` to 0 and observe a fully sparse stack
# 3. Set `q_lora_rank=None` and `kv_lora_rank=None` to compare attention paths
# 4. Increase `block_size` and `max_position_embeddings` together
# 5. Compare this tiny DeepSeek-style model with gpt2.py layer by layer
