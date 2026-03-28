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
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

# %% Load dataset
script_dir = Path(__file__).parent
data_file = (script_dir / "data_examples/the-verdict.txt").__str__()
tokenizer_dir = (script_dir / "tokenizers/verdict_tokenizer").__str__()
output_dir = (script_dir / "models/deepseek-r1-debug").__str__()

dataset = load_dataset("text", data_files=data_file)

if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset["validation"] = dataset.pop("test")

print("Dataset loaded.")
print(dataset)

# %% Explore dataset (optional)
print("\nSample text:")
print(dataset["train"][0]["text"][:500])


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
if not os.path.exists(tokenizer_dir):
    print("Training new tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=data_file,
        vocab_size=1500,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_model(tokenizer_dir)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
else:
    print("Loading existing tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)

tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"
tokenizer.bos_token = "<s>"
tokenizer.unk_token = "<unk>"

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
tokenized = dataset.map(
    lambda x: tokenizer(x["text"]),
    batched=True,
    remove_columns=["text"],
)
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

block_size = 16


def group_texts(examples):
    concat = {k: sum(examples[k], []) for k in examples}
    total_len = (len(concat["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
        for k, t in concat.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized.map(group_texts, batched=True, batch_size=32)

for split in lm_datasets:
    for col in ["input_ids", "attention_mask", "labels"]:
        assert col in lm_datasets[split].features
print("\nDataset ready for training!")

# %% Training
# Keep this short so it is practical for interactive debugging.
training_args = TrainingArguments(
    output_dir=output_dir,
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
"""Stepwise Debugging:
1. Open the file: .venv/lib/python3.11/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py
2. Set breakpoints at:
   - DeepseekV3Model.forward()
   - DeepseekV3DecoderLayer.forward()
   - DeepseekV3Attention.forward()
   - DeepseekV3MoE.forward()
3. Start this script in the debugger and step into the forward pass.
"""

"""Loss verification sketch:
`nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)`

import numpy as np

t = target.cpu().numpy()
s = source.cpu().detach().numpy()
tt = t[t != ignore_index]
ss = s[t != ignore_index]
ss_e = np.exp(ss)
ss_e_sum = ss_e.sum(axis=1, keepdims=True)
p = ss_e / ss_e_sum
sum([-np.log(p[i][j]) for i, j in enumerate(tt)])
"""

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