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
)

from workshop_common import (
    build_workshop_paths,
    build_trainer,
    build_training_args,
    detect_device_and_optimizer,
    generate_and_print_samples,
    load_or_train_byte_level_bpe_tokenizer,
    load_text_dataset_with_validation,
    prepare_causal_lm_datasets,
    print_dataset_overview,
    print_runtime_info,
    print_tokenizer_preview,
    train_and_report,
)

# %% Load dataset
paths = build_workshop_paths(
    __file__,
    tokenizer_subdir="tokenizers/verdict_tokenizer",
    output_subdir="models/deepseek-r1-debug",
)

dataset = load_text_dataset_with_validation(paths.data_file)
print_dataset_overview(dataset)

# %% Device & optimizer detection
device_type, optim_type = detect_device_and_optimizer()
print_runtime_info(device_type, optim_type)

# %% Tokenizer: Train or load
tokenizer = load_or_train_byte_level_bpe_tokenizer(
    data_file=paths.data_file,
    tokenizer_dir=paths.tokenizer_dir,
)

print_tokenizer_preview(tokenizer, sample_text="I love large language models")

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
setattr(config, "_attn_implementation", "eager")
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
lm_datasets = prepare_causal_lm_datasets(dataset, tokenizer, block_size=block_size)

# %% Training
# Keep this short so it is practical for interactive debugging.
training_args = build_training_args(
    paths.output_dir,
    optim_type,
    eval_strategy="no",
    learning_rate=5e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=3,
    weight_decay=0.01,
    save_strategy="no",
    logging_steps=1,
)

trainer = build_trainer(model, training_args, lm_datasets)

# Stepwise debugging:
# 1. Open .venv/lib/python3.11/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py
# 2. Set breakpoints at DeepseekV3Model.forward(), DeepseekV3DecoderLayer.forward(),
#    DeepseekV3Attention.forward(), and DeepseekV3MoE.forward()
# 3. Start this script in the debugger and step into the forward pass

results, perplexity = train_and_report(trainer)

# %% Text generation
generate_and_print_samples(
    model,
    tokenizer,
    input_text="He laughed slightly",
    max_length=model.config.max_position_embeddings,
    num_return_sequences=3,
    do_sample=True,
    top_k=20,
    top_p=0.9,
)

# %% Optional exercises for students
# 1. Increase `n_routed_experts` and inspect router behavior
# 2. Change `first_k_dense_replace` to 0 and observe a fully sparse stack
# 3. Set `q_lora_rank=None` and `kv_lora_rank=None` to compare attention paths
# 4. Increase `block_size` and `max_position_embeddings` together
# 5. Compare this tiny DeepSeek-style model with gpt2.py layer by layer
