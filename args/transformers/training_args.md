# TrainingArguments Quick Reference (transformers/training_args.py)

Source reference: `.venv/lib/python3.11/site-packages/transformers/training_args.py`. The `TrainingArguments` dataclass is the primary configuration object consumed by `Trainer`. Below is a field-by-field overview grouped by theme, with code snippets showing how the arguments are used in `Trainer`’s implementation and typical user workflows.

---

## Core Training Hyperparameters

| Argument | Default | Trainer behaviour | Why tweak it |
| --- | --- | --- | --- |
| `output_dir` | `""` | Directory is created and checkpoints/logs are written here. | Required for saving progress, resuming, or pushing to hub. |
| `overwrite_output_dir` | `False` | Trainer refuses to overwrite an existing path unless set to `True`. | Re-run experiments in the same folder without manual cleanup. |
| `do_train` / `do_eval` / `do_predict` | `False` | Control which phases the Trainer executes. | Useful in scripts that combine training, eval, and inference knobs. |
| `per_device_train_batch_size` | `8` | Determines per-device batch; total batch = per-device × world size × grad accum. | Fit training batches to GPU memory or increase throughput. |
| `per_device_eval_batch_size` | `8` | Sets evaluation/prediction batch per device. | Raise for faster eval on large GPUs; lower to avoid OOM. |
| `gradient_accumulation_steps` | `1` | Accumulates gradients over multiple forward passes before stepping optimizer. | Simulate larger batches when memory limited. |
| `num_train_epochs` | `3.0` | Stops training after N passes over dataset when `max_steps` not set. | Increase for more training cycles; reduce for quick runs. |
| `max_steps` | `-1` | When >0, overrides epochs and trains for exact optimizer steps. | Use for very large datasets or curriculum training where epochs aren’t convenient. |

Example usage:

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="checkpoints/gpt2-ft",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    gradient_accumulation_steps=8,  # global batch = 4 * 8 = 32 per device
    do_train=True,
    do_eval=True,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
trainer.train()
```

**Source anchor:** `transformers/training_args.py:228`, `transformers/trainer.py:1128`, `transformers/trainer.py:2353`

**Slide takeaway**
- Compute global batch as `per_device × devices × grad_accum`; trim it with `max_steps` when dataset-size math is awkward.
- Keep defaults for quickstarts, then tweak `gradient_accumulation_steps` or `num_train_epochs` once you inspect loss curves.

---

## Optimizer & Scheduler

| Argument | Default | Trainer behaviour | Why tweak it |
| --- | --- | --- | --- |
| `learning_rate` (`lr`) | `5e-5` | Initial LR passed to optimizer. | Tune for convergence speed/stability. |
| `weight_decay` | `0.0` | Applied to optimizer parameter groups. | Reduce overfitting by penalizing large weights. |
| `adam_beta1`, `adam_beta2`, `adam_epsilon` | `0.9`, `0.999`, `1e-8` | Configure AdamW internals. | Match hyperparameters from other training regimes. |
| `adam_bias_correction` | `True` | Optionally disable bias correction for fused optimizers. | Slight speedup when using fused Adam on supported GPUs. |
| `lr_scheduler_type` | `"linear"` | Chooses scheduler (linear, cosine, constant, etc.). | Shape learning-rate decay to match task. |
| `warmup_steps`, `warmup_ratio` | `0`, `0.0` | Defines warmup duration before full LR is reached. They’re just two ways to describe the same warmup interval. warmup_steps pins it to an exact number of optimizer steps, while warmup_ratio sets it as a fraction of the total training steps (warmup_steps = warmup_ratio * total_steps). Most configs let you specify either one; if both are given the code usually prefers the explicit step count. | Prevent large initial updates; use ratio for dataset-size independence. |
| `max_grad_norm` | `1.0` | Clipped gradient norm before optimizer step. | Stabilise training by preventing exploding gradients. |
| `optim` | `"adamw_torch"` | Selects optimizer implementation. | Switch to `adamw_torch_fused`, `adafactor`, or others for performance/memory trade-offs. |
| `optim_args` | `None` | Extra kwargs forwarded to backend optimizers. | Pass advanced options (e.g., weight decay decoupling parameters). |

#### How adam_beta1, adam_beta2, adam_epsilon are used?

In Adam/AdamW each parameter theta keeps two exponentially weighted moving averages:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t      # first moment (gradient mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t**2   # second moment (gradient variance)
```

Here `beta1 = adam_beta1`, `beta2 = adam_beta2`, and `g_t` is the current gradient. Because both accumulators start at zero, the algorithm divides them by the shrinkage factors to remove the initialization bias:

```
m_hat_t = m_t / (1 - beta1**t)
v_hat_t = v_t / (1 - beta2**t)
```

`adam_epsilon` (epsilon) appears in the parameter update to keep the denominator well-behaved:

```
theta_t = theta_{t-1} - lr * m_hat_t / (torch.sqrt(v_hat_t) + epsilon)
```

Weight decay in AdamW is applied alongside this ratio, but beta1, beta2, and epsilon are consumed exactly as shown here.

> adam_bias_correction
Adam keeps running averages of gradients (m_t) and squared gradients (v_t). Early in training those averages are biased toward zero, so the classic algorithm divides them by (1 - β₁^t) and (1 - β₂^t)—the “bias correction”—before updating parameters. Fused Adam kernels (the GPU-optimized versions) let you skip that extra math. If you toggle adam_bias_correction=False, each step is a bit cheaper, but you lose the early-step correction. On well-behaved workloads the difference is usually negligible, so “disable bias correction” becomes a small speed boost that you only take when you know the fused kernel supports it.

Snippet: optimizer creation (in `training_args.py`, `_setup_trainer_kwargs` and `Trainer.create_optimizer`):

```python
optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
```

Changing scheduler:

```python
args = TrainingArguments(
    output_dir="…",
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    warmup_ratio=0.1,
)
```

### Scheduler cheat sheet

Warmup first ramps the learning rate linearly from 0 to the base LR over `warmup_steps` (or `warmup_ratio × total_steps`), i.e.  
`lr(t) = base_lr × t / warmup_steps` for `t < warmup_steps`. After warmup each scheduler follows its own decay curve:

| `lr_scheduler_type` | Post-warmup schedule `lr(t)` (for `t ≥ warmup_steps`, `T = total_steps`) | When it shines |
| --- | --- | --- |
| `linear` (default) | `base_lr × (1 - (t - warmup_steps) / (T - warmup_steps))` | General-purpose fine-tuning; steady, predictable decay. |
| `cosine` | `0.5 × base_lr × (1 + cos(π × (t - warmup_steps) / (T - warmup_steps)))` | Smoothly anneals toward zero; good for long runs without sharp drops. |
| `cosine_with_restarts` | Same as cosine but restarts every `num_cycles`; LR jumps back to base at each restart. | Large-scale training where periodic resets help escape shallow minima. |
| `constant` | `base_lr` (no decay) | Very short runs or when external logic (e.g., manual decay) controls LR. |
| `constant_with_warmup` | Warmup ramp, then constant `base_lr`. | Keeps LR fixed after warmup; common for small LoRA/adapter runs. |
| `polynomial` | `base_lr × (1 - (t - warmup_steps) / (T - warmup_steps))^power` | Lets you tune how aggressively LR decays (e.g., power=2 gives quadratic). |

Tip: `transformers.get_scheduler` converts `lr_scheduler_type`, `num_warmup_steps`, and `num_training_steps` into the matching PyTorch scheduler, so you can experiment without touching the Trainer.

```python
import torch
from transformers import get_scheduler
import matplotlib.pyplot as plt

def plot_schedule(scheduler_type, total_steps=1000, warmup_ratio=0.1):
    base_lr = 2e-4
    optimizer = torch.optim.AdamW([torch.zeros(1)], lr=base_lr)
    scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    lrs = []
    for step in range(total_steps):
        optimizer.step()  # no-op updates
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    plt.plot(lrs, label=scheduler_type)
    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.title("Warmup + decay profile")
    plt.legend()
    plt.show()

plot_schedule("linear")
plot_schedule("cosine")
```

If you do not have matplotlib installed, swap the plotting code for a `print(lrs[:10])` to inspect numeric values instead.

### Optimizer options (`optim`)

All accepted strings live in `OptimizerNames` (`transformers/training_args.py:147`). They cluster into the families below:

| Family | `optim` values | Extra dependency | How it differs from the default |
| --- | --- | --- | --- |
| PyTorch AdamW (default) | `adamw_torch`, `adamw_torch_fused`, `adamw_torch_xla`, `adamw_torch_npu_fused` | None for CPU/GPU; fused path needs CUDA-capable PyTorch ≥2.0; XLA/NPU builds target TPUs/Ascend NPUs | Classic AdamW implementation. `*_fused` uses PyTorch’s fused kernels for better GPU throughput; `*_xla`/`*_npu_fused` match specialised hardware stacks. |
| Apex / AnyPrecision | `adamw_apex_fused`, `adamw_anyprecision` | NVIDIA Apex for fused; `torchdistx` for AnyPrecision | Lets you keep using legacy Apex kernels or mix parameter/state dtypes (e.g. bf16 math, fp32 states) to stabilise mixed-precision runs. |
| TorchAO low-bit AdamW | `adamw_torch_4bit`, `adamw_torch_8bit` | `torchao>=0.4.0`, PyTorch >2.4 | Stores optimizer states in 4/8-bit using PyTorch AO so you can fit large models without switching to bitsandbytes. |
| BitsAndBytes AdamW | `adamw_bnb_8bit` (`adamw_8bit`), `paged_adamw_32bit`, `paged_adamw_8bit` | `bitsandbytes>=0.41.1` | Quantises optimizer states; paged variants keep memory flat during LoRA-style long sequences. Good when GPU RAM is the bottleneck. |
| BitsAndBytes Lion / AdEMAMix / RMSprop | `lion_32bit`, `lion_8bit`, `paged_lion_32bit`, `paged_lion_8bit`, `ademamix`, `ademamix_8bit`, `paged_ademamix_32bit`, `paged_ademamix_8bit`, `rmsprop_bnb`, `rmsprop_bnb_8bit`, `rmsprop_bnb_32bit` | `bitsandbytes` (AdEMAMix needs ≥0.44.0) | Alternative first-order optimizers with 32/8-bit and paged flavours; trade higher throughput or different momentum schedules against AdamW-like behaviour. |
| Classic Torch optimizers | `adafactor`, `sgd`, `adagrad`, `rmsprop` | None | Use when reproducing older baselines (`sgd`, `adagrad`, `rmsprop`) or when you want Adafactor’s small-memory behaviour for very large language models. |
| Schedule-free family | `schedule_free_adamw`, `schedule_free_radam`, `schedule_free_sgd` | `schedulefree` (≥1.4.0 for RAdam) and `accelerate>=0.30.0` | Removes the need for explicit LR schedules by adapting updates internally—handy for quick experiments or when you dislike tuning warmup/decay. |
| Low-rank optimizers | `galore_adamw`, `galore_adamw_8bit`, `galore_adafactor`, `galore_adamw_layerwise`, `galore_adamw_8bit_layerwise`, `galore_adafactor_layerwise`, `apollo_adamw`, `apollo_adamw_layerwise` | `galore_torch` or `apollo_torch`; require `optim_target_modules` targeting `nn.Linear` layers | Injects low-rank updates into selected layers to save memory/compute during PEFT or giant-model training; layerwise modes forbid gradient accumulation. |
| LoMo variants | `lomo`, `adalomo` | `lomo-optim`, `accelerate>=0.30.0`, pass `model` to optimizer init | Memory-efficient optimizer for attention-heavy models; AdaLoMo adds adaptivity. Requires specifying target modules. |
| Other specialised choices | `grokadamw`, `stable_adamw` | `grokadamw`, `torch-optimi` (plus `torchdistx` for Stable AdamW) | Research optimizers: GrokAdamW emphasises faster grokking; Stable AdamW improves numeric stability via compensated summation. |

Choosing tips:
- Start with the default (`adamw_torch`)—the Trainer will silently upgrade to `adamw_torch_fused` when your PyTorch build supports it.
- Switch to bitsandbytes (`adamw_bnb_8bit` or paged variants) if optimizer-state memory is now your limiting factor; remember to install `bitsandbytes` on CUDA machines.
- Prefer `adamw_torch_xla` or `adamw_torch_npu_fused` on TPU/NPU jobs so you get vendor-maintained kernels.
- Those “low-rank” optimizer variants—galore_*, apollo_*, lomo—only apply their low-rank updates on the specific modules you point them at. They won’t automatically cover every weight matrix; you have to choose which linear layers get the compressed gradients, and you do that by filling in optim_target_modules. If you aren’t deliberately narrowing the optimizer’s scope to a curated subset of layers (and ready to name them), skip these modes and stick with a standard optimizer.
- “Schedule-free” optimizers skip the usual learning-rate schedule; they hold the LR constant or adapt it internally. Because schedulers often control early warm-up and later decay, switching them off can make training speed up, slow down, or diverge differently. So if you try the schedule_free_* settings, keep that change isolated (don’t mix in other big tweaks) and watch the training curves closely to see how convergence behavior shifts before you trust the run.

> What is PyTorch’s fused kernels
PyTorch’s “fused kernels” are single GPU kernels that bundle together what would normally be several back‑to‑back tensor operations (for example, matrix multiply → bias add → activation). Launching one fused kernel instead of many smaller ones cuts launch overhead, improves locality in GPU memory, and lets PyTorch keep data in registers/shared memory instead of writing intermediate tensors to global memory. The end result is lower latency and, on throughput workloads, noticeably higher FLOP utilization compared to running each op as its own kernel.

Example override with additional arguments:

```python
args = TrainingArguments(
    output_dir="…",
    optim="adamw_anyprecision",
    optim_args="momentum_dtype=float32,variance_dtype=bfloat16",
    gradient_checkpointing=True,
)
```

**Source anchor:** `transformers/training_args.py:147`, `transformers/trainer.py:1250`, `transformers/trainer.py:1376`

**Slide takeaway**
- Pick the optimizer family that matches your hardware limits (GPU memory, TPU/NPU kernels) before tuning exotic options.
- Treat warmup as mandatory for fresh training runs; visualise the schedule so you know when LR actually peaks.

---

## DataLoader & Collation

| Argument | Default | Trainer behaviour | Why tweak it |
| --- | --- | --- | --- |
| `dataloader_num_workers` | `0` | Number of subprocesses for data loading. | Increase for faster data pipeline (CPU permitting). |
| `dataloader_pin_memory` | `True` | Uses pinned memory for faster host→GPU copies. | Disable on CPU-only training or when memory is constrained. |
| `dataloader_drop_last` | `False` | Drops last incomplete batch when True. | Avoid very small batches when using BatchNorm or for even step counts. |
| `remove_unused_columns` | `True` | Strips dataset columns not in model inputs. | Disable if model uses extra fields from dataset. |
| `label_names` | `None` | Columns treated as labels (kept even when unused). | Manually control which dataset fields are left for compute_metrics. |
| `group_by_length` | `False` | Sorts batches by sequence length to minimize padding. | Improve efficiency for variable-length datasets; needs `length_column_name`. |
| `length_column_name` | `"length"` | Column used for length-based grouping. | Set to dataset-specific column (e.g., `"input_length"`). |
| `sortish_sampler` | `False` | Enables sortish sampler (Torch-XLA / TPU friendly). | Use on TPUs to balance sequence lengths with randomness. |
| `train_batch_size` | property | Auto-derived total batch size property. | Read-only helper; adjust per-device size / world size / grad accumulation. |

**Source anchor:** `transformers/training_args.py:433`, `transformers/trainer.py:1128`, `transformers/trainer.py:1208`

**Slide takeaway**
- Increase `dataloader_num_workers` and enable pinning once the model saturates the GPU.
- Leave `remove_unused_columns=True` unless your collator needs custom fields; it prevents silent shape mismatches.

---

## Logging, Saving, Evaluation

| Argument | Default | Trainer behaviour | Why tweak it |
| --- | --- | --- | --- |
| `logging_dir` | `None` → defaults to `os.path.join(output_dir, "runs")` | TensorBoard files are written here. | Set custom path or reuse existing logging directory. |
| `logging_strategy` | `"steps"` | Determines logging cadence (`"no"`, `"epoch"`, `"steps"`). | Match visibility requirements (per step vs per epoch). |
| `logging_steps` | `500` | When strategy=`"steps"`, log every N steps. | Increase for less console noise, decrease for finer granularity. |
| `save_strategy` | `"steps"` | Controls checkpoint frequency. | Switch to `"epoch"` for per-epoch checkpoints or `"no"` to disable. |
| `save_steps` | `500` | Step interval when strategy=`"steps"`. | Align with logging/evaluation or to manage disk usage. |
| `save_total_limit` | `None` | Caps number of checkpoints to retain. | Prevent disk from filling with older checkpoints. |
| `save_on_each_node` | `False` | Saves only on rank 0 unless True. | Enable to keep checkpoints on every node (rare need). |
| `eval_strategy` | `"no"` | Schedules evaluation; alias `evaluation_strategy`. | Choose `"steps"` or `"epoch"` to monitor validation metrics. |
| `eval_steps` | `None` | Step interval when eval strategy is `"steps"`. | Sync with logging or after each training chunk. |
| `eval_delay` | `0` | Skips initial evaluations until specified step. | Allow warmup before first eval. |
| `metric_for_best_model` | `None` | Picks metric for best model tracking. | Set to e.g. `"accuracy"` when `load_best_model_at_end=True`. |
| `greater_is_better` | `None` | Controls comparison direction for metric. | Set `False` for loss metrics; `True` for accuracy/F1. |
| `load_best_model_at_end` | `False` | After training, reloads best checkpoint. | Ensure best weights are used for final eval/push. |
| `early_stopping_patience` | `None` | Only active with callbacks (EarlyStopping). | Combine with evaluation strategy to stop early. |

Example enabling periodic eval/save:

```python
args = TrainingArguments(
    output_dir="runs",
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)
```

In `Trainer._maybe_log_save_evaluate`, these fields gate logging/eval/save frequency.

**Source anchor:** `transformers/training_args.py:502`, `transformers/trainer.py:3190`

**Slide takeaway**
- Align `logging_steps`, `save_steps`, and `eval_steps` so dashboards, checkpoints, and validation metrics line up.
- Enable `load_best_model_at_end` only when `metric_for_best_model` is set and monitored frequently enough.

---

## Evaluation & Prediction

| Argument | Default | Trainer behaviour | Why tweak it |
| --- | --- | --- | --- |
| `prediction_loss_only` | `False` | When True, evaluator returns only loss (no logits). | Reduce memory usage during large evals. |
| `include_inputs_for_metrics` | `False` | Passes model inputs to `compute_metrics`. | Needed when metrics rely on original tokens. |
| `past_index` | `-1` | Controls extraction of past_key_values for caching. | Rarely touched; set when custom caching required. |
| `eval_accumulation_steps` | `None` | Accumulates eval outputs before concatenation. | Avoid OOM by chunking large evaluation sets. |
| `predict_with_generate` | `False` | Uses `model.generate` in evaluation loop. | Evaluate generative quality (ROUGE, BLEU) instead of raw logits. |
| `generation_config` | `None` | Provides `GenerationConfig` passed to generate. | Configure decoding without mutating model defaults. |
| `generation_max_length`, `generation_num_beams` | `None` | Convenience overrides for generate parameters. | Quickly adjust decoding length/beam search in eval. |

Snippet (from `Trainer.prediction_step`):

```python
if self.args.predict_with_generate and not ignore_keys_generate:
    generated_tokens = self.model.generate(
        inputs["input_ids"],
        max_length=self.args.generation_max_length,
        num_beams=self.args.generation_num_beams,
        generation_config=self.generation_config,
    )
```

**Source anchor:** `transformers/training_args.py:566`, `transformers/trainer.py:4824`

**Slide takeaway**
- Toggle `predict_with_generate` only when metrics need decoded text; it slows evaluation if you just need logits.
- Use `eval_accumulation_steps` to avoid GPU OOM when datasets don't fit in memory during metric computation.

---

## Distributed / Parallel Training

| Argument | Default | Trainer behaviour | Why tweak it |
| --- | --- | --- | --- |
| `ddp_backend` | `None` | Chooses distributed backend. | Force `nccl`/`gloo` in heterogeneous environments. |
| `ddp_bucket_cap_mb` | `None` | Sets gradient bucket size. | Tune for communication efficiency. |
| `ddp_find_unused_parameters` | `None` | Controls DDP unused parameter detection. | Set to `False` for models without unused parameters to avoid overhead. |
| `ddp_gradient_as_bucket_view` | `None` | Use buckets as views when supported. | Slightly faster when model compliant. |
| `sharded_ddp` | `""` | Enables ZeRO-style sharding. | Lower optimizer states / gradients memory. |
| `fsdp`, `fsdp_min_num_params`, `fsdp_config` | `None` | Configure PyTorch FSDP. | Fully shard large models across nodes. |
| `deepspeed` | `None` | Loads DeepSpeed config file. | Access ZeRO Stage 3, offloading, etc. |
| `tpu_num_cores` | `None` | Specify TPU cores for XLA. | Required on TPU training. |
| `xpu_backend` | `None` | Selects Intel XPU backend. | Enable training on Intel GPUs. |
| `mp_parameters` | `""` | Megatron-LM integration string. | Required for Megatron pipeline/tensor parallelism. |
| `dispatch_batches` | `None` | Deprecated SageMaker setting. | Rarely used; keep default. |

These flags are parsed in `Trainer.__init__` and `transformers/integrations` to configure distributed strategy before the training loop starts.

**Source anchor:** `transformers/training_args.py:561`, `transformers/trainer.py:2058`, `transformers/trainer.py:2194`

**Slide takeaway**
- Match the backend (`ddp_backend`, DeepSpeed, FSDP) to your cluster tooling; mixing them leads to hard-to-debug hangs.
- Set `deepspeed`/`fsdp` only after validating single-GPU training so you can isolate distributed issues quickly.

---

## Mixed Precision & Performance

| Argument | Default | Trainer behaviour | Why tweak it |
| --- | --- | --- | --- |
| `fp16` | `False` | Activates AMP autocast for training. | Speed up training on GPUs with minimal quality loss. |
| `bf16` | `False` | Uses bfloat16 autocast when available. | Prefer on Ampere/TPU for stability over fp16. |
| `fp16_backend` | `"auto"` | Selects AMP implementation (`cuda`, `amp`, `apex`). | Override when forcing Apex or older backends. |
| `fp16_full_eval`, `bf16_full_eval` | `False` | Apply mixed precision during evaluation too. | Speed up eval but ensure metrics unaffected. |
| `half_precision_backend` | `"auto"` | Alias for mixed-precision backend choice. | Keep default unless troubleshooting. |
| `tf32` | `None` | Enables TF32 matmul on Ampere. | Boost throughput with minimal precision hit. |
| `fp16_opt_level` | `"O1"` | Legacy Apex opt level. | Only relevant when using Apex. |
| `skip_memory_metrics` | `True` | Skips memory usage reporting. | Disable to monitor GPU RAM at cost of speed. |
| `gradient_checkpointing` | `False` | Trades compute for memory by checkpointing activations. | Train larger models on limited GPUs. |
| `gradient_checkpointing_kwargs` | `{}` | Fine-tunes checkpointing behaviour. | e.g., `{"use_reentrant": False}` for compatibility. |
| `low_cpu_mem_usage` | `True` | Optimizes state loading to use less CPU RAM. | Disable if encountering edge-case bugs. |
| `optim_target_precision` | `None` | Experimental target precision control. | Future-proofing for FP8 workflows. |
| `profile` | `False` | Runs PyTorch profiler around training loop. | Debug performance hotspots (slower). |
| `label_smoothing_factor` | `0.0` | Applies label smoothing to loss computation. | Regularize classification/LangModel training. |

Usage snippet:

```python
args = TrainingArguments(
    output_dir="…",
    fp16=True,                   # use mixed precision
    gradient_checkpointing=True, # reduce activation memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
    tf32=True,                   # allow TF32 on Ampere GPUs
)
```

In `Trainer._inner_training_loop`, AMP contexts are enabled based on `fp16`/`bf16`.

**Source anchor:** `transformers/training_args.py:690`, `transformers/trainer.py:742`, `transformers/trainer.py:2446`

**Slide takeaway**
- Prefer `bf16` on Ampere/TPU when available; fall back to `fp16` plus `gradient_checkpointing` for older GPUs.
- Test toggles one at a time (precision, checkpointing, TF32) so you can attribute throughput gains or instabilities.

---

## Reproducibility & Seeds

| Argument | Default | Trainer behaviour | Why tweak it |
| --- | --- | --- | --- |
| `seed` | `42` | Calls `set_seed` to reset RNGs. | Ensure reproducibility across runs. |
| `data_seed` | `None` | Seed used for dataset shuffling. | Decouple data shuffles from global seed. |
| `dataloader_seed` | `None` | Seed for DataLoader sampling. | Deterministic batch ordering in distributed settings. |
| `disable_tqdm` | `False` | Disables tqdm progress bars when True. | Clean logging for scripts/notebooks without progress display. |
| `report_to` | `["tensorboard"]` | Configures experiment trackers. | Add `"wandb"`, `"mlflow"`, etc. |
| `run_name` | `None` | Sets experiment name for reporters. | Helps label runs in dashboards. |
| `push_to_hub` | `False` | Enables HF Hub pushes. | Share checkpoints automatically. |
| `hub_model_id`, `hub_strategy`, `hub_token`, `hub_private_repo` | `None`/defaults | Configure hub repo name, push cadence, authentication. | Required when `push_to_hub=True`. |

**Source anchor:** `transformers/training_args.py:386`, `transformers/trainer.py:452`, `transformers/trainer.py:2283`

**Slide takeaway**
- Call out `seed`/`data_seed` separately so dataset shuffling changes don’t break reproducibility.
- Disable tqdm or route `report_to` before launching multi-node jobs to avoid log spam and broken progress bars.

---

## Hands-on Exploration Ideas

- **Scheduler shootout**: Run the plotting helper with different `lr_scheduler_type` values and warmup ratios, then overlay validation loss curves to see which decay profile your task prefers.
- **Warmup sensitivity**: Train for a few hundred steps while sweeping `warmup_ratio` (0, 0.1, 0.2) to observe stability and loss spikes during the first epoch.
- **Gradient accumulation trade-offs**: Fix the global batch size and vary `per_device_train_batch_size` with matching `gradient_accumulation_steps` to spot changes in throughput and convergence.
- **Mixed precision toggles**: Compare wall-clock time and GPU memory by enabling `fp16`, `bf16`, and `gradient_checkpointing` individually so you know the cost/benefit on your hardware.
- **Logging cadence tuning**: Adjust `logging_steps`/`eval_steps` to build intuition for how often metrics should be captured for your dataset size and desired dashboard granularity.
