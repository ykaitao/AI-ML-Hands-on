# Trainer.py Study Notes

## Teaching Overview

### Course Context
- Intended for a 90–120 minute hands-on class introducing Hugging Face `Trainer` for PyTorch users comfortable with transformers basics.
- Pairs with the live-coding notebook in `notebooks/llm/gpt2.py`; students clone the repo and run inside the project root.
- Focus is on demystifying high-level APIs by mapping user-visible knobs to the underlying source code.

### Learning Objectives
- Explain the life cycle of `Trainer` from initialization through training, evaluation, and checkpointing.
- Configure optimizers, schedulers, accelerator plugins, and data pipelines using `TrainingArguments`.
- Extend `Trainer` with custom loss functions, callbacks, and prediction logic.
- Run and debug hands-on labs that fine-tune a model, add custom logging/metrics, and resume from checkpoints.
- Interpret log outputs and checkpoints to reason about distributed or accelerated training runs.

### Session Flow (Suggested Timing)
- **10 min — Orientation:** Walk through the Big Picture & Constructor summary while students open the repo.
- **25 min — Lab 1: Bootstrapping Trainer:** Follow the Bootstrapping snippets to run a baseline fine-tune and inspect dataloaders.
- **30 min — Lab 2: Optimization Experiments:** Use the Optimization and Training Flow snippets to try alternative optimizers and logging callbacks.
- **20 min — Lab 3: Evaluation & Checkpointing:** Execute Evaluation & Prediction plus Checkpointing labs, compare metrics, resume from saved state.
- **15 min — Capstone:** Combine Utilities (hyperparameter search, RNG control) for a mini-experiment or homework launch.

### Prerequisites & Setup Checklist
- Python ≥3.11 with this repo’s `.venv` activated (`python -m venv .venv && source .venv/bin/activate`).
- Install dependencies: `pip install -r requirements.txt` (ensure GPU extras if available).
- VS Code users enable `"markdown.updateLinksOnFileMove": "prompt"` to keep links accurate if files move.
- Verify access to Hugging Face Hub (optional) for the checkpointing lab (`huggingface-cli login`).
- Download a small dataset or rely on the inline synthetic dataset in the snippets for quick demos.

## Reference Summary

### Big Picture
- [`Trainer`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L410) is the orchestration layer for 🤗 Transformers training, evaluation, prediction, checkpointing, and hub pushes.
- The class wraps PyTorch modules with Accelerate, Deepspeed, FSDP, or TP plugins ([`create_accelerator_and_postprocess`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5442)).
- Most public entry points (`train`, `evaluate`, `predict`, `save_model`, `hyperparameter_search`) are thin shells around deeper helpers so understanding those helpers is key.

### Constructor & Setup
- [`.venv/lib/python3.11/site-packages/transformers/trainer.py#L410`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L410) validates `TrainingArguments`, seeds RNGs, spins up a `TrainerMemoryTracker`, and instantiates the model either directly or via `model_init`.
- Device placement and distributed checks (model parallel, Deepspeed, FSDP, quantization) are handled before the model is potentially moved to device ([`Trainer.__init__`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L524)).
- Input processing defaults are selected here: `data_collator`, dataset references, tokenizer alignment, and gradient checkpointing flags.
- Accelerator configuration is centralized in [`create_accelerator_and_postprocess`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5442), which:
  - Builds an `Accelerator` with optional Deepspeed/FSDP/TP plugins.
  - Applies Trainer arguments to the plugin (gradient accumulation, non-blocking transfers).
  - Enforces compatibility constraints (e.g., `auto_find_batch_size` with ZeRO-3, `save_only_model` with FSDP).

### Data Loading Pipeline
- [`_remove_unused_columns`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1003) trims dataset columns not accepted by the model’s `forward`.
- [`_get_dataloader`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1082) wraps Torch `DataLoader` creation, adding samplers, collators, worker setup, and calls to `accelerator.prepare`.
- Public helpers `get_train_dataloader`, `get_eval_dataloader`, and `get_test_dataloader` ([`1128`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1128), [`2031`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2031), [`2075`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2075)) delegate to `_get_dataloader` with role-specific samplers.

### Optimizer & Scheduler Assembly
- [`create_optimizer_and_scheduler`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1250) is the main entry point: it calls `create_optimizer` then `create_scheduler`.
- [`create_optimizer`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1278) builds parameter groups with/without weight decay, honours injected optimizer classes via `optimizer_cls_and_kwargs`, and wires in bitsandbytes, GaLore, Apollo, LoMo, or schedule-free optimizers when requested.
- [`get_optimizer_cls_and_kwargs`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1376) is the large switchboard that translates `TrainingArguments.optim` into concrete optimizer classes, performing package availability checks and argument massaging.
- [`create_scheduler`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1843) defers to `transformers.optimization.get_scheduler`, passing warmup settings computed from the arguments.

### Training Loop Flow
- [`train`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2213) handles high-level concerns: syncing tokenizer special tokens, activating NEFTune if requested, resuming checkpoints, toggling gradient checkpointing, and dispatching into `_inner_training_loop`.
- [`_inner_training_loop`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2353) does the heavy lifting:
  - Creates/warms `DataLoader`s, computes epoch/step counts, and optionally counts tokens.
  - Lazily creates optimizers/schedulers when FSDP or model-parallel plugins require delayed initialization.
  - Wraps the model with the right parallelism (`accelerator.prepare`, Deepspeed, FSDP).
  - Implements gradient accumulation, learning-rate stepping, gradient clipping, logging cadence, and checkpoint triggers.
- [`_maybe_log_save_evaluate`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3190) centralizes the “every logging interval” book-keeping — logging losses/LR, running evals, and calling `_save_checkpoint` when needed.

### Gradient Step & Loss Computation
- [`training_step`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3981) prepares context-parallel buffers, ensures the model/optimizer are in train mode, scales the loss for gradient accumulation, and calls `accelerator.backward`.
- [`compute_loss_context_manager`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3957) and [`autocast_smart_context_manager`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3969) decide whether to run under autocast/AMP.
- [`compute_loss`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4075) handles the interplay between custom `compute_loss_func`, label smoothing, and ModelOutput dictionaries, including optional normalization by `num_items_in_batch`.
- [`_prepare_input`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3795)/[`_prepare_inputs`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3813) recursively move tensors to the target device and inject cached `past` states.

### Checkpointing & Resuming
- [`_load_from_checkpoint`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2878) restores model weights across plain, safetensors, FSDP, and PEFT adapter checkpoints, with special handling for quantized and sharded formats.
- [`_save_checkpoint`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3312) writes model, optimizer, scheduler, RNG, and trainer state, and handles save-strategy/bookkeeping.
- [`_save_optimizer_and_scheduler`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3422)/[`_load_optimizer_and_scheduler`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3487) abstract away plugin-specific serialization (TPU, Deepspeed, FSDP, SageMaker MP).
- [`save_model`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4177) is the public hook; it defers to `_save`/`_save_tpu` while respecting `save_only_model`, PEFT wrappers, and hub pushes.

### Evaluation & Prediction APIs
- [`evaluate`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4421) builds an eval dataloader, runs `evaluation_loop`, logs metrics, and returns prefixed results.
- [`predict`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4525) mirrors `evaluate` but always returns predictions/labels/metrics as a `PredictionOutput`.
- [`evaluation_loop`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4589) is the modern inference loop: wraps the model for eval, gathers tensors across processes, optionally streams batch-level metrics, and pads variable-length sequences.
- [`prediction_step`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4824) runs a single forward pass under `torch.no_grad`, collecting loss/logits/labels while supporting PEFT, Deepspeed, and context parallel buffers.
- [`prediction_loop`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5201) remains for legacy behavior and is gradually superseded by `evaluation_loop`.

### Hyperparameter Search & Utilities
- [`hyperparameter_search`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3688) delegates to Optuna, Ray Tune, SigOpt, or W&B sweeps; requires `model_init` and reinitializes arguments per trial.
- Helper getters provide quick stats: [`get_total_train_batch_size`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2303), [`get_num_trainable_parameters`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1512), etc.
- RNG management ([`_load_rng_state`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3257)/[`_save_rng_state`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3388)) ensures reproducibility when resuming.
- Hub integration lives in `push_to_hub`, `_push_from_checkpoint`, and `create_model_card` ([`4305`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4305), [`4361`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4361), [`5729`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5729)), letting long trainings stream checkpoints to 🤗 Hub.

## Study Suggestions
- Read constructor → optimizer setup → training loop in order; keep the file open with search for “def train” and “def _inner_training_loop” to follow control flow.
- Cross-reference `TrainingArguments` docs (see `args/transformers/training_args.md`) when a branch depends on a specific flag (e.g., `save_only_model`, `schedule_free_*`).
- When diving into plugin-specific behavior (Deepspeed, FSDP, TP), jump between `Trainer` helpers and Accelerate plugins to understand the hand-off boundaries.

## Lab Guide & Snippet Library

Unless otherwise noted, complete the shared setup once per terminal session, then follow the lab modules in order. Each lab includes instructor prompts (`Discuss`) and student actions (`Do`).

### Shared Setup (Run Once)

```python
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

raw_ds = Dataset.from_dict(
    {"text": ["good", "bad"], "labels": [1, 0], "meta": [0.9, 0.1]}
)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

tokenized = raw_ds.map(tokenize, batched=True)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )

base_args = TrainingArguments(
    output_dir="out/basic",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    remove_unused_columns=False,
    logging_steps=10,
    seed=42,
)
```

### Lab 1 – Bootstrapping Trainer
- **Goal:** Map the constructor flow and default data pipeline to source code.
- **Discuss:** Have students locate [`Trainer`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L410) and follow `__init__` through dataset preparation.
- **Do:** Run each task, inspecting console logs to connect arguments with behavior.

1. **Task A — Basic supervised run** ([trainer.py:410](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L410))

```python
trainer = Trainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
)
trainer.train()
```

2. **Task B — `model_init` pathway** ([trainer.py:463](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L463))

```python
model_init_trainer = Trainer(
    model_init=model_init,
    args=base_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
)
model_init_trainer.train()
```

3. **Task C — Accelerator plugins** ([trainer.py:5442](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5442))

```python
fsdp_args = TrainingArguments(
    output_dir="out/fsdp",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    fsdp="full_shard",
    gradient_checkpointing=True,
    torch_compile=True,
)
fsdp_trainer = Trainer(
    model=model_init(),
    args=fsdp_args,
    train_dataset=tokenized,
)
fsdp_trainer.train()
```

4. **Task D — Custom data collator** ([trainer.py:520](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L520), [trainer.py:1082](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1082))

```python
from dataclasses import dataclass
from transformers import DataCollatorWithPadding

@dataclass
class MetaCollator(DataCollatorWithPadding):
    def __call__(self, features):
        meta = [f["meta"] for f in features]
        for f in features:
            f.pop("meta")
        batch = super().__call__(features)
        batch["meta"] = meta
        return batch

collator = MetaCollator(tokenizer=tokenizer, padding=True)
collator_trainer = Trainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
    data_collator=collator,
)
collator_trainer.train()
```

5. **Task E — Custom dataloader** ([trainer.py:1082](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1082))

```python
import torch

class WeightedTrainer(Trainer):
    def get_train_dataloader(self):
        weights = torch.tensor(
            [5.0 if example["labels"] == 0 else 1.0 for example in self.train_dataset]
        )
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

weighted_trainer = WeightedTrainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
)
weighted_trainer.train()
```

### Lab 2 – Optimization Experiments
- **Goal:** Experiment with optimizer switches, schedulers, and gradient controls.
- **Discuss:** Compare parameter grouping in [`create_optimizer`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1278) vs. manual injection.
- **Do:** Mix and match snippets; encourage side-by-side diffing of `trainer_state.json`.

1. **Task A — Adafactor & warmup** ([trainer.py:1376](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1376), [trainer.py:1843](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1843))

```python
adafactor_args = TrainingArguments(
    output_dir="out/adafactor",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    optim="adafactor",
    learning_rate=5e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.1,
)
adafactor_trainer = Trainer(
    model=model_init(),
    args=adafactor_args,
    train_dataset=tokenized,
)
adafactor_trainer.train()
```

2. **Task B — Inject a custom optimizer** ([trainer.py:1278](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1278))

```python
from torch.optim import AdamW

model = model_init()
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

trainer_with_optimizer = Trainer(
    model=model,
    args=base_args,
    train_dataset=tokenized,
    optimizers=(optimizer, None),
)
trainer_with_optimizer.train()
```

3. **Task C — Manual optimizer + scheduler** ([trainer.py:1250](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1250))

```python
manual_trainer = Trainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
)
num_batches_per_epoch = len(manual_trainer.get_train_dataloader())
num_training_steps = num_batches_per_epoch * manual_trainer.args.num_train_epochs
manual_trainer.create_optimizer_and_scheduler(num_training_steps=num_training_steps)
manual_trainer.train()
```

4. **Task D — Cosine schedule** ([trainer.py:1843](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1843))

```python
cosine_args = TrainingArguments(
    output_dir="out/cosine",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
)
cosine_trainer = Trainer(
    model=model_init(),
    args=cosine_args,
    train_dataset=tokenized,
)
cosine_trainer.create_optimizer()
cosine_trainer.create_scheduler(num_training_steps=1000)
cosine_trainer.train()
```

5. **Task E — Gradient clipping & norm logging** ([trainer.py:2611](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2611), [trainer.py:3190](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3190))

```python
from transformers import TrainerCallback

class GradNormPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "grad_norm" in logs:
            print(f"step={state.global_step} grad_norm={logs['grad_norm']:.4f}")

clip_args = TrainingArguments(
    output_dir="out/clip",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    max_grad_norm=0.5,
    gradient_accumulation_steps=2,
    logging_strategy="steps",
    logging_steps=10,
)
clip_trainer = Trainer(
    model=model_init(),
    args=clip_args,
    train_dataset=tokenized,
    callbacks=[GradNormPrinter()],
)
clip_trainer.train()
```

### Lab 3 – Training Flow & Control
- **Goal:** Manipulate the core training loop and reason about gradient flow.
- **Discuss:** Step through [`_inner_training_loop`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2353) while running a short training job.
- **Do:** Encourage learners to inspect `trainer_state.json` between tasks.

1. **Task A — Resume from checkpoint** ([trainer.py:2213](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2213), [trainer.py:2878](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2878))

```python
resume_args = TrainingArguments(
    output_dir="out/resume",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=50,
)
resume_trainer = Trainer(
    model=model_init(),
    args=resume_args,
    train_dataset=tokenized,
)
resume_trainer.train()
resume_trainer.train(resume_from_checkpoint="out/resume/checkpoint-50")
```

2. **Task B — Custom loss shaping** ([trainer.py:4075](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4075))

```python
import torch
import torch.nn.functional as F

class AuxPenaltyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        base_loss = outputs.loss
        probs = F.softmax(outputs.logits, dim=-1)
        entropy_penalty = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        loss = base_loss + 0.01 * entropy_penalty
        return (loss, outputs) if return_outputs else loss

aux_trainer = AuxPenaltyTrainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
)
aux_trainer.train()
```

3. **Task C — Override `training_step`** ([trainer.py:3981](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3981))

```python
class ManualStepTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = outputs.loss / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.detach()

manual_trainer = ManualStepTrainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
)
manual_trainer.train()
```

4. **Task D — Custom logging hook** ([trainer.py:3190](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3190))

```python
class LossPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"step={state.global_step} loss={logs['loss']:.4f}")

callback_trainer = Trainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
    callbacks=[LossPrinter()],
)
callback_trainer.train()
```

### Lab 4 – Evaluation & Prediction
- **Goal:** Customize evaluation loops and prediction outputs.
- **Discuss:** Compare [`evaluation_loop`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4589) vs [`prediction_loop`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5201) in source.
- **Do:** Capture metrics, logits, and probability transforms for downstream tasks.

1. **Task A — Evaluation with metrics** ([trainer.py:4421](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4421))

```python
import numpy as np

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return {"validation/accuracy": (predictions == labels).mean()}

metric_trainer = Trainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
    compute_metrics=compute_metrics,
)
validation_metrics = metric_trainer.evaluate(metric_key_prefix="validation")
print(validation_metrics)
```

2. **Task B — Prediction API** ([trainer.py:4525](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4525))

```python
prediction_output = metric_trainer.predict(tokenized, metric_key_prefix="test")
print(prediction_output.predictions[:2])
print(prediction_output.metrics)
```

3. **Task C — Custom `prediction_step`** ([trainer.py:4824](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4824))

```python
import torch.nn.functional as F

class SoftmaxPredictionTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        probs = None if logits is None else F.softmax(logits, dim=-1)
        return loss, probs, labels

softmax_trainer = SoftmaxPredictionTrainer(
    model=model_init(),
    args=base_args,
    eval_dataset=tokenized,
)
softmax_trainer.evaluate()
```

4. **Task D — Loss-only evaluation** ([trainer.py:4589](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4589))

```python
loss_only_args = TrainingArguments(
    output_dir="out/loss-only",
    per_device_eval_batch_size=16,
    prediction_loss_only=True,
    eval_accumulation_steps=32,
)
loss_only_trainer = Trainer(
    model=model_init(),
    args=loss_only_args,
    eval_dataset=tokenized,
)
loss_only_trainer.evaluate()
```

5. **Task E — Legacy prediction loop** ([trainer.py:5201](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5201))

```python
legacy_trainer = Trainer(
    model=model_init(),
    args=base_args,
    eval_dataset=tokenized,
)
eval_loader = legacy_trainer.get_eval_dataloader()
legacy_output = legacy_trainer.prediction_loop(
    dataloader=eval_loader,
    description="legacy-eval",
    prediction_loss_only=False,
    ignore_keys=None,
)
print(legacy_output.metrics)
```

### Lab 5 – Checkpointing & Hub Sharing
- **Goal:** Manage checkpoints locally and push selectively to Hugging Face Hub.
- **Discuss:** Trace [`_save_checkpoint`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3312) and highlight what goes into `trainer_state.json`.
- **Do:** Emphasize cleanup and disk budgeting when running multiple saves.

1. **Task A — Manual model save** ([trainer.py:4177](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4177))

```python
save_args = TrainingArguments(
    output_dir="out/manual",
    per_device_train_batch_size=8,
    num_train_epochs=1,
)
save_trainer = Trainer(
    model=model_init(),
    args=save_args,
    train_dataset=tokenized,
)
save_trainer.train()
save_trainer.save_model("out/manual-save")
```

2. **Task B — Full state capture** ([trainer.py:3312](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3312), [trainer.py:3388](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3388))

```python
save_trainer.save_state()
save_trainer._save_rng_state("out/manual/rng")
save_trainer._load_rng_state("out/manual/rng")
```

3. **Task C — Push to 🤗 Hub** ([trainer.py:4305](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L4305))

```python
save_trainer.push_to_hub(
    repo_id="username/distilbert-demo",
    commit_message="Initial fine-tune",
    private=True,
)
```

4. **Task D — Custom model card** ([trainer.py:5729](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5729))

```python
save_trainer.create_model_card(
    model_name="distilbert-demo",
    dataset_name="toy-sentiment",
    language="en",
    metrics=validation_metrics,
)
```

### Lab 6 – Utilities & Extensions
- **Goal:** Explore advanced APIs for experimentation and distributed runs.
- **Discuss:** Review [`hyperparameter_search`](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3688) requirements (`model_init`, reinitialization).
- **Do:** Position these as optional stretch goals or homework.

1. **Task A — Hyperparameter search** ([trainer.py:3688](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3688))

```python
import optuna

def hp_space(trial):
    return {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)}

search_args = TrainingArguments(
    output_dir="out/hp",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=50,
)
search_trainer = Trainer(
    model_init=model_init,
    args=search_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
    compute_metrics=compute_metrics,
)

best_run = search_trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    compute_objective=lambda metrics: metrics["validation/accuracy"],
    n_trials=5,
)
print(best_run)
```

2. **Task B — Parameter accounting** ([trainer.py:1512](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L1512))

```python
param_trainer = Trainer(
    model=model_init(),
    args=base_args,
    train_dataset=tokenized,
)
print(param_trainer.get_num_trainable_parameters())
```

3. **Task C — Effective batch size** ([trainer.py:2303](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L2303))

```python
print(param_trainer.get_total_train_batch_size())
```

4. **Task D — Explicit RNG control** ([trainer.py:3257](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L3257))

```python
param_trainer._save_rng_state("out/basic/rng")
param_trainer._load_rng_state("out/basic/rng")
```

5. **Task E — Distributed barrier helpers** ([trainer.py:5757](../../.venv/lib/python3.11/site-packages/transformers/trainer.py#L5757))

```python
param_trainer.accelerator.wait_for_everyone()
if param_trainer.is_world_process_zero():
    print("Primary rank finished syncing.")
```
