# [SFT](https://huggingface.co/docs/trl/main/en/sft_trainer): When do we need "Train on assistant messages only" and why?

**Short Answer:**  
Training on assistant-only messages is preferred for building modern instruction-following LLMs.

**Reasons:**
- The model learns to answer, not to ask questions.
- Stabilizes conversational behavior.
- Produces better instruction following.
- Avoids teaching the model messy or ungrammatical user text.
- Improves sample efficiency (all gradient goes to improving responses).
- This is the standard for building modern instruction-following LLMs (e.g., GPT-4/5, Claude, Gemini, Llama-3-Instruct, Qwen).

**Summary:**  
Focusing training on assistant messages leads to more reliable, efficient, and high-quality models.

---

# [SFT](https://huggingface.co/docs/trl/main/en/sft_trainer): What is the difference between "Train on assistant messages only" and "Train on completion only "?

🟢 Summary (short version)

* **Train on assistant messages only**
  = Used for **chat models**, mask *assistant messages* only.

* **Train on completion only**
  = Used for **prompt → completion** models, mask *completion section* only.

They may look similar, but they apply to **different dataset formats** and support **different user experiences** (chat vs. prompt completion).

---

# [SFT](https://huggingface.co/docs/trl/main/en/sft_trainer): What is the difference trainer = SFTTrainer( "Qwen/Qwen3-0.6B", ...), and trainer = SFTTrainer( model=model, ...)?

| Parameter passed        | What happens                                              | Use when                                                                   |
| ----------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------- |
| **`"Qwen/Qwen3-0.6B"`** | Trainer automatically loads model + tokenizer from HF Hub | Convenience, default settings                                              |
| **`model=model`**       | Trainer uses *your already-loaded* model                  | Custom architectures, local checkpoints, LoRA, quantization, model surgery |

---

# [SFT](https://huggingface.co/docs/trl/main/en/sft_trainer): If I want to apply LoRA adapters, must I go for the "pass a model object" approach?

 🟦 SHORT ANSWER

### ✔ If you want to apply LoRA, **you should always pass a model object**.

```
trainer = SFTTrainer(
    model=model_with_lora,
    ... 
)
```

🟢 **Correct workflow for LoRA**

To use LoRA, you must:

### Step 1 — Load the base model manually

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

### Step 2 — Apply LoRA (or QLoRA, or DoRA, etc.)

Example:

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, config)
```

### Step 3 — Pass the modified model to SFTTrainer

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    ...
)
```

---

---
---
---