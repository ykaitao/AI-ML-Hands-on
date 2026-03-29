import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar

import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast, LlamaTokenizer, Trainer, TrainingArguments


DEFAULT_DATA_SUBPATH: Final[str] = "data_examples/the-verdict.txt"
DEFAULT_DATASET_TEST_SIZE: Final[float] = 0.1
DEFAULT_DATASET_SEED: Final[int] = 42
DEFAULT_TOKENIZER_PREVIEW_CHARS: Final[int] = 500
DEFAULT_TOKENIZER_PREVIEW_TEXT: Final[str] = "I love large language models"
DEFAULT_MAP_BATCH_SIZE: Final[int] = 32
DEFAULT_NUM_RETURN_SEQUENCES: Final[int] = 3
DEFAULT_TOP_K: Final[int] = 50
DEFAULT_TOP_P: Final[float] = 0.95

SPECIAL_TOKENS: Final[dict[str, str]] = {
    "bos_token": "<s>",
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
}

BYTE_LEVEL_SPECIAL_TOKENS: Final[list[str]] = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
]
TRAINING_ARG_DEFAULTS: Final[dict[str, Any]] = {
    "report_to": "none",
    "dataloader_num_workers": 0,
    "fp16": False,
}

TokenizerType = TypeVar("TokenizerType")
ExampleBatch = dict[str, list[list[int]]]
DecodedExample = tuple[str, Sequence[int]]


@dataclass(frozen=True)
class WorkshopPaths:
    """Resolved paths shared by the workshop scripts."""

    script_dir: str
    data_file: str
    tokenizer_dir: str
    output_dir: str


def build_workshop_paths(
    script_file: str,
    tokenizer_subdir: str,
    output_subdir: str,
    data_subpath: str = DEFAULT_DATA_SUBPATH,
) -> WorkshopPaths:
    """Build the file-system paths used by a workshop script."""

    script_dir = Path(script_file).resolve().parent
    return WorkshopPaths(
        script_dir=str(script_dir),
        data_file=str(script_dir / data_subpath),
        tokenizer_dir=str(script_dir / tokenizer_subdir),
        output_dir=str(script_dir / output_subdir),
    )


def load_text_dataset_with_validation(
    data_file: str,
    test_size: float = DEFAULT_DATASET_TEST_SIZE,
    seed: int = DEFAULT_DATASET_SEED,
):
    """Load a text dataset and create a validation split when absent."""

    dataset = load_dataset("text", data_files=data_file)
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
        dataset["validation"] = dataset.pop("test")
    return dataset


def count_unique_words(split: Iterable[dict[str, str]]) -> int:
    """Count unique whitespace-delimited words in a dataset split."""

    return len(set(word for example in split for word in example["text"].split()))


def print_dataset_overview(
    dataset: Any, preview_chars: int = DEFAULT_TOKENIZER_PREVIEW_CHARS
) -> None:
    """Print a compact overview of the loaded dataset."""

    print("Dataset loaded.")
    print(dataset)
    print("\nSample text:")
    print(dataset["train"][0]["text"][:preview_chars])
    print("Unique words in training set:", count_unique_words(dataset["train"]))


def detect_device_and_optimizer() -> tuple[str, str]:
    """Choose a simple device label and optimizer name for the current runtime."""

    if "COLAB_TPU_ADDR" in os.environ:
        return "tpu", "adamw_torch"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and torch.backends.mps.is_available():
        return "mps", "adamw_torch"

    if torch.cuda.is_available():
        return "gpu", "adamw_torch_fused"

    return "cpu", "adamw_torch"


def apply_standard_special_tokens(tokenizer: TokenizerType) -> TokenizerType:
    """Apply the same special-token configuration across tokenizer types."""

    tokenizer.pad_token = SPECIAL_TOKENS["pad_token"]
    tokenizer.eos_token = SPECIAL_TOKENS["eos_token"]
    tokenizer.bos_token = SPECIAL_TOKENS["bos_token"]
    tokenizer.unk_token = SPECIAL_TOKENS["unk_token"]
    return tokenizer


def load_or_train_byte_level_bpe_tokenizer(
    data_file: str,
    tokenizer_dir: str,
    vocab_size: int = 1500,
    min_frequency: int = 2,
) -> GPT2TokenizerFast:
    """Load an existing byte-level BPE tokenizer or train it on demand."""

    tokenizer_path = Path(tokenizer_dir)
    if not tokenizer_path.exists():
        print("Training new tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=data_file,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=BYTE_LEVEL_SPECIAL_TOKENS,
        )
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(str(tokenizer_path))

    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_path))
    return apply_standard_special_tokens(tokenizer)


def load_or_train_sentencepiece_tokenizer(
    data_file: str,
    tokenizer_dir: str,
    vocab_size: int,
    model_prefix: str = "llama",
) -> LlamaTokenizer:
    """Load an existing SentencePiece tokenizer or train it on demand."""

    import sentencepiece as spm

    tokenizer_path = Path(tokenizer_dir)
    spm_model_path = tokenizer_path / f"{model_prefix}.model"

    if not spm_model_path.exists():
        print("Training new SentencePiece tokenizer...")
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.Train(
            input=data_file,
            model_prefix=str(tokenizer_path / model_prefix),
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="unigram",
            user_defined_symbols=["<s>", "</s>", "<pad>", "<mask>"],
        )

    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer(vocab_file=str(spm_model_path))
    return apply_standard_special_tokens(tokenizer)


def _group_texts_into_blocks(examples: ExampleBatch, block_size: int) -> ExampleBatch:
    concatenated = {key: sum(examples[key], []) for key in examples}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size

    grouped_examples = {
        key: [
            tokens[index : index + block_size]
            for index in range(0, total_length, block_size)
        ]
        for key, tokens in concatenated.items()
    }
    grouped_examples["labels"] = grouped_examples["input_ids"].copy()
    return grouped_examples


def tokenize_and_group_texts(
    dataset: Any,
    tokenizer: Any,
    block_size: int,
    map_batch_size: int = DEFAULT_MAP_BATCH_SIZE,
):
    """Tokenize text examples and group them into fixed-length causal-LM blocks."""

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example["input_ids"]) > 0
    )

    return tokenized_dataset.map(
        lambda examples: _group_texts_into_blocks(examples, block_size),
        batched=True,
        batch_size=map_batch_size,
    )


def verify_causal_lm_dataset(lm_datasets: Any) -> None:
    """Assert that the causal-LM dataset contains the expected columns."""

    for split in lm_datasets:
        for column in ["input_ids", "attention_mask", "labels"]:
            assert column in lm_datasets[split].features


def prepare_causal_lm_datasets(
    dataset: Any,
    tokenizer: Any,
    block_size: int,
    map_batch_size: int = DEFAULT_MAP_BATCH_SIZE,
):
    """Create and validate tokenized causal-LM datasets."""

    lm_datasets = tokenize_and_group_texts(
        dataset,
        tokenizer,
        block_size=block_size,
        map_batch_size=map_batch_size,
    )
    verify_causal_lm_dataset(lm_datasets)
    print("\nDataset ready for training!")
    return lm_datasets


def print_tokenizer_preview(
    tokenizer: Any,
    sample_text: str = DEFAULT_TOKENIZER_PREVIEW_TEXT,
    extra_decode_examples: Sequence[DecodedExample] | None = None,
) -> None:
    """Print a compact tokenizer smoke test for workshop notebooks."""

    print(f"\nTokenizer ready. Vocab size: {len(tokenizer)}")
    sample_ids = tokenizer.encode(sample_text)
    print("Encoded:", sample_ids)
    print("Decoded:", tokenizer.decode(sample_ids))

    if extra_decode_examples is not None:
        for label, token_ids in extra_decode_examples:
            print(label, tokenizer.decode(token_ids))


def build_training_args(
    output_dir: str, optim_type: str, **overrides: Any
) -> TrainingArguments:
    """Build TrainingArguments with shared defaults for the workshop scripts."""

    training_kwargs = {
        "output_dir": output_dir,
        "optim": optim_type,
        **TRAINING_ARG_DEFAULTS,
    }
    training_kwargs.update(overrides)
    return TrainingArguments(**training_kwargs)


def train_and_report(trainer: Trainer) -> tuple[dict[str, Any], torch.Tensor]:
    """Run training, evaluate, and report perplexity."""

    print("\nStarting training...")
    trainer.train()
    results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(results["eval_loss"]))
    print(f"\nEvaluation Perplexity: {perplexity.item():.2f}")
    return results, perplexity


def generate_and_print_samples(
    model: Any,
    tokenizer: Any,
    input_text: str,
    max_length: int,
    num_return_sequences: int = DEFAULT_NUM_RETURN_SEQUENCES,
    do_sample: bool = True,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    **overrides: Any,
):
    """Generate and print decoded samples from a causal language model."""

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_length": max_length,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs.update(overrides)

    outputs = model.generate(**inputs, **generation_kwargs)
    print("\nGenerated Texts:")
    for output in outputs:
        print(tokenizer.decode(output, skip_special_tokens=True))
    return outputs
