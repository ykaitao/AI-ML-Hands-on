from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import os

import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast, LlamaTokenizer


SPECIAL_TOKENS = {
    "bos_token": "<s>",
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
}

BYTE_LEVEL_SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
TokenizerType = TypeVar("TokenizerType")


@dataclass(frozen=True)
class WorkshopPaths:
    script_dir: str
    data_file: str
    tokenizer_dir: str
    output_dir: str


def build_workshop_paths(
    script_file,
    tokenizer_subdir,
    output_subdir,
    data_subpath="data_examples/the-verdict.txt",
):
    script_dir = Path(script_file).resolve().parent
    return WorkshopPaths(
        script_dir=str(script_dir),
        data_file=str(script_dir / data_subpath),
        tokenizer_dir=str(script_dir / tokenizer_subdir),
        output_dir=str(script_dir / output_subdir),
    )


def load_text_dataset_with_validation(data_file, test_size=0.1, seed=42):
    dataset = load_dataset("text", data_files=data_file)
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
        dataset["validation"] = dataset.pop("test")
    return dataset


def count_unique_words(split):
    return len(set(word for example in split for word in example["text"].split()))


def detect_device_and_optimizer():
    return (
        ("tpu", "adamw_torch")
        if "COLAB_TPU_ADDR" in os.environ
        else (
            ("mps", "adamw_torch")
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else (
                ("gpu", "adamw_torch_fused")
                if torch.cuda.is_available()
                else ("cpu", "adamw_torch")
            )
        )
    )


def apply_standard_special_tokens(tokenizer: TokenizerType) -> TokenizerType:
    tokenizer.pad_token = SPECIAL_TOKENS["pad_token"]
    tokenizer.eos_token = SPECIAL_TOKENS["eos_token"]
    tokenizer.bos_token = SPECIAL_TOKENS["bos_token"]
    tokenizer.unk_token = SPECIAL_TOKENS["unk_token"]
    return tokenizer


def load_or_train_byte_level_bpe_tokenizer(
    data_file,
    tokenizer_dir,
    vocab_size=1500,
    min_frequency=2,
) -> GPT2TokenizerFast:
    if not os.path.exists(tokenizer_dir):
        print("Training new tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=data_file,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=BYTE_LEVEL_SPECIAL_TOKENS,
        )
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_model(tokenizer_dir)

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    return apply_standard_special_tokens(tokenizer)


def load_or_train_sentencepiece_tokenizer(
    data_file,
    tokenizer_dir,
    vocab_size,
    model_prefix="llama",
) -> LlamaTokenizer:
    import sentencepiece as spm

    spm_model_path = os.path.join(tokenizer_dir, f"{model_prefix}.model")

    if not os.path.exists(spm_model_path):
        print("Training new SentencePiece tokenizer...")
        os.makedirs(tokenizer_dir, exist_ok=True)
        spm.SentencePieceTrainer.Train(
            input=data_file,
            model_prefix=os.path.join(tokenizer_dir, model_prefix),
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="unigram",
            user_defined_symbols=["<s>", "</s>", "<pad>", "<mask>"],
        )

    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer(vocab_file=spm_model_path)
    return apply_standard_special_tokens(tokenizer)


def tokenize_and_group_texts(dataset, tokenizer, block_size, map_batch_size=32):
    tokenized = dataset.map(
        lambda example: tokenizer(example["text"]),
        batched=True,
        remove_columns=["text"],
    )
    tokenized = tokenized.filter(lambda example: len(example["input_ids"]) > 0)

    def group_texts(examples):
        concatenated = {key: sum(examples[key], []) for key in examples}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            key: [
                tokens[index : index + block_size]
                for index in range(0, total_length, block_size)
            ]
            for key, tokens in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(group_texts, batched=True, batch_size=map_batch_size)


def verify_causal_lm_dataset(lm_datasets):
    for split in lm_datasets:
        for column in ["input_ids", "attention_mask", "labels"]:
            assert column in lm_datasets[split].features
