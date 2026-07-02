"""Preprocessing utilities for Amazon review sentiment classification.

This module handles:
1) streaming a small subset of Amazon Polarity,
2) preparing train/validation/test splits,
3) tokenizing text with bert-base-uncased.
"""

from __future__ import annotations

from itertools import islice
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import TRAINING_CONFIG


def stream_tokenized_train(
    dataset_id: str = "fancyzhx/amazon_polarity",
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_length: int | None = None,
    batch_size: int = 8,
) -> Iterable:
    """Return an Iterable (streaming) tokenized training split.

    The returned object is an IterableDataset (datasets streaming) that
    tokenizes batches on-the-fly. Do NOT materialize the entire split.
    """
    resolved_tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        TRAINING_CONFIG.model_name
    )
    resolved_max_length = max_length or TRAINING_CONFIG.max_length

    def tokenize_batch(batch: dict) -> dict:
        return resolved_tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=resolved_max_length,
        )

    streamed = load_dataset(dataset_id, split="train", streaming=True)
    # Use batched mapping in streaming mode; this returns an IterableDataset.
    tokenized_stream = streamed.map(tokenize_batch, batched=True, batch_size=batch_size)
    return tokenized_stream


def materialize_validation_from_test(
    dataset_id: str = "fancyzhx/amazon_polarity",
    validation_size: int | None = None,
    seed: int | None = None,
) -> Dataset:
    """Materialize a small validation set from the test split (streamed)."""
    resolved_size = (
        validation_size if validation_size is not None else TRAINING_CONFIG.validation_materialize_size
    )
    streamed_test = load_dataset(dataset_id, split="test", streaming=True)
    rows = list(islice(streamed_test, resolved_size))
    return Dataset.from_list(rows)


def stream_tokenized_test(
    dataset_id: str = "fancyzhx/amazon_polarity",
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
) -> Iterable:
    """Return a streaming tokenized test iterable for batched evaluation.

    Use this for evaluation in streaming mode to avoid materializing the whole
    test split into RAM.
    """
    resolved_tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        TRAINING_CONFIG.model_name
    )
    resolved_max_length = max_length or TRAINING_CONFIG.max_length

    def tokenize_batch(batch: dict) -> dict:
        return resolved_tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=resolved_max_length,
        )

    streamed = load_dataset(dataset_id, split="test", streaming=True)
    tokenized_stream = streamed.map(tokenize_batch, batched=True, batch_size=batch_size)
    return tokenized_stream


def _materialize_stream_subset(
    dataset_id: str,
    split_name: str,
    sample_count: int,
) -> Dataset:
    """Stream one split and convert a small subset to an in-memory Dataset."""
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero.")

    streamed = load_dataset(dataset_id, split=split_name, streaming=True)
    subset_rows = list(islice(streamed, sample_count))

    if not subset_rows:
        raise RuntimeError(f"No rows received for split={split_name} from {dataset_id}.")

    return Dataset.from_list(subset_rows)


def load_amazon_polarity_streamed(
    train_samples: int = 2000,
    test_samples: int = 500,
) -> DatasetDict:
    """Load a real but small Amazon Polarity subset using streaming mode."""
    candidate_dataset_ids = ["fancyzhx/amazon_polarity", "amazon_polarity"]
    last_error: Exception | None = None

    for dataset_id in candidate_dataset_ids:
        try:
            print(f"Trying streaming dataset source: {dataset_id}")
            train_ds = _materialize_stream_subset(dataset_id, "train", train_samples)
            test_ds = _materialize_stream_subset(dataset_id, "test", test_samples)

            dataset = DatasetDict({"train": train_ds, "test": test_ds})
            print(f"Loaded dataset from: {dataset_id}")
            return dataset
        except Exception as error:  # noqa: BLE001
            last_error = error
            print(f"Failed source {dataset_id}: {error}")

    raise RuntimeError("Unable to load Amazon Polarity dataset from candidate sources.") from last_error


def add_text_column(dataset: DatasetDict) -> DatasetDict:
    """Combine title and content into a single text field for BERT input."""

    def combine_title_and_content(example: dict) -> dict:
        title = str(example.get("title", "")).strip()
        content = str(example.get("content", "")).strip()
        text = f"{title} {content}".strip()
        return {"text": text}

    return dataset.map(combine_title_and_content)


def create_train_validation_test_split(
    dataset: DatasetDict,
    validation_ratio: float = 0.10,
    random_seed: int = 42,
) -> DatasetDict:
    """Create train/validation/test split from streamed subset.

    Note: Stratified split is attempted first. If the label feature type does not
    support stratification, this falls back to a regular random split.
    """
    if "train" not in dataset or "test" not in dataset:
        raise KeyError("Input dataset must contain 'train' and 'test' splits.")

    try:
        train_val = dataset["train"].train_test_split(
            test_size=validation_ratio,
            seed=random_seed,
            stratify_by_column="label",
        )
    except Exception:
        train_val = dataset["train"].train_test_split(
            test_size=validation_ratio,
            seed=random_seed,
        )

    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": dataset["test"],
        }
    )


def load_tokenizer(model_name: str | None = None) -> PreTrainedTokenizerBase:
    """Load AutoTokenizer for the configured BERT model."""
    resolved_model_name = model_name or TRAINING_CONFIG.model_name
    return AutoTokenizer.from_pretrained(resolved_model_name)


def tokenize_splits(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> DatasetDict:
    """Tokenize dataset splits with truncation and fixed max length padding."""
    resolved_max_length = max_length or TRAINING_CONFIG.max_length

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=resolved_max_length,
        )

    return dataset.map(tokenize_batch, batched=True)


def print_dataset_sizes(dataset: DatasetDict) -> None:
    """Print row counts per split for quick verification."""
    print("Dataset sizes:")
    for split_name, split_data in dataset.items():
        print(f"- {split_name}: {len(split_data)}")


def print_tokenization_preview(
    tokenized_dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Print one tokenized example for sanity-checking."""
    first_row = tokenized_dataset["train"][0]
    print("Tokenization preview:")
    print(f"- label: {first_row['label']}")
    print(f"- input_ids length: {len(first_row['input_ids'])}")
    print(f"- attention_mask length: {len(first_row['attention_mask'])}")
    print(f"- first 20 token ids: {first_row['input_ids'][:20]}")
    print("- decoded preview:")
    print(tokenizer.decode(first_row["input_ids"][:30], skip_special_tokens=False))


def prepare_tokenized_datasets(
    train_samples: int = 2000,
    test_samples: int = 500,
    validation_ratio: float = 0.10,
    random_seed: int | None = None,
) -> tuple[DatasetDict, PreTrainedTokenizerBase]:
    """Full preprocessing pipeline used by training/evaluation modules."""
    resolved_seed = TRAINING_CONFIG.random_seed if random_seed is None else random_seed

    streamed_dataset = load_amazon_polarity_streamed(
        train_samples=train_samples,
        test_samples=test_samples,
    )
    text_dataset = add_text_column(streamed_dataset)
    split_dataset = create_train_validation_test_split(
        text_dataset,
        validation_ratio=validation_ratio,
        random_seed=resolved_seed,
    )

    tokenizer = load_tokenizer(TRAINING_CONFIG.model_name)
    tokenized_dataset = tokenize_splits(
        split_dataset,
        tokenizer=tokenizer,
        max_length=TRAINING_CONFIG.max_length,
    )

    return tokenized_dataset, tokenizer
