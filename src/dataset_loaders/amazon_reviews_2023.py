"""Amazon Reviews 2023 dataset preparation for multimodal analysis.

Loads the Amazon Reviews 2023 dataset from Hugging Face and creates balanced
subsets for specific product categories (Health_and_Household, Electronics).

This module prepares data for both text-only and multimodal pipelines by
preserving review text, product title, ASIN, and category information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

# Target categories for multimodal analysis
# Note: Must use underscores instead of ampersand/space to match dataset configs
TARGET_CATEGORIES = ["Health_and_Household", "Electronics"]

# Target sample count per category
SAMPLES_PER_CATEGORY = 15000


def load_amazon_reviews_2023(
    sample_limit: int | None = None,
) -> Dataset:
    """Load Amazon Reviews 2023 dataset from Hugging Face.

    Args:
        sample_limit: If provided, limit total samples loaded for debugging.

    Returns:
        Hugging Face Dataset with Amazon Reviews 2023 data for target categories.
    """
    print("Loading Amazon Reviews 2023 dataset from Hugging Face...")
    print("(Using HTTP streaming to avoid downloading full category files)")
    
    import requests
    
    # Calculate samples per category
    samples_per_category = None
    if sample_limit:
        samples_per_category = sample_limit // len(TARGET_CATEGORIES) + 1
    
    # Load each target category separately using HTTP streaming
    all_examples = []
    
    for category in TARGET_CATEGORIES:
        print(f"  Loading {category}...")
        filename = f"raw/review_categories/{category}.jsonl"
        
        # Stream the JSONL file directly from Hugging Face
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/{filename}"
        
        category_examples = []
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.strip():
                        category_examples.append(json.loads(line))
                        
                        # Stop early if we have enough samples for this category
                        if samples_per_category and len(category_examples) >= samples_per_category:
                            break
                            
        except Exception as e:
            print(f"    Error streaming {category}: {e}")
            print(f"    Falling back to hf_hub_download...")
            
            # Fallback: download the file
            filepath = hf_hub_download(
                repo_id="McAuley-Lab/Amazon-Reviews-2023",
                filename=filename,
                repo_type="dataset"
            )
            
            category_examples = []
            with open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        category_examples.append(json.loads(line))
                        
                        # Stop early if we have enough samples for this category
                        if samples_per_category and len(category_examples) >= samples_per_category:
                            break
        
        print(f"    Loaded {len(category_examples):,} examples from {category}")
        all_examples.extend(category_examples)
    
    # Create a Dataset from all examples
    dataset = Dataset.from_list(all_examples)
    
    # Apply final sample limit if specified
    if sample_limit and len(dataset) > sample_limit:
        print(f"Limiting to {sample_limit} samples for testing")
        dataset = dataset.select(range(sample_limit))

    return dataset


def filter_categories(dataset: Dataset, categories: list[str]) -> Dataset:
    """Filter dataset to specific product categories.

    Args:
        dataset: Input dataset with category information.
        categories: List of category names to keep.

    Returns:
        Filtered dataset containing only specified categories.
    """
    print(f"\nFiltering dataset for categories: {categories}")

    def is_target_category(example: dict) -> bool:
        """Check if example belongs to target category."""
        category = example.get("category", "")
        return category in categories

    filtered = dataset.filter(is_target_category)
    print(f"Filtered dataset size: {len(filtered)} samples")

    return filtered


def balance_by_category(
    dataset: Dataset,
    categories: list[str],
    samples_per_category: int,
) -> Dataset:
    """Balance dataset to have equal samples per category.

    Args:
        dataset: Input dataset (pre-filtered by category).
        categories: List of categories to balance.
        samples_per_category: Target samples per category.

    Returns:
        Balanced dataset with ~samples_per_category per category.
    """
    print(f"\nBalancing dataset to {samples_per_category} samples per category")

    balanced_data = []

    for category in categories:
        print(f"  Processing {category}...")

        def is_category(example: dict) -> bool:
            return example.get("category", "") == category

        category_dataset = dataset.filter(is_category)
        category_size = len(category_dataset)
        print(f"    Found {category_size} samples in {category}")

        if category_size < samples_per_category:
            print(
                f"    ⚠️  Only {category_size} samples (target: {samples_per_category})"
            )
            samples_to_take = category_size
        else:
            samples_to_take = samples_per_category

        # Take first N samples (could randomize if desired)
        subset = category_dataset.select(range(samples_to_take))
        balanced_data.append(subset)

        print(f"    Took {samples_to_take} samples from {category}")

    # Combine all categories
    from datasets import concatenate_datasets
    combined = concatenate_datasets(balanced_data)
    print(f"\nTotal balanced dataset size: {len(combined)} samples")

    return combined


def extract_fields(dataset: Dataset) -> Dataset:
    """Extract and standardize required fields for multimodal pipeline.

    Args:
        dataset: Input dataset.

    Returns:
        Dataset with standardized fields:
        - review_text: Cleaned review text
        - product_title: Product title
        - asin: Amazon Standard Identification Number
        - category: Product category
        - rating: Review rating (1-5)
    """
    print("\nExtracting and standardizing fields...")

    def extract_example(example: dict) -> dict:
        """Extract required fields from example."""
        return {
            "review_text": example.get("text", "").strip(),
            "product_title": example.get("title", "").strip(),
            "asin": example.get("asin", ""),
            "category": example.get("category", ""),
            "rating": example.get("rating", 0),
        }

    extracted = dataset.map(extract_example, remove_columns=dataset.column_names)
    return extracted


def remove_duplicates(dataset: Dataset) -> Dataset:
    """Remove duplicate reviews (by ASIN + review text).

    Args:
        dataset: Input dataset.

    Returns:
        Dataset with duplicate reviews removed.
    """
    print("\nRemoving duplicate reviews...")

    # Convert to pandas for deduplication
    df = dataset.to_pandas()
    initial_size = len(df)

    # Drop exact duplicates based on ASIN + review_text
    df = df.drop_duplicates(subset=["asin", "review_text"], keep="first")

    removed_count = initial_size - len(df)
    print(f"Removed {removed_count} duplicate reviews")
    print(f"Final dataset size: {len(df)} samples")

    # Convert back to Dataset
    return Dataset.from_pandas(df)


def remove_empty_reviews(dataset: Dataset) -> Dataset:
    """Remove reviews with missing or empty review text.

    Args:
        dataset: Input dataset.

    Returns:
        Dataset with empty reviews removed.
    """
    print("\nRemoving empty/invalid reviews...")

    df = dataset.to_pandas()
    initial_size = len(df)

    # Remove rows with empty review text
    df = df[df["review_text"].notna()]
    df = df[df["review_text"].str.len() > 5]  # Minimum 5 characters

    removed_count = initial_size - len(df)
    print(f"Removed {removed_count} empty/invalid reviews")
    print(f"Dataset size after cleanup: {len(df)} samples")

    return Dataset.from_pandas(df)


def get_category_statistics(dataset: Dataset) -> dict:
    """Compute statistics on category distribution.

    Args:
        dataset: Input dataset.

    Returns:
        Dictionary with category statistics.
    """
    df = dataset.to_pandas()
    stats = {
        "total_samples": len(df),
        "categories": df["category"].value_counts().to_dict(),
        "avg_review_length": df["review_text"].str.len().mean(),
        "min_review_length": df["review_text"].str.len().min(),
        "max_review_length": df["review_text"].str.len().max(),
        "avg_rating": df["rating"].mean(),
    }
    return stats


def print_statistics(stats: dict) -> None:
    """Pretty-print dataset statistics.

    Args:
        stats: Statistics dictionary.
    """
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total samples: {stats['total_samples']:,}")
    print("\nSamples per category:")
    for category, count in stats["categories"].items():
        pct = (count / stats["total_samples"]) * 100
        print(f"  - {category}: {count:,} ({pct:.1f}%)")
    print("\nReview text length (characters):")
    print(f"  - Average: {stats['avg_review_length']:.0f}")
    print(f"  - Min: {stats['min_review_length']:.0f}")
    print(f"  - Max: {stats['max_review_length']:.0f}")
    print(f"\nAverage rating: {stats['avg_rating']:.2f} / 5.0")
    print("=" * 70)


def prepare_dataset(
    output_path: Path | str,
    samples_per_category: int = SAMPLES_PER_CATEGORY,
    categories: list[str] | None = None,
    sample_limit: int | None = None,
) -> pd.DataFrame:
    """End-to-end dataset preparation pipeline.

    Args:
        output_path: Path to save processed CSV.
        samples_per_category: Target samples per category.
        categories: Target categories (default: Health & Household, Electronics).
        sample_limit: Limit samples loaded (for testing).

    Returns:
        Processed dataset as pandas DataFrame.
    """
    output_path = Path(output_path)
    categories = categories or TARGET_CATEGORIES

    print("\n" + "=" * 70)
    print("AMAZON REVIEWS 2023 - DATASET PREPARATION")
    print("=" * 70)

    # Step 1: Load full dataset
    dataset = load_amazon_reviews_2023(sample_limit=sample_limit)
    print(f"Loaded dataset size: {len(dataset):,} samples")

    # Step 2: Filter to target categories
    dataset = filter_categories(dataset, categories)

    # Step 3: Balance by category
    dataset = balance_by_category(dataset, categories, samples_per_category)

    # Step 4: Extract required fields
    dataset = extract_fields(dataset)

    # Step 5: Remove duplicates
    dataset = remove_duplicates(dataset)

    # Step 6: Remove empty reviews
    dataset = remove_empty_reviews(dataset)

    # Step 7: Get statistics
    stats = get_category_statistics(dataset)
    print_statistics(stats)

    # Step 8: Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = dataset.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved to: {output_path}")

    # Save statistics
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics saved to: {stats_path}")

    return df


if __name__ == "__main__":
    from configs.config import PATH_CONFIG

    # Prepare dataset
    output_csv = PATH_CONFIG.data_dir / "processed" / "amazon_reviews_2023_balanced.csv"
    df = prepare_dataset(output_csv)

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
