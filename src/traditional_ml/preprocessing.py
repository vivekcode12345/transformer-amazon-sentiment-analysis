"""
Text Preprocessing and Feature Extraction for Traditional ML Models
====================================================================

This module handles:
1. Loading the Amazon Polarity dataset
2. Cleaning and preparing text data
3. Converting text to TF-IDF features
4. Splitting data into train and test sets

WHY TF-IDF?
TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numbers
that traditional ML algorithms can understand. It measures how important a
word is to a document relative to all documents.

REUSES:
- Dataset loading from src.dataset_loaders (no duplication!)
"""

from __future__ import annotations

import re  # Regular expressions for text cleaning
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/preprocessing.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import Hugging Face datasets for loading amazon_polarity
from datasets import load_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default settings for text cleaning and TF-IDF
# You can modify these based on your needs

TEXT_CLEANING_CONFIG = {
    "lowercase": True,           # Convert all text to lowercase
    "remove_urls": True,         # Remove website URLs
    "remove_emails": True,       # Remove email addresses
    "remove_special_chars": True, # Remove punctuation and special characters
    "remove_numbers": False,     # Keep numbers (they might be important)
    "remove_extra_spaces": True, # Remove multiple spaces
}

TFIDF_CONFIG = {
    "max_features": 10000,       # Use top 10,000 most important words
    "ngram_range": (1, 2),       # Use single words and pairs of words
    "stop_words": "english",     # Remove common words (the, a, is, etc.)
    "max_df": 0.95,              # Ignore words that appear in 95%+ of documents
    "min_df": 2,                 # Ignore words that appear in only 1 document
}


# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    This function removes noise from text data to improve model performance.
    Think of it as cleaning your data before analysis.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
        
    Example:
        Input:  "Check out https://example.com! Great product!!!"
        Output: "check out great product"
    """
    
    # Make sure input is a string
    if not isinstance(text, str):
        return ""
    
    # Step 1: Convert to lowercase
    # Why? "Great" and "great" should be treated as the same word
    if TEXT_CLEANING_CONFIG["lowercase"]:
        text = text.lower()
    
    # Step 2: Remove URLs
    # Why? URLs don't help with sentiment analysis
    if TEXT_CLEANING_CONFIG["remove_urls"]:
        # This regex pattern matches http/https URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Step 3: Remove email addresses
    # Why? Email addresses don't indicate sentiment
    if TEXT_CLEANING_CONFIG["remove_emails"]:
        # This regex pattern matches email addresses
        text = re.sub(r'\S+@\S+', '', text)
    
    # Step 4: Remove special characters and punctuation
    # Why? We only want words, not punctuation
    if TEXT_CLEANING_CONFIG["remove_special_chars"]:
        # Keep only letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Step 5: Remove numbers (optional)
    # Why? Numbers might not be relevant for sentiment
    if TEXT_CLEANING_CONFIG["remove_numbers"]:
        text = re.sub(r'\d+', '', text)
    
    # Step 6: Remove extra spaces
    # Why? Clean up the text after removing other elements
    if TEXT_CLEANING_CONFIG["remove_extra_spaces"]:
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        text = text.strip()
    
    return text


def clean_dataset(texts: list[str]) -> list[str]:
    """
    Clean a list of texts.
    
    Args:
        texts (list[str]): List of raw texts
        
    Returns:
        list[str]: List of cleaned texts
        
    Example:
        Input:  ["Great product!", "Bad quality..."]
        Output: ["great product", "bad quality"]
    """
    # Apply clean_text to each text in the list
    cleaned_texts = [clean_text(text) for text in texts]
    
    return cleaned_texts


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_amazon_polarity_data(
    sample_limit: int | None = None,
) -> pd.DataFrame:
    """
    Load the Amazon Polarity dataset for traditional ML.
    
    This function loads the fancyzhx/amazon_polarity dataset to match
    the transformer experiments for fair comparison.
    
    Args:
        sample_limit (int, optional): Limit number of samples (for testing)
        
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
        
    Example:
        >>> df = load_amazon_polarity_data(sample_limit=None)
        >>> print(df.head())
           text  label
        0  great product  1
        1  bad quality    0
    """
    
    print("\n" + "=" * 70)
    print("LOADING AMAZON POLARITY DATASET")
    print("=" * 70)
    
    # Step 1: Load dataset from Hugging Face
    # Using fancyzhx/amazon_polarity to match transformer experiments
    print("\n1. Loading dataset from Hugging Face...")
    print("   Dataset: fancyzhx/amazon_polarity")
    
    # Load the dataset (has 'train' and 'test' splits)
    dataset = load_dataset("fancyzhx/amazon_polarity")
    
    # Step 2: Get train and test splits
    print("2. Getting train and test splits...")
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Sample exactly 50,000 train and 10,000 test samples to match transformer experiments
    print("   Sampling 50,000 train / 10,000 test samples to match transformer experiments")
    train_data = train_data.select(range(50000))
    test_data = test_data.select(range(10000))
    
    # Step 3: Convert to pandas DataFrame
    print("3. Converting to pandas DataFrame...")
    
    # Combine train and test
    train_df = train_data.to_pandas()
    test_df = test_data.to_pandas()
    
    # The dataset has 'content' and 'label' columns
    # Label is already 0 (negative) or 1 (positive)
    # Use 'content' column for review text
    df_clean = pd.DataFrame({
        'text': pd.concat([train_df['content'], test_df['content']], ignore_index=True),
        'label': pd.concat([train_df['label'], test_df['label']], ignore_index=True)
    })
    
    # Step 4: Remove any missing values
    print("4. Removing missing values...")
    initial_size = len(df_clean)
    df_clean = df_clean.dropna(subset=['text', 'label'])
    removed = initial_size - len(df_clean)
    print(f"   Removed {removed} rows with missing values")
    
    # Step 5: Display dataset info
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(df_clean):,}")
    print(f"  Positive reviews: {df_clean['label'].sum():,} ({df_clean['label'].mean():.1%})")
    print(f"  Negative reviews: {len(df_clean) - df_clean['label'].sum():,} ({1 - df_clean['label'].mean():.1%})")
    print("=" * 70)
    
    return df_clean


# ============================================================================
# TF-IDF VECTORIZATION
# ============================================================================

def create_tfidf_features(
    train_texts: list[str],
    test_texts: list[str],
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple:
    """
    Create TF-IDF features from text data.
    
    TF-IDF converts text into numerical features that ML models can use.
    It stands for Term Frequency-Inverse Document Frequency.
    
    HOW IT WORKS:
    1. Term Frequency (TF): How often a word appears in a document
    2. Inverse Document Frequency (IDF): How rare a word is across all documents
    3. TF-IDF = TF × IDF (high score = important and rare word)
    
    Args:
        train_texts (list[str]): Training text data
        test_texts (list[str]): Test text data
        max_features (int): Maximum number of words to keep
        ngram_range (tuple): Use single words (1,1) or pairs (1,2)
        
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
            - X_train_tfidf: TF-IDF features for training
            - X_test_tfidf: TF-IDF features for testing
            - vectorizer: Fitted TF-IDF vectorizer (save this for later!)
    
    Example:
        >>> train_texts = ["great product", "bad quality"]
        >>> test_texts = ["amazing item"]
        >>> X_train, X_test, vectorizer = create_tfidf_features(train_texts, test_texts)
        >>> print(X_train.shape)  # (2, some_number)
    """
    
    print("\n" + "=" * 70)
    print("CREATING TF-IDF FEATURES")
    print("=" * 70)
    
    # Step 1: Initialize the TF-IDF Vectorizer
    print("\n1. Initializing TF-IDF Vectorizer...")
    print(f"   Max features: {max_features}")
    print(f"   N-gram range: {ngram_range}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,      # Keep top N words
        ngram_range=ngram_range,        # Use unigrams and bigrams
        stop_words="english",           # Remove common English words
        max_df=0.95,                    # Ignore words in 95%+ of docs
        min_df=2,                       # Ignore words in only 1 doc
    )
    
    # Step 2: Fit on training data and transform both train and test
    print("2. Fitting TF-IDF on training data...")
    # fit() learns the vocabulary from training data
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    
    print("3. Transforming test data...")
    # transform() converts test texts using the same vocabulary
    X_test_tfidf = vectorizer.transform(test_texts)
    
    # Step 3: Display feature information
    print("\n✓ TF-IDF features created!")
    print(f"  Training features shape: {X_train_tfidf.shape}")
    print(f"  Test features shape: {X_test_tfidf.shape}")
    print(f"  Number of features (words): {len(vectorizer.get_feature_names_out()):,}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print("=" * 70)
    
    # Return the features and the fitted vectorizer
    return X_train_tfidf, X_test_tfidf, vectorizer


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def prepare_data_for_training(
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> Tuple:
    """
    Complete preprocessing pipeline for traditional ML models.
    
    This is the main function you'll use to prepare data for training.
    It does everything:
    1. Loads the dataset (50k train / 10k test to match transformer experiments)
    2. Cleans the text (optional)
    3. Uses dataset's native train/test split (no additional splitting)
    4. Creates TF-IDF features
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (unused, kept for compatibility)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before TF-IDF
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer)
            - X_train: TF-IDF features for training (sparse matrix)
            - X_test: TF-IDF features for testing (sparse matrix)
            - y_train: Training labels (pandas Series)
            - y_test: Test labels (pandas Series)
            - vectorizer: Fitted TF-IDF vectorizer (save this!)
    
    Example:
        >>> X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training()
        >>> # Now you can train any model:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression()
        >>> model.fit(X_train, y_train)
    """
    
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE FOR TRADITIONAL ML")
    print("=" * 70)
    
    # Step 1: Load the dataset
    print("\n[Step 1/5] Loading dataset...")
    df = load_amazon_polarity_data(sample_limit=sample_limit)
    
    # Step 2: Clean the text (optional)
    if clean:
        print("\n[Step 2/5] Cleaning text...")
        df['text_cleaned'] = df['text'].apply(clean_text)
        print(f"   Cleaned {len(df)} texts")
    else:
        df['text_cleaned'] = df['text']
        print("\n[Step 2/5] Skipping text cleaning (clean=False)")
    
    # Step 3: Use dataset's native train/test split (50k train / 10k test)
    # The dataset already has the correct split to match transformer experiments
    print("\n[Step 3/5] Using dataset's native train/test split...")
    print("   (50,000 train / 10,000 test to match transformer experiments)")
    
    # First 50,000 samples are train, next 10,000 are test
    train_mask = list(range(50000))
    test_mask = list(range(50000, 60000))
    
    X_train = df['text_cleaned'].iloc[train_mask]
    X_test = df['text_cleaned'].iloc[test_mask]
    y_train = df['label'].iloc[train_mask]
    y_test = df['label'].iloc[test_mask]
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Training label distribution: {y_train.mean():.1%} positive")
    print(f"   Test label distribution: {y_test.mean():.1%} positive")
    
    # Step 4: Create TF-IDF features
    print("\n[Step 4/5] Creating TF-IDF features...")
    
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(
        train_texts=X_train.tolist(),
        test_texts=X_test.tolist(),
        max_features=TFIDF_CONFIG["max_features"],
        ngram_range=TFIDF_CONFIG["ngram_range"],
    )
    
    # Step 5: Summary
    print("\n[Step 5/5] Preprocessing complete!")
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Training set:   {X_train_tfidf.shape[0]:,} samples × {X_train_tfidf.shape[1]:,} features")
    print(f"Test set:       {X_test_tfidf.shape[0]:,} samples × {X_test_tfidf.shape[1]:,} features")
    print(f"Vectorizer:     TF-IDF with {TFIDF_CONFIG['ngram_range']} n-grams")
    print(f"Max features:   {TFIDF_CONFIG['max_features']:,}")
    print("=" * 70)
    
    # Return everything needed for training
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_names(vectorizer: TfidfVectorizer) -> list[str]:
    """
    Get the feature names (words) from the fitted vectorizer.
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        list[str]: List of feature names (words)
    
    Example:
        >>> feature_names = get_feature_names(vectorizer)
        >>> print(feature_names[:10])  # ['great', 'product', 'bad', ...]
    """
    return list(vectorizer.get_feature_names_out())


def print_top_features(
    vectorizer: TfidfVectorizer,
    class_index: int = 1,
    top_n: int = 10,
) -> None:
    """
    Print the top features (words) for a specific class.
    
    This helps you understand what words are most important for each sentiment.
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        class_index (int): 0 for negative, 1 for positive
        top_n (int): Number of top features to display
        
    Example:
        >>> print_top_features(vectorizer, class_index=1, top_n=10)
        Top 10 features for Positive class:
          1. amazing (0.0234)
          2. excellent (0.0198)
          ...
    """
    feature_names = get_feature_names(vectorizer)
    
    # This is a simplified version - you'd need model coefficients for full implementation
    print(f"\nTop {top_n} features in vocabulary:")
    for i, feature in enumerate(feature_names[:top_n], 1):
        print(f"  {i}. {feature}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Run this script to test the preprocessing pipeline.
    
    This will:
    1. Load a small sample of the dataset
    2. Clean the text
    3. Create TF-IDF features
    4. Show you the results
    """
    
    print("\n" + "=" * 70)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Use a small sample for testing
    SAMPLE_LIMIT = None  # Change this to test with more/less data
    
    # Run the complete preprocessing pipeline
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = prepare_data_for_training(
        sample_limit=SAMPLE_LIMIT,
        test_size=0.2,
        random_state=42,
        clean=True,
    )
    
    # Show some examples
    print("\n" + "=" * 70)
    print("EXAMPLE: First 5 training samples (cleaned text)")
    print("=" * 70)
    
    # Note: X_train_tfidf is a sparse matrix, so we can't show text directly
    # In real usage, you would keep the original text separately before TF-IDF
    print("\n(Text samples are converted to TF-IDF features for training)")
    print("The vectorizer can transform new text using: vectorizer.transform(['your text'])")
    print("\nLabel distribution in training set:")
    print(f"  Positive: {y_train.sum()} ({y_train.mean():.1%})")
    print(f"  Negative: {len(y_train) - y_train.sum()} ({1 - y_train.mean():.1%})")
    
    print("\n" + "=" * 70)
    print("✓ Preprocessing test complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Save the vectorizer for later use: import joblib; joblib.dump(vectorizer, 'vectorizer.pkl')")
    print("2. Use X_train, y_train to train your models")
    print("3. Use X_test, y_test to evaluate your models")
    print("=" * 70)