# Traditional ML Module

A clean, modular preprocessing pipeline for traditional machine learning algorithms on the Amazon Polarity dataset.

## What This Module Does

This module prepares your data for traditional ML algorithms (Logistic Regression, Naive Bayes, SVM, etc.) by:

1. **Loading** the Amazon Polarity dataset (reuses your existing dataset loader)
2. **Cleaning** text data (removing URLs, emails, special characters, etc.)
3. **Splitting** data into train and test sets
4. **Vectorizing** text into TF-IDF features that ML models can understand

## Quick Start

### Basic Usage

```python
from src.traditional_ml.preprocessing import prepare_data_for_training

# Prepare data for training
X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
    sample_limit=1000,  # Use 1000 samples for testing
    test_size=0.2,      # 20% for testing
    clean=True          # Clean text before vectorization
)

# Now you can train any scikit-learn model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Text Cleaning

```python
from src.traditional_ml.preprocessing import clean_text, clean_dataset

# Clean single text
cleaned = clean_text("Check out https://example.com! Great product!!!")
print(cleaned)  # Output: "check out great product"

# Clean multiple texts
texts = ["Great product!", "Bad quality...", "Visit http://test.com"]
cleaned_texts = clean_dataset(texts)
print(cleaned_texts)  # Output: ['great product', 'bad quality', 'visit']
```

### TF-IDF Features

```python
from src.traditional_ml.preprocessing import create_tfidf_features

train_texts = ["great product", "bad quality", "amazing item"]
test_texts = ["excellent purchase"]

X_train, X_test, vectorizer = create_tfidf_features(
    train_texts=train_texts,
    test_texts=test_texts,
    max_features=1000,
    ngram_range=(1, 2)
)

print(f"Training features shape: {X_train.shape}")  # (3, some_number)
print(f"Test features shape: {X_test.shape}")        # (1, some_number)
```

## Module Structure

```
src/traditional_ml/
├── __init__.py              # Package initialization
├── README.md                # This file
├── preprocessing.py         # Main preprocessing pipeline
│
├── (Coming Soon)
├── models/                  # Model implementations
│   ├── base_model.py
│   ├── logistic_regression.py
│   ├── multinomial_nb.py
│   ├── bernoulli_nb.py
│   ├── linear_svm.py
│   ├── sgd_classifier.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── knn.py
│
├── trainer.py               # Training orchestration
├── evaluator.py             # Model evaluation
├── metrics.py               # Metrics computation
├── utils.py                 # Helper utilities
└── cli.py                   # Command-line interface
```

## Key Features

### 1. Text Cleaning

The `clean_text()` function removes noise from text:

- **Lowercase conversion**: "Great" → "great"
- **URL removal**: Removes http/https links
- **Email removal**: Removes email addresses
- **Special character removal**: Removes punctuation
- **Number removal** (optional): Can keep or remove numbers
- **Extra space removal**: Cleans up whitespace

### 2. TF-IDF Vectorization

Converts text into numerical features using Term Frequency-Inverse Document Frequency:

- **max_features**: Keep top N most important words (default: 10,000)
- **ngram_range**: Use single words (1,1) or pairs (1,2) (default: (1,2))
- **stop_words**: Remove common words like "the", "a", "is" (default: "english")
- **max_df**: Ignore words that appear in 95%+ of documents
- **min_df**: Ignore words that appear in only 1 document

### 3. Data Splitting

Automatically splits data into train and test sets:

- **stratified sampling**: Keeps same label distribution in train/test
- **reproducibility**: Fixed random seed for consistent results
- **flexible test size**: Configure test set proportion

## Configuration

### Text Cleaning Configuration

```python
TEXT_CLEANING_CONFIG = {
    "lowercase": True,           # Convert to lowercase
    "remove_urls": True,         # Remove URLs
    "remove_emails": True,       # Remove emails
    "remove_special_chars": True, # Remove punctuation
    "remove_numbers": False,     # Keep numbers
    "remove_extra_spaces": True, # Remove extra spaces
}
```

### TF-IDF Configuration

```python
TFIDF_CONFIG = {
    "max_features": 10000,       # Top 10,000 words
    "ngram_range": (1, 2),       # Unigrams + bigrams
    "stop_words": "english",     # Remove common words
    "max_df": 0.95,              # Ignore words in 95%+ docs
    "min_df": 2,                 # Ignore words in 1 doc
}
```

## API Reference

### Main Functions

#### `prepare_data_for_training()`

Complete preprocessing pipeline.

```python
def prepare_data_for_training(
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> Tuple:
    """
    Returns:
        X_train: TF-IDF features for training (sparse matrix)
        X_test: TF-IDF features for testing (sparse matrix)
        y_train: Training labels (pandas Series)
        y_test: Test labels (pandas Series)
        vectorizer: Fitted TF-IDF vectorizer
    """
```

#### `clean_text()`

Clean and normalize text.

```python
def clean_text(text: str) -> str:
    """
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
```

#### `create_tfidf_features()`

Create TF-IDF features from text.

```python
def create_tfidf_features(
    train_texts: list[str],
    test_texts: list[str],
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple:
    """
    Returns:
        X_train_tfidf: TF-IDF features for training
        X_test_tfidf: TF-IDF features for testing
        vectorizer: Fitted TF-IDF vectorizer
    """
```

#### `load_amazon_polarity_data()`

Load the Amazon Polarity dataset.

```python
def load_amazon_polarity_data(
    sample_limit: int | None = None,
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with 'text' and 'label' columns
    """
```

### Utility Functions

#### `get_feature_names()`

Get feature names from vectorizer.

```python
def get_feature_names(vectorizer: TfidfVectorizer) -> list[str]:
    """Returns list of feature names (words)"""
```

#### `print_top_features()`

Print top features for a class.

```python
def print_top_features(
    vectorizer: TfidfVectorizer,
    class_index: int = 1,
    top_n: int = 10,
) -> None:
    """Print top N features in vocabulary"""
```

## Examples

### Example 1: Complete Pipeline

```python
from src.traditional_ml.preprocessing import prepare_data_for_training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare data
X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
    sample_limit=2000,
    test_size=0.2
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")
```

### Example 2: Save and Load Vectorizer

```python
import joblib
from src.traditional_ml.preprocessing import prepare_data_for_training

# Prepare data
X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training()

# Save vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load vectorizer later
vectorizer = joblib.load('vectorizer.pkl')

# Use it to transform new text
new_text = ["This product is amazing!"]
new_features = vectorizer.transform(new_text)
```

### Example 3: Custom Text Cleaning

```python
from src.traditional_ml.preprocessing import clean_text

# Default cleaning
text1 = clean_text("Check out https://example.com! Great product!!!")
print(text1)  # "check out great product"

# Modify configuration
from src.traditional_ml.preprocessing import TEXT_CLEANING_CONFIG
TEXT_CLEANING_CONFIG["remove_numbers"] = True

text2 = clean_text("Product costs $99 and has 5 stars")
print(text2)  # "product costs and has stars"
```

## Dataset Information

The module uses the **Amazon Reviews 2023** dataset from Hugging Face:

- **Source**: McAuley-Lab/Amazon-Reviews-2023
- **Categories**: Health_and_Household, Electronics
- **Labels**: Binary sentiment (Positive ≥ 4 stars, Negative < 4 stars)
- **Format**: Streaming mode for memory efficiency

## Dependencies

- `pandas`: Data manipulation
- `scikit-learn`: TF-IDF vectorization and train/test split
- `datasets`: Hugging Face dataset loading
- `requests`: HTTP streaming for dataset

## Notes

- The module **reuses** your existing dataset loader (`src.dataset_loaders.amazon_reviews_2023`)
- No code duplication - single source of truth for data loading
- TF-IDF features are sparse matrices (memory efficient)
- Vectorizer must be saved for later use in production
- Text cleaning is optional (set `clean=False` to skip)

## Next Steps

After preprocessing, you can:

1. Train traditional ML models (coming soon in `models/`)
2. Evaluate model performance (coming soon in `evaluator.py`)
3. Compare different algorithms (coming soon in `trainer.py`)
4. Use command-line interface (coming soon in `cli.py`)

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'src'`, make sure you're running from the project root:

```bash
cd "/Users/vivekverma/MEGA downloads/Amazon Research"
python src/traditional_ml/preprocessing.py
```

### Memory Issues

If you run out of memory, reduce the sample limit:

```python
X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
    sample_limit=500  # Use fewer samples
)
```

### Slow TF-IDF

If TF-IDF is slow, reduce max_features:

```python
from src.traditional_ml.preprocessing import TFIDF_CONFIG
TFIDF_CONFIG["max_features"] = 5000  # Reduce from 10000
```

## License

MIT License - see LICENSE file for details.