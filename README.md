# Transformer-Based Sentiment Prediction of Amazon Reviews

This project builds an end-to-end sentiment analysis system for Amazon reviews using a transformer model, preferably BERT.

## In Simple Terms

The model reads a review like "This product is amazing" and learns to predict whether the sentiment is positive or negative.

## Project Architecture

1. Collect review data from an Amazon reviews dataset.
2. Clean the text and labels.
3. Explore the data to understand class balance and review lengths.
4. Convert labels into numbers that the model can learn.
5. Split the data into training and testing sets.
6. Tokenize the reviews using a BERT tokenizer.
7. Fine-tune a BERT-based model on the training data.
8. Evaluate the model using accuracy, precision, recall, and F1 score.
9. Use the trained model to predict sentiment for custom reviews.

## Suggested Dataset

Recommended default dataset: **Amazon Reviews Polarity**.

Why this dataset:
- It is widely used for sentiment classification research.
- It already contains positive and negative labels.
- It is large enough for transformer fine-tuning.

You can use it from Hugging Face Datasets or download a Kaggle version if your supervisor prefers local files.

## Environment Setup

The workspace is configured to use a Python virtual environment at `.venv`.

Install packages with:

```bash
"/Users/vivekverma/MEGA downloads/Amazon Research/.venv/bin/python" -m pip install -r requirements.txt
```

If you need to create the environment manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Current Starter File

Run the first working script with:

```bash
"/Users/vivekverma/MEGA downloads/Amazon Research/.venv/bin/python" src/main.py
```

This script creates a tiny demo dataset if no real dataset is present, so you can test the pipeline immediately.
