"""
Simple Sentiment Analysis Inference Script
==========================================

This script loads your trained ELECTRA model and predicts whether a review
is Positive or Negative.

HOW TO USE:
1. Make sure you have trained your model first (run: python -m src.main text-train)
2. Run this script: python simple_inference.py
3. Enter your review text when prompted
4. See the prediction and confidence score

WHAT THIS SCRIPT DOES:
1. Loads your saved model from models/bert_finetuned/best_model/
2. Loads the tokenizer from models/bert_finetuned/tokenizer/
3. Takes your input text
4. Converts text to numbers the model can understand (tokenization)
5. Runs the model to get predictions
6. Shows you the result with confidence percentage
"""

# ============================================================================
# STEP 1: Import required libraries
# ============================================================================
# These are the tools we need to load the model and make predictions

import torch  # PyTorch - the deep learning framework
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Hugging Face Transformers library

# ============================================================================
# STEP 2: Define where your model is saved
# ============================================================================
# These paths point to where your trained model and tokenizer are stored

MODEL_PATH = "models/bert_finetuned/best_model"  # Path to your trained model
TOKENIZER_PATH = "models/bert_finetuned/tokenizer"  # Path to your saved tokenizer

# ============================================================================
# STEP 3: Load the trained model
# ============================================================================
# This loads your fine-tuned ELECTRA model that you trained earlier

print("Loading model...")  # Let user know we're loading

# Load the model from the saved directory
# AutoModelForSequenceClassification automatically loads the right model architecture
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Set the model to evaluation mode
# This is important! It tells the model we're doing prediction, not training
# It disables training-specific features like dropout
model.eval()

print("✓ Model loaded successfully!")

# ============================================================================
# STEP 4: Load the tokenizer
# ============================================================================
# The tokenizer converts text into numbers the model can understand

print("Loading tokenizer...")

# Load the tokenizer that was saved during training
# The tokenizer knows how to split text into tokens and convert to IDs
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

print("✓ Tokenizer loaded successfully!")

# ============================================================================
# STEP 5: Check if we can use GPU (optional speed boost)
# ============================================================================
# This checks if you have a CUDA GPU (NVIDIA) or MPS (Apple Silicon)
# If not, it will use CPU (slower but works everywhere)

if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
    print("✓ Using CUDA GPU (fast!)")
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon (M1/M2/M3)
    print("✓ Using Apple Silicon GPU (fast!)")
else:
    device = "cpu"  # CPU only
    print("⚠ Using CPU (slower, but will work)")

# Move the model to the selected device (GPU or CPU)
model = model.to(device)

# ============================================================================
# STEP 6: Define the prediction function
# ============================================================================
# This function takes a review text and returns the prediction

def predict_sentiment(review_text):
    """
    Predict if a review is Positive or Negative.
    
    Args:
        review_text (str): The review text to analyze
        
    Returns:
        dict: Contains the label, confidence score, and probabilities
    """
    
    # -------------------------------------------------------------------------
    # 6a: Validate input
    # -------------------------------------------------------------------------
    # Make sure the input is not empty
    if not review_text or not review_text.strip():
        raise ValueError("Please provide some text to analyze!")
    
    # -------------------------------------------------------------------------
    # 6b: Tokenize the text
    # -------------------------------------------------------------------------
    # Tokenization = converting text into numbers the model understands
    
    # The tokenizer does several things:
    # 1. Splits text into words/subwords (tokens)
    # 2. Converts tokens to IDs (numbers)
    # 3. Adds special tokens like [CLS] and [SEP]
    # 4. Pads/truncates to fixed length (128 tokens)
    # 5. Creates attention mask (tells model which tokens are real vs padding)
    
    inputs = tokenizer(
        review_text,                    # The text to tokenize
        padding="max_length",           # Pad to max length (128)
        truncation=True,                # Cut off if longer than 128 tokens
        max_length=128,                 # Maximum token length
        return_tensors="pt",            # Return PyTorch tensors (not plain lists)
    )
    
    # Move the tokenized inputs to the same device as the model (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # -------------------------------------------------------------------------
    # 6c: Run the model (make prediction)
    # -------------------------------------------------------------------------
    # torch.no_grad() tells PyTorch we don't need to calculate gradients
    # This saves memory and speeds up prediction
    with torch.no_grad():
        
        # Run the model on our tokenized input
        # The model returns logits (raw scores before softmax)
        outputs = model(**inputs)
        
        # Get the logits (raw prediction scores)
        logits = outputs.logits[0]  # [0] because we only have 1 input
        
        # Convert logits to probabilities using softmax
        # Softmax turns raw scores into percentages that sum to 1
        probabilities = torch.softmax(logits, dim=0)
    
    # -------------------------------------------------------------------------
    # 6d: Interpret the results
    # -------------------------------------------------------------------------
    
    # Get the index of the highest probability (0 or 1)
    # 0 = Negative, 1 = Positive
    predicted_label_id = torch.argmax(logits).item()
    
    # Convert index to label name
    predicted_label = "Positive" if predicted_label_id == 1 else "Negative"
    
    # Get the confidence score (probability of the predicted class)
    confidence_score = probabilities[predicted_label_id].item()
    
    # Get probabilities for both classes
    negative_prob = probabilities[0].item()
    positive_prob = probabilities[1].item()
    
    # -------------------------------------------------------------------------
    # 6e: Return results
    # -------------------------------------------------------------------------
    return {
        "text": review_text,
        "label": predicted_label,
        "label_id": predicted_label_id,
        "confidence": confidence_score,
        "probabilities": {
            "Negative": negative_prob,
            "Positive": positive_prob
        }
    }

# ============================================================================
# STEP 7: Main execution (run when script is executed)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS - PREDICTION")
    print("=" * 70)
    print("\nThis model predicts if a review is Positive or Negative.")
    print("Type your review and press Enter to see the prediction.\n")
    
    # Loop to allow multiple predictions
    while True:
        # Get user input
        user_input = input("\nEnter a review (or 'quit' to exit): ").strip()
        
        # Check if user wants to quit
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break
        
        # Check if input is empty
        if not user_input:
            print("⚠ Please enter some text!")
            continue
        
        # Make prediction
        try:
            result = predict_sentiment(user_input)
            
            # Display results
            print("\n" + "-" * 70)
            print("PREDICTION RESULT")
            print("-" * 70)
            print(f"Review: {result['text'][:100]}...")  # Show first 100 chars
            print(f"Sentiment: {result['label']}")
            print(f"Confidence: {result['confidence']:.2%}")  # Format as percentage
            print(f"\nProbabilities:")
            print(f"  Negative: {result['probabilities']['Negative']:.2%}")
            print(f"  Positive: {result['probabilities']['Positive']:.2%}")
            print("-" * 70)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")