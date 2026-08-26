from pydantic import BaseModel, Field

class SentimentRequest(BaseModel):
    """Schema representing input review text for sentiment classification."""
    text: str = Field(
        ..., 
        min_length=1, 
        description="The customer review text to classify."
    )

class SentimentResponse(BaseModel):
    """Schema representing structured classification output."""
    text: str = Field(..., description="Original input review text.")
    label: str = Field(..., description="Predicted sentiment label ('Positive' or 'Negative').")
    confidence: float = Field(..., description="Probability score of the predicted label.")
    probabilities: dict = Field(..., description="Probability distribution across classes.")
