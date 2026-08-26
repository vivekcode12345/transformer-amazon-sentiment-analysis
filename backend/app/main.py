import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Amazon Review Sentiment Analysis API",
    description="Production-grade FastAPI backend serving a fine-tuned Transformer model.",
    version="1.0.0",
)

# Configure CORS Middleware for future Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to match your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server availability."""
    return {"status": "ok"}
