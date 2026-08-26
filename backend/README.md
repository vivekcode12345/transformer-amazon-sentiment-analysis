# Amazon Review Sentiment Analysis Backend

A production-ready ASGI backend API built with **FastAPI** to serve sentiment analysis predictions using a fine-tuned ELECTRA/BERT transformer model.

---

## ⚙️ Requirements

- **Python**: 3.8+ (Tested on Python 3.14.0)
- **Framework**: FastAPI
- **Server**: Uvicorn

---

## 🚀 Getting Started

### 1. Set Up Environment

From the `backend/` directory, create a virtual environment and install dependencies:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Development Server

Start the FastAPI application using `uvicorn`:

```bash
# Run server from the backend/ directory
uvicorn app.main:app --reload --port 8000
```

The API will be available at: [http://localhost:8000](http://localhost:8000)
- Interactive API Documentation (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative API Documentation (ReDoc): [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🛣️ API Endpoints

### 🩺 Health Check
- **Endpoint**: `GET /health`
- **Description**: Verification endpoint to ensure the API server is up and responsive.
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```
