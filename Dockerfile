# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (leverage Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
