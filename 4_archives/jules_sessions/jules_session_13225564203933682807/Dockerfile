# Dockerfile for Phoenix Protocol (NeuroKAN)
# Production-ready image for training and inference

# Base image with TensorFlow and GPU support
FROM tensorflow/tensorflow:2.15.0-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user for security
RUN useradd -m phoenix && chown -R phoenix:phoenix /app
USER phoenix

# Expose port for API serving
EXPOSE 8000

# Default command: Run API server
CMD ["python", "src/serve.py"]
