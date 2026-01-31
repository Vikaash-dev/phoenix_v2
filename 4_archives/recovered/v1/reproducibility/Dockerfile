FROM tensorflow/tensorflow:2.10.0-gpu

# Set working directory
WORKDIR /app

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV TF_DETERMINISTIC_OPS=1
ENV TF_CUDNN_DETERMINISTIC=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/train data/validation data/test \
    research_results/figures research_results/sota_comparison research_results/ablation_studies

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command
CMD ["python", "run_research_experiments.py"]
