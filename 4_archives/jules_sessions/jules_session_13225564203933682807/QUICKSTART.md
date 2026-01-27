# Quick Start Guide

## Prerequisites
- Python 3.10+
- TensorFlow 2.10+
- GPU recommended (but not required)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vikaash-dev/Ai-research-paper-and-implementation-of-brain-tumor-detection-.git
   cd Ai-research-paper-and-implementation-of-brain-tumor-detection-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Production Deployment (Docker)

To run the API server in a container:

```bash
# Build image
docker build -t phoenix-protocol:latest .

# Run container (exposing port 8000)
docker run -p 8000:8000 phoenix-protocol:latest
```

The API will be available at `http://localhost:8000`. Documentation at `http://localhost:8000/docs`.

## Training (NeuroKAN)

Train the SOTA NeuroKAN architecture with Mixed Precision:

```bash
python src/train_phoenix.py \
    --model-type neurokan \
    --mixed-precision \
    --epochs 50 \
    --data-dir ./data
```

## INT8 Quantization

Create an optimized INT8 model for edge devices:

```bash
python src/int8_quantization.py \
    --model-path results/neurokan_best.h5 \
    --output-path deploy/neurokan_int8.tflite \
    --data-dir ./data
```
