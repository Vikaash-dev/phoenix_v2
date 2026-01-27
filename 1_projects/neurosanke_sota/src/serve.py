"""
Phoenix Protocol Model Serving API
FastAPI-based inference server for NeuroKAN models.
"""

import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

# Import project modules
from src.clinical_preprocessing import ClinicalPreprocessing
from src.clinical_postprocessing import ClinicalPostProcessing
from models.neurokan_model import create_neurokan_model
import config

# Global state
model = None
preprocessor = None
postprocessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and resources on startup."""
    global model, preprocessor, postprocessor
    
    print("Loading NeuroKAN model...")
    # In production, load weights from a specific path
    # For now, we initialize a fresh model structure or load if path exists
    model_path = os.getenv("MODEL_PATH", "results/neurokan_best.h5")
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}. Initializing empty NeuroKAN structure.")
        model = create_neurokan_model()
    
    preprocessor = ClinicalPreprocessing(
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)
    )
    
    postprocessor = ClinicalPostProcessing(
        confidence_threshold=0.8,
        use_tta=True
    )
    
    yield
    
    # Cleanup
    print("Shutting down...")

app = FastAPI(title="Phoenix Protocol API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run inference on an uploaded MRI image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        # Preprocess
        processed_img = preprocessor.preprocess(image_np)
        
        # Inference (with TTA)
        mean_pred, variance = postprocessor.test_time_augmentation(model, processed_img)
        
        # Post-process
        uncertainty = postprocessor.compute_uncertainty(mean_pred[np.newaxis], variance[np.newaxis])[0]
        
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'] # Standard BraTS/Sartaj classes
        
        # Generate report
        report = postprocessor.generate_clinical_report(
            mean_pred,
            uncertainty,
            class_names
        )
        
        return JSONResponse(content=report)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=False)
