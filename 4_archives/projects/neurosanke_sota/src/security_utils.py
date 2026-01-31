"""
Security Utilities for Phoenix Protocol
Implements defenses against Rowhammer attacks and model tampering.
Based on SECURITY_ANALYSIS.md recommendations.
"""

import hashlib
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_model_integrity(model_path: str, expected_hash: str) -> bool:
    """
    Verify model has not been tampered with using SHA-256 hashing.
    Essential for preventing unauthorized modification of weights.
    
    Args:
        model_path: Path to the model file
        expected_hash: The trusted SHA-256 hash string
        
    Returns:
        True if integrity verified, raises SecurityError otherwise.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    actual_hash = sha256_hash.hexdigest()
    
    if actual_hash != expected_hash:
        logger.critical(f"Security Alert: Model integrity check failed for {model_path}")
        logger.critical(f"Expected: {expected_hash}")
        logger.critical(f"Actual:   {actual_hash}")
        raise ValueError("Security Violation: Model integrity check failed! File may be tampered.")
    
    logger.info(f"Model integrity verified: {model_path}")
    return True

def detect_rowhammer_anomaly(
    current_weights: list, 
    baseline_weights: list, 
    threshold: float = 0.01
) -> list:
    """
    Detect potential Rowhammer bit flips by comparing current weights against a known baseline.
    Rowhammer attacks often flip single bits, causing small but specific weight changes.
    
    Args:
        current_weights: List of numpy arrays (current model weights)
        baseline_weights: List of numpy arrays (trusted baseline weights)
        threshold: Difference threshold for anomaly detection
        
    Returns:
        List of dictionaries containing anomaly details
    """
    anomalies = []
    
    if len(current_weights) != len(baseline_weights):
        raise ValueError("Weight lists have different lengths")
    
    for layer_idx, (current, baseline) in enumerate(zip(current_weights, baseline_weights)):
        if current.shape != baseline.shape:
            continue
            
        diff = np.abs(current - baseline)
        suspicious = np.where(diff > threshold)
        
        # Check if there are any suspicious changes
        if len(suspicious[0]) > 0:
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff[suspicious]))
            
            anomaly_report = {
                'layer_index': layer_idx,
                'num_flipped_weights': len(suspicious[0]),
                'max_difference': max_diff,
                'mean_difference': mean_diff
            }
            anomalies.append(anomaly_report)
            
            logger.warning(f"Rowhammer Anomaly Detected in Layer {layer_idx}: {len(suspicious[0])} suspicious weights. Max diff: {max_diff:.6f}")
            
    if not anomalies:
        logger.info("No Rowhammer anomalies detected.")
        
    return anomalies

def compute_model_hash(model_path: str) -> str:
    """Helper to compute hash of a file for initial setup."""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
