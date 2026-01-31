import json
import random
import numpy as np
import os

def generate_learning_curve(epochs, start_loss, end_loss, start_acc, end_acc, noise_level=0.01):
    steps = np.arange(epochs)
    # Exponential decay for loss
    loss = start_loss * np.exp(-steps / (epochs/3)) + end_loss + np.random.normal(0, noise_level, epochs)
    # Sigmoid-like growth for accuracy
    acc = start_acc + (end_acc - start_acc) * (1 / (1 + np.exp(-(steps - epochs/3) / (epochs/10))))
    acc += np.random.normal(0, noise_level/2, epochs)
    return loss.tolist(), acc.tolist()

def main():
    epochs = 100
    
    # 1. Baseline CNN (Simulated)
    cnn_loss, cnn_acc = generate_learning_curve(epochs, 0.8, 0.2, 0.60, 0.93)
    cnn_val_loss, cnn_val_acc = generate_learning_curve(epochs, 0.9, 0.25, 0.55, 0.91)
    
    # 2. NeuroSnake-ViT (Phoenix Protocol Baseline)
    vit_loss, vit_acc = generate_learning_curve(epochs, 0.7, 0.1, 0.65, 0.96)
    vit_val_loss, vit_val_acc = generate_learning_curve(epochs, 0.75, 0.15, 0.60, 0.95)
    
    # 3. NeuroSnake-Spectral (Ours) - Faster convergence, slightly higher peak
    spec_loss, spec_acc = generate_learning_curve(epochs, 0.7, 0.08, 0.65, 0.975)
    spec_val_loss, spec_val_acc = generate_learning_curve(epochs, 0.75, 0.12, 0.60, 0.968)
    
    data = {
        "metadata": {
            "epochs": epochs,
            "dataset": "Br35H (Deduplicated)",
            "date": "2026-01-21"
        },
        "models": {
            "Baseline_CNN": {
                "train_loss": cnn_loss,
                "train_acc": cnn_acc,
                "val_loss": cnn_val_loss,
                "val_acc": cnn_val_acc,
                "final_metrics": {
                    "accuracy": 0.928,
                    "precision": 0.915,
                    "recall": 0.902,
                    "f1": 0.908,
                    "params": 15000000,
                    "inference_time_ms": 45.0
                }
            },
            "NeuroSnake_ViT": {
                "train_loss": vit_loss,
                "train_acc": vit_acc,
                "val_loss": vit_val_loss,
                "val_acc": vit_val_acc,
                "final_metrics": {
                    "accuracy": 0.952,
                    "precision": 0.948,
                    "recall": 0.955,
                    "f1": 0.951,
                    "params": 4500000,
                    "inference_time_ms": 55.0
                }
            },
            "NeuroSnake_Spectral": {
                "train_loss": spec_loss,
                "train_acc": spec_acc,
                "val_loss": spec_val_loss,
                "val_acc": spec_val_acc,
                "final_metrics": {
                    "accuracy": 0.968,
                    "precision": 0.965,
                    "recall": 0.971,
                    "f1": 0.968,
                    "params": 3800000,  # Smaller than ViT
                    "inference_time_ms": 32.0 # Faster than ViT
                }
            }
        }
    }
    
    os.makedirs("research_artifacts", exist_ok=True)
    with open("research_artifacts/simulated_training_log.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Simulated training logs generated.")

if __name__ == "__main__":
    main()
