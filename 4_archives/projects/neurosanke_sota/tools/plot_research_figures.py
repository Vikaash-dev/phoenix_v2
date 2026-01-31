import json
import matplotlib.pyplot as plt
import os

def plot_metrics(json_path="results/research_simulation/metrics.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
        
    epochs = range(len(data["baseline"]["loss"]))
    
    plt.figure(figsize=(10, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data["baseline"]["loss"], label="Baseline CNN", alpha=0.7)
    plt.plot(epochs, data["neurokan"]["loss"], label="NeuroKAN (SOTA)", alpha=0.8)
    plt.plot(epochs, data["spectral"]["loss"], label="Spectral-Snake (Ours)", linewidth=2)
    plt.title("Training Loss Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, data["baseline"]["accuracy"], label="Baseline CNN", alpha=0.7)
    plt.plot(epochs, data["neurokan"]["accuracy"], label="NeuroKAN (SOTA)", alpha=0.8)
    plt.plot(epochs, data["spectral"]["accuracy"], label="Spectral-Snake (Ours)", linewidth=2)
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results/figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/figures/research_comparison.png")
    print("Plot saved to results/figures/research_comparison.png")

if __name__ == "__main__":
    plot_metrics()
