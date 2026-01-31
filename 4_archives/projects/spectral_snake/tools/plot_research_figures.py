import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results():
    with open("research_artifacts/simulated_training_log.json", "r") as f:
        data = json.load(f)
    
    epochs = range(1, data["metadata"]["epochs"] + 1)
    models = data["models"]
    
    # 1. Accuracy Plot
    plt.figure(figsize=(10, 6))
    for name, metrics in models.items():
        plt.plot(epochs, metrics["val_acc"], label=f"{name} (Val)", linewidth=2)
        # plt.plot(epochs, metrics["train_acc"], linestyle='--', alpha=0.5)
        
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research_artifacts/accuracy_comparison.png")
    plt.close()
    
    # 2. Loss Plot
    plt.figure(figsize=(10, 6))
    for name, metrics in models.items():
        plt.plot(epochs, metrics["val_loss"], label=f"{name} (Val)", linewidth=2)
        
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research_artifacts/loss_comparison.png")
    plt.close()
    
    # 3. Efficiency Plot (Scatter)
    plt.figure(figsize=(8, 6))
    for name, metrics in models.items():
        params = metrics["final_metrics"]["params"] / 1e6 # Millions
        acc = metrics["final_metrics"]["accuracy"] * 100
        inf_time = metrics["final_metrics"]["inference_time_ms"]
        
        plt.scatter(params, acc, s=inf_time*10, label=name, alpha=0.7)
        plt.text(params, acc+0.1, f"{name}\n({inf_time}ms)", ha='center')
        
    plt.title("Efficiency vs Accuracy Trade-off")
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig("research_artifacts/efficiency_tradeoff.png")
    plt.close()
    
    print("Plots generated in research_artifacts/")

if __name__ == "__main__":
    plot_results()
