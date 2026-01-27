import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results():
    with open("research_artifacts/iteration_2/simulated_training_log.json", "r") as f:
        data = json.load(f)
    
    epochs = range(1, data["metadata"]["epochs"] + 1)
    models = data["models"]
    
    # 1. Accuracy Plot
    plt.figure(figsize=(10, 6))
    for name, metrics in models.items():
        plt.plot(epochs, metrics["val_acc"], label=f"{name} (Val)", linewidth=2)
        
    plt.title("NeuroSnake-KAN vs Spectral: Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research_artifacts/iteration_2/accuracy_comparison.png")
    plt.close()
    
    # 2. Efficiency Plot
    plt.figure(figsize=(8, 6))
    for name, metrics in models.items():
        params = metrics["final_metrics"]["params"] / 1e6 # Millions
        acc = metrics["final_metrics"]["accuracy"] * 100
        inf_time = metrics["final_metrics"]["inference_time_ms"]
        
        plt.scatter(params, acc, s=inf_time*10, label=name, alpha=0.7)
        plt.text(params, acc+0.05, f"{name}\n({params:.2f}M)", ha='center')
        
    plt.title("KAN: Parameter Efficiency Breakthrough")
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig("research_artifacts/iteration_2/efficiency_tradeoff.png")
    plt.close()
    
    print("KAN plots generated.")

if __name__ == "__main__":
    plot_results()
