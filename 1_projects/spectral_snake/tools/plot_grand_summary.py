import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_grand_summary():
    with open("research_artifacts/GRAND_BENCHMARK.json", "r") as f:
        data = json.load(f)
    
    # Filter successful runs
    models = [d for d in data if d["status"] == "Success"]
    
    names = [m["name"] for m in models]
    params = [m["params"] / 1e6 for m in models] # Million
    acc = [m["accuracy"] for m in models]
    robustness = [m["robustness"] for m in models]
    
    # 1. Pareto Frontier (Efficiency vs Accuracy)
    plt.figure(figsize=(10, 6))
    plt.scatter(params, acc, c=robustness, cmap='viridis', s=200, alpha=0.8)
    plt.colorbar(label='Robustness Score')
    
    for i, name in enumerate(names):
        plt.text(params[i], acc[i]+0.2, name, ha='center', fontsize=9)
        
    plt.title("The 'AI Scientist' Discovery Journey: Efficiency vs Accuracy")
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig("research_artifacts/grand_pareto.png")
    plt.close()
    
    # 2. Radar Chart (Multi-Metric)
    # Normalize metrics
    def normalize(vals):
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: return [1.0]*len(vals)
        return [(v - min_v)/(max_v - min_v) for v in vals]
        
    n_metrics = 3
    angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    for i, model in enumerate(models):
        # Metrics: Accuracy, Robustness, Efficiency (1/Params)
        # Normalize roughly manually for visualization
        val_acc = model["accuracy"] / 100
        val_rob = model["robustness"] / 100
        val_eff = 1.0 - (model["params"] / 2.5e6) # Rough normalization
        
        values = [val_acc, val_rob, val_eff]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, label=model["name"])
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'Robustness', 'Efficiency'])
    ax.set_title("Architectural Trade-offs")
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig("research_artifacts/grand_radar.png")
    plt.close()
    
    print("Grand Summary plots generated.")

if __name__ == "__main__":
    plot_grand_summary()
