import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results():
    with open("research_artifacts/iteration_4/simulated_robustness_log.json", "r") as f:
        data = json.load(f)
    
    models = data["models"]
    noise = models["Liquid_Snake"]["noise_levels"]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(noise, [a*100 for a in models["Liquid_Snake"]["accuracy"]], 'o-', label="Liquid-Snake (Ours)", linewidth=3, color='teal')
    plt.plot(noise, [a*100 for a in models["NeuroSnake_KAN"]["accuracy"]], 's--', label="NeuroSnake-KAN (Baseline)", linewidth=2, color='gray')
    
    plt.title("Robustness to MRI Noise (Rician)")
    plt.xlabel("Noise Level (Sigma)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Fill gap
    plt.fill_between(noise, 
                     [a*100 for a in models["NeuroSnake_KAN"]["accuracy"]], 
                     [a*100 for a in models["Liquid_Snake"]["accuracy"]], 
                     alpha=0.1, color='teal')
    
    plt.savefig("research_artifacts/iteration_4/robustness_comparison.png")
    plt.close()
    
    print("Liquid plots generated.")

if __name__ == "__main__":
    plot_results()
