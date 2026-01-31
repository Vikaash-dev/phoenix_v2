import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results():
    with open("research_artifacts/iteration_3/simulated_ttt_log.json", "r") as f:
        data = json.load(f)
    
    models = data["models"]
    
    # Adaptation Plot
    plt.figure(figsize=(10, 6))
    
    steps = models["TTT_KAN"]["inference_steps"]
    acc_ttt = models["TTT_KAN"]["accuracy"]
    acc_base = models["NeuroSnake_KAN_Standard"]["accuracy"]
    
    plt.plot(steps, [a*100 for a in acc_ttt], 'o-', label="TTT-KAN (Adaptive)", linewidth=3, color='crimson')
    plt.plot(steps, [a*100 for a in acc_base], '--', label="Standard KAN (Static)", linewidth=2, color='gray')
    
    plt.title("Zero-Shot Adaptation: Test-Time Training steps")
    plt.xlabel("Gradient Steps on Test Sample")
    plt.ylabel("Accuracy on OOD Data (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate improvement
    improvement = (acc_ttt[-1] - acc_base[-1]) * 100
    plt.arrow(steps[-1], acc_base[-1]*100, 0, improvement-1, head_width=0.2, head_length=1, fc='k', ec='k')
    plt.text(steps[-1]-1, (acc_base[-1] + acc_ttt[-1])*50, f"+{improvement:.1f}%", fontsize=12)
    
    plt.savefig("research_artifacts/iteration_3/ttt_adaptation.png")
    plt.close()
    
    print("TTT plots generated.")

if __name__ == "__main__":
    plot_results()
