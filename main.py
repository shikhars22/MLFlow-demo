import os
import numpy as np

# --- CONFIGURATION ---
# Define the range for Alpha (from 0.01 to 0.3)
# We use np.linspace to get 5 evenly spaced values in this range
# We also adding our own custom values after consulting with ChatGPT
alphas = np.linspace(0.01, 0.3, 5)
alphas = np.unique(np.concatenate([alphas, [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]]))

# Define the range for L1 Ratio (from 0.1 to 0.5)
# We use np.linspace to get 5 evenly spaced values in this range
# We also adding our own custom values after consulting with ChatGPT
l1_ratios = np.linspace(0.1, 0.5, 5)
l1_ratios = np.unique(np.concatenate([l1_ratios, [0.01, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]]))

def run_grid_search():
    print(f"🚀 Starting Grid Search...")
    print(f"   Alphas to test: {np.round(alphas, 3)}")
    print(f"   L1 Ratios to test: {np.round(l1_ratios, 3)}")
    print("-" * 40)

    total_runs = len(alphas) * len(l1_ratios)
    count = 1

    # Nested loop to try every combination of alpha and l1_ratio
    for alpha in alphas:
        for l1 in l1_ratios:
            print(f"🏃 Run {count}/{total_runs}: Testing Alpha={alpha:.3f}, L1_Ratio={l1:.3f}")
            
            # Execute the demo.py script with the current hyperparameters
            # We use os.system to call it just like you would in the terminal
            os.system(f"python demo.py {alpha} {l1}")
            
            count += 1

    print("-" * 40)
    print(f"✅ Grid Search Complete! All {total_runs} runs are now on your DagsHub Dashboard.")

if __name__ == "__main__":
    run_grid_search()
