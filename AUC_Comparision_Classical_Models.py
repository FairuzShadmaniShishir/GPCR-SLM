import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# File paths
# ==============================
csv1 = "classical_Ml_results/LogisticRegression_metrics.csv"
csv2 = "classical_Ml_results/SVM_metrics.csv"
csv3 = "classical_Ml_results/XGBoost_metrics.csv"

model_names = ["LogisticRegression", "SVM", "XGBoost"]
metrics = ["accuracy", "sensitivity", "specificity", "auc"]

# ==============================
# Load CSV files
# ==============================
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)

dfs = [df1, df2, df3]

# ==============================
# Compute mean and std
# ==============================
mean_values = []
std_values = []

for df in dfs:
    mean_values.append([df[m].mean() for m in metrics])
    std_values.append([df[m].std() for m in metrics])

mean_values = np.array(mean_values)
std_values = np.array(std_values)

# ==============================
# Print Mean ± STD
# ==============================
print("\n=== Mean ± STD Performance Metrics ===\n")

for i, model in enumerate(model_names):
    print(f"{model}:")
    for j, metric in enumerate(metrics):
        print(f"  {metric}: {mean_values[i][j]:.4f} ± {std_values[i][j]:.4f}")
    print()

# ==============================
# Plot (mean only)
# ==============================
x = np.arange(len(metrics))
width = 0.25

plt.figure()

plt.bar(x - width, mean_values[0], width, label=model_names[0])
plt.bar(x, mean_values[1], width, label=model_names[1])
plt.bar(x + width, mean_values[2], width, label=model_names[2])

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Model Performance Comparison (Mean)")
plt.legend()

plt.savefig("classical_Ml_results/model_comparison_metrics.png", dpi=300)
plt.show()
