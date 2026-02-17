import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv')

# Load embeddings
with open('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_protbert_embedding.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

df['embedding'] = loaded_data

# Define classifiers
classifiers = {
    "XGBoost": XGBClassifier(
        tree_method='gpu_hist',
        gpu_id=0,
        predictor='gpu_predictor',
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    #"RandomForest": RandomForestClassifier(n_estimators=100),
    #"GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True)
}

# Create results directory
results_dir = "./classical_Ml_results"
os.makedirs(results_dir, exist_ok=True)

# Prepare to store metrics
results = {clf_name: {"accuracy": [], "sensitivity": [], "specificity": [], "auc": []}
           for clf_name in classifiers.keys()}

protein_classes = df['Class'].unique()

# Loop over protein classes
for protein_class in protein_classes:
    print(f"Processing Class: {protein_class}")

    # One-vs-rest dataset
    class_df = df[df['Class'] == protein_class].copy()
    class_df['Class'] = 1
    rest_df = df[df['Class'] != protein_class].copy()
    rest_df['Class'] = 0
    final_df = pd.concat([class_df, rest_df]).sample(frac=1).dropna()

    X = list(final_df['embedding'].values)
    y = final_df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc = roc_auc_score(y_test, y_prob)

        results[clf_name]["accuracy"].append(acc)
        results[clf_name]["sensitivity"].append(sensitivity)
        results[clf_name]["specificity"].append(specificity)
        results[clf_name]["auc"].append(auc)

# Plot comparison curves
metrics = ["accuracy", "sensitivity", "specificity", "auc"]

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for clf_name in classifiers.keys():
        plt.plot(protein_classes, results[clf_name][metric], marker='o', label=clf_name)
    plt.title(f"{metric.capitalize()} Comparison Across Classifiers")
    plt.xlabel("Protein Class")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{metric}_comparison.png"))
    plt.close()

# Save metrics to CSV
for clf_name in classifiers.keys():
    metrics_df = pd.DataFrame(results[clf_name], index=protein_classes)
    metrics_df.index.name = "Protein_Class"
    metrics_df.to_csv(os.path.join(results_dir, f"{clf_name}_metrics.csv"))

print(f"All comparison curves and metrics saved in '{results_dir}/'")
