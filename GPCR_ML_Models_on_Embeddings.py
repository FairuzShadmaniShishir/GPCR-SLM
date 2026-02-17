import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, make_scorer, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import os
import warnings

warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load data
df = pd.read_csv('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv')

# Load all embeddings - UPDATE THESE PATHS WITH YOUR ACTUAL EMBEDDING FILES
embeddings = {

    'ProtGPT2': '/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_protgpt2_embedding.pkl',
    'ESM2': '/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_esm2_embedding.pkl',  # UPDATE THIS PATH
    'Protbert': '/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_protbert_embedding.pkl', # UPDATE THIS PATH
    'ESM2-distilled': '/home/f087s426/PycharmProjects/Protein_Family_Prediction/student_embeddings.pkl' # UPDATE THIS PATH

}

# Load embedding data
embedding_data = {}
for emb_name, emb_path in embeddings.items():
    try:
        with open(emb_path, 'rb') as file:
            embedding_data[emb_name] = pickle.load(file)
        print(f"✓ Loaded {emb_name} embedding: {len(embedding_data[emb_name])} samples")
    except FileNotFoundError:
        print(f"✗ Warning: {emb_name} embedding file not found at {emb_path}")
        print(f"  Skipping {emb_name}...")
    except Exception as e:
        print(f"✗ Error loading {emb_name}: {str(e)}")

# Filter to only use successfully loaded embeddings
embedding_data = {k: v for k, v in embedding_data.items() if v is not None}

if not embedding_data:
    print("No embeddings loaded successfully. Please check your file paths.")
    exit()

print(f"\nUsing {len(embedding_data)} embeddings: {list(embedding_data.keys())}")

# Define models
models = {
    'XGBoost': xgboost.XGBClassifier(random_state=42, eval_metric='logloss'),
    #'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    #'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    #'SVM': SVC(probability=True, random_state=42),
    #'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    #'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Initialize results storage
all_results = []
class_results = {}
embedding_comparison = {}

# Create output directory
output_dir = 'ml_results_multi_embedding'
os.makedirs(output_dir, exist_ok=True)

print(f"\nStarting multi-embedding, multi-model classification with 5-fold cross-validation...")
print("=" * 80)

# Process each embedding
for embedding_name, embedding_values in embedding_data.items():
    print(f"\n{'=' * 20} PROCESSING {embedding_name.upper()} EMBEDDING {'=' * 20}")

    # Add embedding to dataframe
    df_current = df.copy()
    df_current['embedding'] = list(embedding_values)

    embedding_comparison[embedding_name] = {}

    # Process each class
    for class_idx, class_name in enumerate(df_current.Class.unique()):
        print(f'\nClass: {class_name} | Embedding: {embedding_name}')
        print("-" * 60)

        # Prepare binary classification data
        class_df = df_current.where(df_current['Class'] == class_name).copy()
        miscellaneous_df = df_current.where(df_current['Class'] != class_name).copy()

        class_df['Class'] = 1
        miscellaneous_df['Class'] = 0

        final_df = pd.concat([class_df, miscellaneous_df])
        final_df = final_df.sample(frac=1).dropna().reset_index(drop=True)

        # Prepare features and target
        X = np.array(list(final_df['embedding'].values))
        y = final_df['Class'].values

        print(f"Dataset size: {len(final_df)} samples")
        print(f"Class distribution: {np.bincount(y)}")

        # Initialize storage for this class-embedding combination
        if class_name not in class_results:
            class_results[class_name] = {}
        if embedding_name not in class_results[class_name]:
            class_results[class_name][embedding_name] = {}

        # Test each model
        for model_name, model in models.items():
            print(f"\n  Testing {model_name}...")

            try:
                # 5-fold stratified cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # Perform cross-validation
                start_time = time.time()

                # Manual cross-validation for better AUC handling
                auc_scores = []
                other_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

                for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                    # Create fresh model instance for each fold
                    if model_name == 'XGBoost':
                        fold_model = xgboost.XGBClassifier(random_state=42, eval_metric='logloss')
                    elif model_name == 'SVM':
                        fold_model = SVC(probability=True, random_state=42)
                    elif model_name == 'Logistic Regression':
                        fold_model = LogisticRegression(random_state=42, max_iter=1000)
                    elif model_name == 'KNN':
                        fold_model = KNeighborsClassifier(n_neighbors=5)

                    # Fit model
                    fold_model.fit(X_train_fold, y_train_fold)

                    # Get predictions
                    y_pred = fold_model.predict(X_val_fold)

                    # Calculate standard metrics
                    other_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                    other_scores['precision'].append(
                        precision_score(y_val_fold, y_pred, average='weighted', zero_division=0))
                    other_scores['recall'].append(recall_score(y_val_fold, y_pred, average='weighted', zero_division=0))
                    other_scores['f1_score'].append(f1_score(y_val_fold, y_pred, average='weighted', zero_division=0))

                    # Calculate AUC if possible
                    try:
                        if hasattr(fold_model, 'predict_proba'):
                            y_proba = fold_model.predict_proba(X_val_fold)[:, 1]
                        elif hasattr(fold_model, 'decision_function'):
                            y_proba = fold_model.decision_function(X_val_fold)
                        else:
                            y_proba = y_pred

                        auc_score = roc_auc_score(y_val_fold, y_proba)
                        auc_scores.append(auc_score)
                    except Exception as auc_error:
                        auc_scores.append(0.5)  # Random classifier fallback

                end_time = time.time()

                # Calculate mean and std for each metric
                results = {
                    'Class': class_name,
                    'Model': model_name,
                    'Embedding': embedding_name,
                    'Accuracy_Mean': np.mean(other_scores['accuracy']),
                    'Accuracy_Std': np.std(other_scores['accuracy']),
                    'Precision_Mean': np.mean(other_scores['precision']),
                    'Precision_Std': np.std(other_scores['precision']),
                    'Recall_Mean': np.mean(other_scores['recall']),
                    'Recall_Std': np.std(other_scores['recall']),
                    'F1_Mean': np.mean(other_scores['f1_score']),
                    'F1_Std': np.std(other_scores['f1_score']),
                    'AUC_Mean': np.mean(auc_scores) if auc_scores else 0.5,
                    'AUC_Std': np.std(auc_scores) if auc_scores else 0.0,
                    'Training_Time': end_time - start_time
                }

                all_results.append(results)
                class_results[class_name][embedding_name][model_name] = results

                print(f"    Accuracy: {results['Accuracy_Mean']:.4f} ± {results['Accuracy_Std']:.4f}")
                print(f"    Precision: {results['Precision_Mean']:.4f} ± {results['Precision_Std']:.4f}")
                print(f"    Recall: {results['Recall_Mean']:.4f} ± {results['Recall_Std']:.4f}")
                print(f"    F1-Score: {results['F1_Mean']:.4f} ± {results['F1_Std']:.4f}")
                print(f"    AUC: {results['AUC_Mean']:.4f} ± {results['AUC_Std']:.4f}")
                print(f"    Training Time: {results['Training_Time']:.2f}s")

            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                continue

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Save detailed results
results_df.to_csv(f'{output_dir}/detailed_results_all_embeddings.csv', index=False)
print(f"\nDetailed results saved to {output_dir}/detailed_results_all_embeddings.csv")

# Create embedding comparison summary
embedding_summary = results_df.groupby(['Embedding', 'Model']).agg({
    'Accuracy_Mean': 'mean',
    'F1_Mean': 'mean',
    'AUC_Mean': 'mean',
    'Training_Time': 'mean'
}).round(4)

embedding_summary.to_csv(f'{output_dir}/embedding_comparison_summary.csv')

# Print embedding comparison
print("\n" + "=" * 80)
print("EMBEDDING COMPARISON SUMMARY")
print("=" * 80)
print(embedding_summary)

# Create model comparison by embedding
model_summary = results_df.groupby(['Model', 'Embedding']).agg({
    'Accuracy_Mean': 'mean',
    'F1_Mean': 'mean',
    'AUC_Mean': 'mean',
    'Training_Time': 'mean'
}).round(4)

model_summary.to_csv(f'{output_dir}/model_embedding_summary.csv')

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('Multi-Embedding Model Performance Comparison', fontsize=16, fontweight='bold')

# 1. Accuracy by Embedding
sns.boxplot(data=results_df, x='Embedding', y='Accuracy_Mean', hue='Model', ax=axes[0, 0])
axes[0, 0].set_title('Accuracy by Embedding')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. F1-Score by Embedding
sns.boxplot(data=results_df, x='Embedding', y='F1_Mean', hue='Model', ax=axes[0, 1])
axes[0, 1].set_title('F1-Score by Embedding')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. AUC by Embedding
sns.boxplot(data=results_df, x='Embedding', y='AUC_Mean', hue='Model', ax=axes[0, 2])
axes[0, 2].set_title('AUC by Embedding')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 4. Embedding performance heatmap (AUC)
pivot_emb_auc = results_df.pivot_table(index='Embedding', columns='Model', values='AUC_Mean', aggfunc='mean')
sns.heatmap(pivot_emb_auc, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 0])
axes[1, 0].set_title('Average AUC: Embeddings vs Models')

# 5. Training time by embedding
sns.boxplot(data=results_df, x='Embedding', y='Training_Time', ax=axes[1, 1])
axes[1, 1].set_title('Training Time by Embedding')
axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Best embedding per model
best_embedding_per_model = results_df.loc[results_df.groupby('Model')['AUC_Mean'].idxmax()]
sns.barplot(data=best_embedding_per_model, x='Model', y='AUC_Mean', hue='Embedding', ax=axes[1, 2])
axes[1, 2].set_title('Best Embedding per Model (AUC)')
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f'{output_dir}/multi_embedding_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create embedding-specific heatmaps
fig, axes = plt.subplots(1, len(embedding_data), figsize=(6 * len(embedding_data), 8))
if len(embedding_data) == 1:
    axes = [axes]

for idx, embedding_name in enumerate(embedding_data.keys()):
    emb_data = results_df[results_df['Embedding'] == embedding_name]
    pivot_data = emb_data.pivot(index='Class', columns='Model', values='AUC_Mean')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[idx])
    axes[idx].set_title(f'{embedding_name} - AUC by Class')

plt.tight_layout()
plt.savefig(f'{output_dir}/embedding_class_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# Find best embedding-model combinations
best_combinations = results_df.loc[results_df.groupby('Class')['AUC_Mean'].idxmax()]
best_combinations_summary = best_combinations[['Class', 'Model', 'Embedding', 'Accuracy_Mean', 'F1_Mean', 'AUC_Mean']]
best_combinations_summary.to_csv(f'{output_dir}/best_embedding_model_combinations.csv', index=False)

print("\n" + "=" * 80)
print("BEST EMBEDDING-MODEL COMBINATIONS PER CLASS (Based on AUC)")
print("=" * 80)
print(best_combinations_summary.to_string(index=False))

# Overall embedding ranking
embedding_ranking = results_df.groupby('Embedding')['AUC_Mean'].mean().sort_values(ascending=False)
print("\n" + "=" * 80)
print("EMBEDDING RANKING (Average AUC across all classes and models)")
print("=" * 80)
for idx, (embedding, avg_auc) in enumerate(embedding_ranking.items(), 1):
    print(f"{idx}. {embedding}: {avg_auc:.4f}")

# Save all results as pickle
with open(f'{output_dir}/all_results_multi_embedding.pkl', 'wb') as f:
    pickle.dump({
        'results_df': results_df,
        'class_results': class_results,
        'embedding_summary': embedding_summary,
        'model_summary': model_summary,
        'best_combinations': best_combinations_summary,
        'embedding_ranking': embedding_ranking
    }, f)

print(f"\n" + "=" * 80)
print("MULTI-EMBEDDING ANALYSIS COMPLETE!")
print("=" * 80)
print(f"All results saved in '{output_dir}' directory:")
print(f"- detailed_results_all_embeddings.csv: Complete results")
print(f"- embedding_comparison_summary.csv: Embedding performance summary")
print(f"- model_embedding_summary.csv: Model performance by embedding")
print(f"- best_embedding_model_combinations.csv: Best combinations per class")
print(f"- all_results_multi_embedding.pkl: Complete results in pickle format")
print(f"- Various visualization files (.png)")
print("=" * 80)