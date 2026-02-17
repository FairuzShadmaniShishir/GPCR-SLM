import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, make_scorer, roc_auc_score,
                             roc_curve, classification_report, average_precision_score,
                             precision_recall_curve)
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
    # 'ProtGPT2': '/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_protgpt2_embedding.pkl',
    #'ESM2': '/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_esm2_embedding.pkl'  # UPDATE THIS PATH
    # 'Protbert': '/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_protbert_embedding.pkl',
    # # UPDATE THIS PATH
    'ESM2-distilled': '/home/f087s426/PycharmProjects/Protein_Family_Prediction/student_embeddings.pkl'
    # UPDATE THIS PATH
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
    # 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    # 'SVM': SVC(probability=True, random_state=42),
    # 'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    # 'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Initialize results storage
all_cv_results = []  # Cross-validation results
all_test_results = []  # Test set results
class_results = {}
embedding_comparison = {}

# Create output directory
output_dir = 'ml_results_multi_embedding_saved_model'
os.makedirs(output_dir, exist_ok=True)

print(f"\nStarting multi-embedding, multi-model classification with 80:20 split and 5-fold CV...")
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

        # 80:20 Train-Test Split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set size: {len(X_train)} samples")
        print(f"Test set size: {len(X_test)} samples")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")

        # Initialize storage for this class-embedding combination
        if class_name not in class_results:
            class_results[class_name] = {}
        if embedding_name not in class_results[class_name]:
            class_results[class_name][embedding_name] = {}

        # Test each model
        for model_name, model in models.items():
            print(f"\n  Testing {model_name}...")

            try:
                # ============= CROSS-VALIDATION ON TRAINING SET =============
                print(f"    Performing 5-fold CV on training set...")

                # 5-fold stratified cross-validation on training set
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # Manual cross-validation for better metric handling
                cv_scores = {
                    'accuracy': [], 'precision': [], 'recall': [],
                    'f1_score': [], 'auc': [], 'aupr': []
                }

                cv_start_time = time.time()

                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                    # Create fresh model instance for each fold
                    if model_name == 'XGBoost':
                        fold_model = xgboost.XGBClassifier(random_state=42, eval_metric='logloss')
                    elif model_name == 'Random Forest':
                        fold_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif model_name == 'Gradient Boosting':
                        fold_model = GradientBoostingClassifier(random_state=42)
                    elif model_name == 'SVM':
                        fold_model = SVC(probability=True, random_state=42)
                    elif model_name == 'Logistic Regression':
                        fold_model = LogisticRegression(random_state=42, max_iter=1000)
                    elif model_name == 'KNN':
                        fold_model = KNeighborsClassifier(n_neighbors=5)

                    # Fit model
                    fold_model.fit(X_train_fold, y_train_fold)



                    # Get predictions and probabilities
                    y_pred = fold_model.predict(X_val_fold)

                    # Get probabilities for AUC and AUPR
                    if hasattr(fold_model, 'predict_proba'):
                        y_proba = fold_model.predict_proba(X_val_fold)[:, 1]
                    elif hasattr(fold_model, 'decision_function'):
                        y_proba = fold_model.decision_function(X_val_fold)
                    else:
                        y_proba = y_pred

                    # Calculate all metrics
                    cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                    cv_scores['precision'].append(
                        precision_score(y_val_fold, y_pred, average='weighted', zero_division=0))
                    cv_scores['recall'].append(
                        recall_score(y_val_fold, y_pred, average='weighted', zero_division=0))
                    cv_scores['f1_score'].append(
                        f1_score(y_val_fold, y_pred, average='weighted', zero_division=0))

                    # Calculate AUC
                    try:
                        auc_score = roc_auc_score(y_val_fold, y_proba)
                        cv_scores['auc'].append(auc_score)
                    except Exception:
                        cv_scores['auc'].append(0.5)

                    # Calculate AUPR (Average Precision)
                    try:
                        aupr_score = average_precision_score(y_val_fold, y_proba)
                        cv_scores['aupr'].append(aupr_score)
                    except Exception:
                        cv_scores['aupr'].append(0.0)

                cv_end_time = time.time()

                # Calculate CV means and stds
                cv_results = {
                    'Class': class_name,
                    'Model': model_name,
                    'Embedding': embedding_name,
                    'CV_Accuracy_Mean': np.mean(cv_scores['accuracy']),
                    'CV_Accuracy_Std': np.std(cv_scores['accuracy']),
                    'CV_Precision_Mean': np.mean(cv_scores['precision']),
                    'CV_Precision_Std': np.std(cv_scores['precision']),
                    'CV_Recall_Mean': np.mean(cv_scores['recall']),
                    'CV_Recall_Std': np.std(cv_scores['recall']),
                    'CV_F1_Mean': np.mean(cv_scores['f1_score']),
                    'CV_F1_Std': np.std(cv_scores['f1_score']),
                    'CV_AUC_Mean': np.mean(cv_scores['auc']),
                    'CV_AUC_Std': np.std(cv_scores['auc']),
                    'CV_AUPR_Mean': np.mean(cv_scores['aupr']),
                    'CV_AUPR_Std': np.std(cv_scores['aupr']),
                    'CV_Training_Time': cv_end_time - cv_start_time
                }

                all_cv_results.append(cv_results)

                print(f"    CV Results:")
                print(f"      Accuracy: {cv_results['CV_Accuracy_Mean']:.4f} ± {cv_results['CV_Accuracy_Std']:.4f}")
                print(f"      Precision: {cv_results['CV_Precision_Mean']:.4f} ± {cv_results['CV_Precision_Std']:.4f}")
                print(f"      Recall: {cv_results['CV_Recall_Mean']:.4f} ± {cv_results['CV_Recall_Std']:.4f}")
                print(f"      F1-Score: {cv_results['CV_F1_Mean']:.4f} ± {cv_results['CV_F1_Std']:.4f}")
                print(f"      AUC: {cv_results['CV_AUC_Mean']:.4f} ± {cv_results['CV_AUC_Std']:.4f}")
                print(f"      AUPR: {cv_results['CV_AUPR_Mean']:.4f} ± {cv_results['CV_AUPR_Std']:.4f}")

                # ============= FINAL MODEL TRAINING AND TEST SET EVALUATION =============
                print(f"    Training final model on full training set and evaluating on test set...")

                # Create final model
                if model_name == 'XGBoost':
                    final_model = xgboost.XGBClassifier(random_state=42, eval_metric='logloss')
                elif model_name == 'Random Forest':
                    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_name == 'Gradient Boosting':
                    final_model = GradientBoostingClassifier(random_state=42)
                elif model_name == 'SVM':
                    final_model = SVC(probability=True, random_state=42)
                elif model_name == 'Logistic Regression':
                    final_model = LogisticRegression(random_state=42, max_iter=1000)
                elif model_name == 'KNN':
                    final_model = KNeighborsClassifier(n_neighbors=5)

                # Train on full training set
                test_start_time = time.time()
                final_model.fit(X_train, y_train)

                # ===== Save Trained Model by Class Name =====
                model_save_dir = os.path.join(output_dir, "saved_models")
                os.makedirs(model_save_dir, exist_ok=True)

                # Clean and name file
                model_filename = f"{class_name}_{embedding_name}_{model_name}.pkl"
                model_filename = model_filename.replace(" ", "_").replace("/", "_")
                model_save_path = os.path.join(model_save_dir, model_filename)

                # Save model
                with open(model_save_path, 'wb') as f:
                    pickle.dump(final_model, f)

                print(f"    ✓ Saved trained model: {model_save_path}")

                # Predict on test set
                y_test_pred = final_model.predict(X_test)

                # Get probabilities for test set
                if hasattr(final_model, 'predict_proba'):
                    y_test_proba = final_model.predict_proba(X_test)[:, 1]
                elif hasattr(final_model, 'decision_function'):
                    y_test_proba = final_model.decision_function(X_test)
                else:
                    y_test_proba = y_test_pred

                test_end_time = time.time()

                # Calculate test metrics
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

                try:
                    test_auc = roc_auc_score(y_test, y_test_proba)
                except Exception:
                    test_auc = 0.5

                try:
                    test_aupr = average_precision_score(y_test, y_test_proba)
                except Exception:
                    test_aupr = 0.0

                # Store test results
                test_results = {
                    'Class': class_name,
                    'Model': model_name,
                    'Embedding': embedding_name,
                    'Test_Accuracy': test_accuracy,
                    'Test_Precision': test_precision,
                    'Test_Recall': test_recall,
                    'Test_F1': test_f1,
                    'Test_AUC': test_auc,
                    'Test_AUPR': test_aupr,
                    'Test_Training_Time': test_end_time - test_start_time,
                    'Train_Size': len(X_train),
                    'Test_Size': len(X_test)
                }

                all_test_results.append(test_results)

                print(f"    Test Results:")
                print(f"      Accuracy: {test_accuracy:.4f}")
                print(f"      Precision: {test_precision:.4f}")
                print(f"      Recall: {test_recall:.4f}")
                print(f"      F1-Score: {test_f1:.4f}")
                print(f"      AUC: {test_auc:.4f}")
                print(f"      AUPR: {test_aupr:.4f}")

                # Store combined results
                combined_results = {**cv_results, **test_results}
                class_results[class_name][embedding_name][model_name] = combined_results

            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                continue

# Convert results to DataFrames
cv_results_df = pd.DataFrame(all_cv_results)
test_results_df = pd.DataFrame(all_test_results)

# Merge CV and test results
merged_results_df = pd.merge(
    cv_results_df, test_results_df,
    on=['Class', 'Model', 'Embedding'],
    how='inner'
)

# Save detailed results
cv_results_df.to_csv(f'{output_dir}/cv_results_detailed.csv', index=False)
test_results_df.to_csv(f'{output_dir}/test_results_detailed.csv', index=False)
merged_results_df.to_csv(f'{output_dir}/merged_results_detailed.csv', index=False)

print(f"\nDetailed results saved:")
print(f"- CV results: {output_dir}/cv_results_detailed.csv")
print(f"- Test results: {output_dir}/test_results_detailed.csv")
print(f"- Merged results: {output_dir}/merged_results_detailed.csv")

# Create embedding comparison summaries
cv_embedding_summary = cv_results_df.groupby(['Embedding', 'Model']).agg({
    'CV_Accuracy_Mean': 'mean',
    'CV_F1_Mean': 'mean',
    'CV_AUC_Mean': 'mean',
    'CV_AUPR_Mean': 'mean',
    'CV_Training_Time': 'mean'
}).round(4)

test_embedding_summary = test_results_df.groupby(['Embedding', 'Model']).agg({
    'Test_Accuracy': 'mean',
    'Test_F1': 'mean',
    'Test_AUC': 'mean',
    'Test_AUPR': 'mean',
    'Test_Training_Time': 'mean'
}).round(4)

cv_embedding_summary.to_csv(f'{output_dir}/cv_embedding_summary.csv')
test_embedding_summary.to_csv(f'{output_dir}/test_embedding_summary.csv')

# Print summaries
print("\n" + "=" * 80)
print("CROSS-VALIDATION EMBEDDING COMPARISON SUMMARY")
print("=" * 80)
print(cv_embedding_summary)

print("\n" + "=" * 80)
print("TEST SET EMBEDDING COMPARISON SUMMARY")
print("=" * 80)
print(test_embedding_summary)

# Create comprehensive visualizations
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.suptitle('Multi-Embedding Model Performance: CV vs Test Results', fontsize=16, fontweight='bold')

# Row 1: CV Results
sns.boxplot(data=cv_results_df, x='Embedding', y='CV_Accuracy_Mean', hue='Model', ax=axes[0, 0])
axes[0, 0].set_title('CV Accuracy by Embedding')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.boxplot(data=cv_results_df, x='Embedding', y='CV_AUC_Mean', hue='Model', ax=axes[0, 1])
axes[0, 1].set_title('CV AUC by Embedding')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.boxplot(data=cv_results_df, x='Embedding', y='CV_AUPR_Mean', hue='Model', ax=axes[0, 2])
axes[0, 2].set_title('CV AUPR by Embedding')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.boxplot(data=cv_results_df, x='Embedding', y='CV_F1_Mean', hue='Model', ax=axes[0, 3])
axes[0, 3].set_title('CV F1-Score by Embedding')
axes[0, 3].tick_params(axis='x', rotation=45)
axes[0, 3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Row 2: Test Results
sns.boxplot(data=test_results_df, x='Embedding', y='Test_Accuracy', hue='Model', ax=axes[1, 0])
axes[1, 0].set_title('Test Accuracy by Embedding')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.boxplot(data=test_results_df, x='Embedding', y='Test_AUC', hue='Model', ax=axes[1, 1])
axes[1, 1].set_title('Test AUC by Embedding')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.boxplot(data=test_results_df, x='Embedding', y='Test_AUPR', hue='Model', ax=axes[1, 2])
axes[1, 2].set_title('Test AUPR by Embedding')
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.boxplot(data=test_results_df, x='Embedding', y='Test_F1', hue='Model', ax=axes[1, 3])
axes[1, 3].set_title('Test F1-Score by Embedding')
axes[1, 3].tick_params(axis='x', rotation=45)
axes[1, 3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Row 3: Heatmaps
# CV AUC Heatmap
pivot_cv_auc = cv_results_df.pivot_table(index='Embedding', columns='Model', values='CV_AUC_Mean', aggfunc='mean')
sns.heatmap(pivot_cv_auc, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[2, 0])
axes[2, 0].set_title('CV Average AUC: Embeddings vs Models')

# Test AUC Heatmap
pivot_test_auc = test_results_df.pivot_table(index='Embedding', columns='Model', values='Test_AUC', aggfunc='mean')
sns.heatmap(pivot_test_auc, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[2, 1])
axes[2, 1].set_title('Test Average AUC: Embeddings vs Models')

# CV AUPR Heatmap
pivot_cv_aupr = cv_results_df.pivot_table(index='Embedding', columns='Model', values='CV_AUPR_Mean', aggfunc='mean')
sns.heatmap(pivot_cv_aupr, annot=True, fmt='.3f', cmap='viridis', ax=axes[2, 2])
axes[2, 2].set_title('CV Average AUPR: Embeddings vs Models')

# Test AUPR Heatmap
pivot_test_aupr = test_results_df.pivot_table(index='Embedding', columns='Model', values='Test_AUPR', aggfunc='mean')
sns.heatmap(pivot_test_aupr, annot=True, fmt='.3f', cmap='viridis', ax=axes[2, 3])
axes[2, 3].set_title('Test Average AUPR: Embeddings vs Models')

plt.tight_layout()
plt.savefig(f'{output_dir}/comprehensive_cv_test_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Find best combinations based on test performance
best_test_auc = test_results_df.loc[test_results_df.groupby('Class')['Test_AUC'].idxmax()]
best_test_aupr = test_results_df.loc[test_results_df.groupby('Class')['Test_AUPR'].idxmax()]

best_test_auc_summary = best_test_auc[
    ['Class', 'Model', 'Embedding', 'Test_Accuracy', 'Test_F1', 'Test_AUC', 'Test_AUPR']]
best_test_aupr_summary = best_test_aupr[
    ['Class', 'Model', 'Embedding', 'Test_Accuracy', 'Test_F1', 'Test_AUC', 'Test_AUPR']]

best_test_auc_summary.to_csv(f'{output_dir}/best_test_auc_combinations.csv', index=False)
best_test_aupr_summary.to_csv(f'{output_dir}/best_test_aupr_combinations.csv', index=False)

print("\n" + "=" * 80)
print("BEST TEST COMBINATIONS PER CLASS (Based on AUC)")
print("=" * 80)
print(best_test_auc_summary.to_string(index=False))

print("\n" + "=" * 80)
print("BEST TEST COMBINATIONS PER CLASS (Based on AUPR)")
print("=" * 80)
print(best_test_aupr_summary.to_string(index=False))

# Overall embedding rankings
cv_embedding_ranking_auc = cv_results_df.groupby('Embedding')['CV_AUC_Mean'].mean().sort_values(ascending=False)
cv_embedding_ranking_aupr = cv_results_df.groupby('Embedding')['CV_AUPR_Mean'].mean().sort_values(ascending=False)
test_embedding_ranking_auc = test_results_df.groupby('Embedding')['Test_AUC'].mean().sort_values(ascending=False)
test_embedding_ranking_aupr = test_results_df.groupby('Embedding')['Test_AUPR'].mean().sort_values(ascending=False)

print("\n" + "=" * 80)
print("EMBEDDING RANKINGS")
print("=" * 80)

print("\nCV AUC Ranking:")
for idx, (embedding, avg_auc) in enumerate(cv_embedding_ranking_auc.items(), 1):
    print(f"{idx}. {embedding}: {avg_auc:.4f}")

print("\nCV AUPR Ranking:")
for idx, (embedding, avg_aupr) in enumerate(cv_embedding_ranking_aupr.items(), 1):
    print(f"{idx}. {embedding}: {avg_aupr:.4f}")

print("\nTest AUC Ranking:")
for idx, (embedding, avg_auc) in enumerate(test_embedding_ranking_auc.items(), 1):
    print(f"{idx}. {embedding}: {avg_auc:.4f}")

print("\nTest AUPR Ranking:")
for idx, (embedding, avg_aupr) in enumerate(test_embedding_ranking_aupr.items(), 1):
    print(f"{idx}. {embedding}: {avg_aupr:.4f}")

# Save all results as pickle
with open(f'{output_dir}/all_results_cv_test_split.pkl', 'wb') as f:
    pickle.dump({
        'cv_results_df': cv_results_df,
        'test_results_df': test_results_df,
        'merged_results_df': merged_results_df,
        'cv_embedding_summary': cv_embedding_summary,
        'test_embedding_summary': test_embedding_summary,
        'best_test_auc_combinations': best_test_auc_summary,
        'best_test_aupr_combinations': best_test_aupr_summary,
        'cv_embedding_ranking_auc': cv_embedding_ranking_auc,
        'cv_embedding_ranking_aupr': cv_embedding_ranking_aupr,
        'test_embedding_ranking_auc': test_embedding_ranking_auc,
        'test_embedding_ranking_aupr': test_embedding_ranking_aupr,
        'class_results': class_results
    }, f)

print(f"\n" + "=" * 80)
print("MULTI-EMBEDDING ANALYSIS WITH TRAIN-TEST SPLIT COMPLETE!")
print("=" * 80)
print(f"All results saved in '{output_dir}' directory:")
print(f"- cv_results_detailed.csv: Cross-validation results")
print(f"- test_results_detailed.csv: Test set results")
print(f"- merged_results_detailed.csv: Combined CV and test results")
print(f"- cv_embedding_summary.csv: CV performance summary")
print(f"- test_embedding_summary.csv: Test performance summary")
print(f"- best_test_auc_combinations.csv: Best AUC combinations per class")
print(f"- best_test_aupr_combinations.csv: Best AUPR combinations per class")
print(f"- all_results_cv_test_split.pkl: Complete results in pickle format")
print(f"- comprehensive_cv_test_comparison.png: Visualization")
print("=" * 80)