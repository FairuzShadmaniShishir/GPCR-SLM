import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer, accuracy_score, classification_report
import pickle
import subprocess
import time
import os

# Load data
df = pd.read_csv('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv')

with open('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_protgpt2_embedding.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

df['embedding'] = loaded_data

# Lists to store evaluation metrics
acc_list = []
sensitivity_list = []
specificity_list = []
Class_list = []
AUC_list = []
f1_list = []
Precision_list = []

for i in df.Class.unique():
    print('Class name', i)

    class_df = df.where(df['Class'] == i).dropna()
    miscellaneous_df = df.where(df['Class'] != i).dropna()

    miscellaneous_df['Class'] = 0
    class_df['Class'] = 1

    final_df = pd.concat([class_df, miscellaneous_df])
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(final_df, final_df['Class'], test_size=0.2)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    folder_path = f"/home/f087s426/GPCR_DIAMOND_v3/GPCR_{i}"
    os.makedirs(folder_path, exist_ok=True)

    train_fasta = os.path.join(folder_path, f"GPCR_{i}_train.fasta")
    test_fasta = os.path.join(folder_path, f"GPCR_{i}_test.fasta")
    db_output = os.path.join(folder_path, f"GPCR_{i}_train_db")
    output_file = os.path.join(folder_path, f"results_{i}.tab.out")

    # Write training FASTA
    with open(train_fasta, "w") as train_f:
        for j in range(len(X_train)):
            train_f.write(f'>{X_train["Class"].iloc[j]}_seq{j}\n')
            train_f.write(str(X_train['fragmented_sequence'].iloc[j]) + "\n")

    # Write testing FASTA
    with open(test_fasta, "w") as test_f:
        for j in range(len(X_test)):
            test_f.write(f'>{X_test["Class"].iloc[j]}_seq{j}\n')
            test_f.write(str(X_test['filtered_sequence'].iloc[j]) + "\n")

    diamond_exe = "/home/f087s426/diamond-linux64/diamond"  # <-- Update this path to your diamond binary

    # DIAMOND database creation
    print("Checking if DIAMOND database already exists...")
    if not os.path.isfile(db_output + ".dmnd"):
        print("Database not found. Creating DIAMOND database...")
        diamond_makedb_cmd = [
            diamond_exe,
            "makedb",
            "--in", train_fasta,
            "-d", db_output
        ]
        try:
            subprocess.run(diamond_makedb_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("DIAMOND database creation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during diamond makedb execution:\n{e.stderr.decode()}")
            exit(1)
    else:
        print("DIAMOND database already exists. Skipping creation.")

    # DIAMOND blastp search
    print("Running DIAMOND blastp search...")
    diamond_blastp_cmd = [
        diamond_exe,
        "blastp",
        "-q", test_fasta,
        "-d", db_output,
        "-o", output_file,
        "--outfmt", "6",
        "--evalue", "100000"
    ]
    try:
        start_time = time.time()
        subprocess.run(diamond_blastp_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end_time = time.time()
        print("DIAMOND blastp search completed successfully.")
        print(f"DIAMOND blastp execution time: {end_time - start_time:.2f} seconds")
        print(f"Results saved in {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during diamond blastp execution:\n{e.stderr.decode()}")
        exit(1)

#     # Train XGBoost classifier on embeddings
#     model = xgboost.XGBClassifier()
#
#     model.fit(list(X_train['embedding'].values), y_train)
#
#     # Prediction and evaluation
#     begin = time.time()
#     xgb_prob = model.predict_proba(list(X_test['embedding'].values))[:, 1]
#     end = time.time()
#
#     xgb_auc = roc_auc_score(y_test, xgb_prob)
#     print('ROC AUC=%.3f' % (xgb_auc))
#
#     y_pred_test = model.predict(list(X_test['embedding'].values)) > 0.1
#
#     print("Accuracy:", accuracy_score(y_test, y_pred_test))
#     weighted_f1 = f1_score(y_test, y_pred_test, average='weighted')
#     print(f"Weighted F1-score: {weighted_f1:.4f}")
#     precision = precision_score(y_test, y_pred_test)
#
#     acc_list.append(accuracy_score)
#     cm = confusion_matrix(y_test, y_pred_test)
#     tn, fp, fn, tp = cm.ravel()
#     sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
#     sensitivity_list.append(sensitivity)
#     specificity_list.append(specificity)
#     Class_list.append(i)
#     AUC_list.append(xgb_auc)
#     f1_list.append(weighted_f1)
#     Precision_list.append(precision)
#
# # Summary results
# result_df = pd.DataFrame({
#     'Class': Class_list,
#     'AUC': AUC_list,
#     'F1-score': f1_list,
#     'Precision': Precision_list,
#     'Sensitivity': sensitivity_list,
#     'Specificity': specificity_list
# })
#
# print(result_df)

# Uncomment to save results
# result_df.to_csv('/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/GPCR_xgboost_results.csv', index=False)
