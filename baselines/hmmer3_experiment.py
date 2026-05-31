import pandas as pd
import numpy as np
import os
import subprocess
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost

# Load Data
df = pd.read_csv('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv')
with open('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_protgpt2_embedding.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
df['embedding'] = loaded_data

# Metrics
acc_list = []
sensitivity_list = []
specificity_list = []
Class_list = []
AUC_list = []
f1_list = []
Precision_list = []

# Set HMMER path
hmmer_bin = "/home/f087s426/hmmer-3.4/install/bin"

for i in df.Class.unique()[:1]:
    print('Processing Class:', i)

    # Binary classification (one-vs-rest)
    class_df = df[df['Class'] == i].copy()
    class_df['Class'] = 1
    miscellaneous_df = df[df['Class'] != i].copy()
    miscellaneous_df['Class'] = 0
    final_df = pd.concat([class_df, miscellaneous_df]).sample(frac=1).dropna()

    X_train, X_test, y_train, y_test = train_test_split(final_df, final_df['Class'], test_size=0.2,
                                                        stratify=final_df['Class'])

    #print(X_test['Class'])

    # Create working dir
    folder_path = f"/home/f087s426/GPCR_HMM/GPCR_{i}"
    os.makedirs(folder_path, exist_ok=True)

    train_fasta = os.path.join(folder_path, f"GPCR_{i}_train.fasta")
    test_fasta = os.path.join(folder_path, f"GPCR_{i}_test.fasta")
    msa_file = os.path.join(folder_path, f"GPCR_{i}_train.sto")
    hmm_file = os.path.join(folder_path, f"GPCR_{i}.hmm")
    hmm_out_file = os.path.join(folder_path, f"GPCR_{i}_hmmsearch.out")

    # Write training sequences in FASTA
    with open(train_fasta, "w") as f:
        #for j, row in X_train[X_train['Class'] == 1].iterrows():
        for j, row in X_train.iterrows():

            f.write(f">{row['Class']}_seq{j}\n{row['fragmented_sequence']}\n")

    # Align sequences using MAFFT (optional, but recommended for HMMs)
    aligned_fasta = os.path.join(folder_path, f"GPCR_{i}_aligned.fasta")
    mafft_cmd = ["mafft", "--anysymbol", train_fasta]
    with open(aligned_fasta, "w") as f:
        subprocess.run(mafft_cmd, stdout=f)

    # Convert to Stockholm format using `esl-reformat`
    esl_reformat = os.path.join(hmmer_bin, "esl-reformat")
    with open(msa_file, "w") as f:
        subprocess.run([esl_reformat, "stockholm", aligned_fasta], stdout=f)

    # Build HMM profile using `hmmbuild`
    hmmbuild = os.path.join(hmmer_bin, "hmmbuild")
    subprocess.run([hmmbuild, hmm_file, msa_file], check=True)

    # Write test sequences in FASTA
    with open(test_fasta, "w") as f:
        for j, row in X_test.iterrows():
            f.write(f">{row['Class']}_seq{j}\n{row['filtered_sequence']}\n")

    # Run hmmsearch
    hmmsearch = os.path.join(hmmer_bin, "hmmsearch")
    with open(hmm_out_file, "w") as f:
        subprocess.run([hmmsearch, "--tblout", "/dev/stdout","-E", "100000", hmm_file, test_fasta], stdout=f)

    print(f"HMMER finished for class {i}, output saved in {hmm_out_file}")
