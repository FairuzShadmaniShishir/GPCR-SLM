import pandas as pd
import numpy as np
import os
import subprocess
import time
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv')


# Set HMMER path
hmmer_bin = "/home/f087s426/hmmer-3.4/install/bin"

for i in df.Class.unique():
    print('Processing Class:', i)

    # Binary classification (one-vs-rest)
    class_df = df[df['Class'] == i].copy()
    class_df['Class'] = 1
    miscellaneous_df = df[df['Class'] != i].copy()
    miscellaneous_df['Class'] = 0
    final_df = pd.concat([class_df, miscellaneous_df]).sample(frac=1).dropna()

    X_train, X_test, y_train, y_test = train_test_split(final_df, final_df['Class'], test_size=0.2,
                                                        stratify=final_df['Class'])

    # Create working dir
    folder_path = f"/home/f087s426/GPCR_PHMMER/GPCR_{i}"
    os.makedirs(folder_path, exist_ok=True)

    train_fasta = os.path.join(folder_path, f"GPCR_{i}_train.fasta")
    test_fasta = os.path.join(folder_path, f"GPCR_{i}_test.fasta")
    phmmer_out_file = os.path.join(folder_path, f"GPCR_{i}_phmmer.out")

    # Write training sequences in FASTA (this becomes the target database)
    with open(train_fasta, "w") as f:
        for j, row in X_train.iterrows():
            f.write(f">{row['Class']}_seq{j}\n{row['fragmented_sequence']}\n")

    # Write test sequences in FASTA (these become the query sequences)
    with open(test_fasta, "w") as f:
        for j, row in X_test.iterrows():
            if row['Class'] == 1:
                f.write(f">{row['Class']}_seq{j}\n{row['filtered_sequence']}\n")

    # Run phmmer: test sequences (queries) against training database (targets)
    phmmer = os.path.join(hmmer_bin, "phmmer")
    start_time = time.time()
    with open(phmmer_out_file, "w") as f:
        subprocess.run([phmmer, "--tblout", "/dev/stdout", "-E", "10","--noali","--notextw", test_fasta, train_fasta], stdout=f,stderr=subprocess.DEVNULL)

    print(f"PHMMER finished for class {i}, output saved in {phmmer_out_file}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"PHMMER finished for class {i} in {elapsed_time:.2f} seconds, output saved in {phmmer_out_file}")