import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer, accuracy_score,classification_report
import pickle
import subprocess
import time
import os


df = pd.read_csv('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv')
#df = df.drop('Unnamed: 0', axis=1)
with open('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR_protgpt2_embedding.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

df['embedding'] = loaded_data

acc_list = []
sensitivity_list = []
specificity_list = []
Class_list = []
AUC_list = []
f1_list =[]
Precision_list=[]

for i in df.Class.unique():
    # print(i)
    print('Class name', (i))
    class_df = df.where(df['Class'] == i)
    miscellaneous_df = df.where(df['Class'] != i)
    miscellaneous_df['Class'] = 0
    class_df['Class'] = 1
    final_df = pd.concat([class_df, miscellaneous_df])
    final_df = final_df.sample(frac=1).dropna()
    X_train, X_test, y_train, y_test = train_test_split(final_df, final_df['Class'], test_size=0.2)
    print(len(X_train))
    print(len(X_test))


# (86 *5 4)/(len(X_train)*len(X_test))
    print((63504*15876)/(86*54*60))
#   3618.251162790698 No. of alignment per second in blast
#  (15876*86)/1.03s = 1325568 inference for inference
#  (15876)/6m 1.24s = 43.94 embedding for seconds in esm
# summary y=f(x1=query,x2=target)




    folder_path = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/GPCR_{}".format(i)
    os.makedirs(folder_path, exist_ok=True)  # Create if not exists

    train_fasta = os.path.join(folder_path, "GPCR_{}_train.fasta".format(i))
    test_fasta = os.path.join(folder_path, "GPCR_{}_test.fasta".format(i))
    db_output = os.path.join(folder_path, "GPCR_{}_train_db".format(i))
    output_file = os.path.join(folder_path, "results_{}.tab.out".format(i))

    train_f = open(train_fasta, "w+")
    for j in range(len(X_train)):
        train_f.write('>')
        # print(list(class_df['family_accession'])[j])
        train_f.write(str((X_train['Class'].iloc[j])) + '_''seq' + str(j))
        train_f.write("\n")
        train_f.write(str(X_train['fragmented_sequence'].iloc[j]))
        train_f.write("\n")
    train_f.close()

    test_f = open(test_fasta, "w+")
    for j in range(len(X_test)):
        test_f.write('>')
        # print(list(class_df['family_accession'])[j])
        test_f.write(str((X_test['Class'].iloc[j])) + '_''seq' + str(j))
        test_f.write("\n")
        test_f.write(str(X_test['filtered_sequence'].iloc[j]))
        test_f.write("\n")
    test_f.close()

    # Define absolute paths
    blast_bin = "/home/f087s426/ncbi-blast-2.15.0+/bin/"  # Update this if BLAST is installed elsewhere
    blastp_executable = os.path.join(blast_bin, "blastp")
    makeblastdb_executable = os.path.join(blast_bin, "makeblastdb")

    # query_file = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/GPCR_{}_test.fasta".format(i)
    # subject_file = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/GPCR_{}_train.fasta".format(i)
    # db_output = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/GPCR_{}_train_db".format(i)
    # output_file = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/results{}.tab.out".format(i)

    # # Check if files exist
    # if not os.path.isfile(query_file):
    #     print(f"Error: Query file {query_file} does not exist.")
    #     exit(1)
    # if not os.path.isfile(subject_file):
    #     print(f"Error: Subject file {subject_file} does not exist.")
    #     exit(1)

    # Step 1: Create BLAST database
    print("Checking if database already exists...")
    if not os.path.isfile(db_output + ".pin"):  # Check for BLASTP database file
        print("Database not found. Creating BLAST database...")
        makeblastdb_cmd = [
            makeblastdb_executable,
            "-in", train_fasta,
            "-dbtype", "prot",
            "-out", db_output
        ]
        try:
            subprocess.run(makeblastdb_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Database formatting completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during makeblastdb execution:\n{e.stderr.decode()}")
            exit(1)
    else:
        print("Database already exists. Skipping creation.")

    # Step 2: Run BLASTP
    blastp_cmd = [
        blastp_executable,
        "-query", test_fasta,
        "-db", db_output,  # Use formatted database
        "-out", output_file,
        "-outfmt", "6",  # Tabular format
        "-evalue", "100000"
    ]

    # Measure execution time
    start_time = time.time()
    try:
        print("Running BLASTP...")
        subprocess.run(blastp_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("BLASTP completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during BLASTP execution:\n{e.stderr.decode()}")
        exit(1)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"BLASTP execution time: {elapsed_time:.2f} seconds")
    print(f"Results saved in {output_file}")

    model = xgboost.XGBClassifier()
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='macro')}

    # print(loaded_model)
    model.fit(list(X_train['embedding'].values), y_train)

    # store starting time
    begin = time.time()
    xgb_prob = model.predict_proba(list(X_test['embedding'].values))[:, 1]
    time.sleep(1)
    # store end time
    end = time.time()
    # print(f"Total runtime of the program is {end - begin}")
    xgb_auc = roc_auc_score(y_test, xgb_prob)
    print('ROC AUC=%.3f' % (xgb_auc))
    y_pred_test = model.predict(list(X_test['embedding'].values)) > 0.1
    # best_threshold = 0.5
    # best_f1 = 0

    # for threshold in np.linspace(0.4, 0.7, 50):  # Search over thresholds
    #     y_pred = (xgb_prob > threshold).astype(int)
    #     f1 = f1_score(y_test, y_pred)
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_threshold = threshold

    #print(f"Best Threshold: {best_threshold}")
    print(accuracy_score(y_test, y_pred_test))
    weighted_f1 = f1_score(y_test, y_pred_test, average='weighted')
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    precision = precision_score(y_test, y_pred_test)

    acc_list.append((accuracy_score))
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    Class_list.append(i)
    AUC_list.append(xgb_auc)
    f1_list.append(weighted_f1)
    Precision_list.append(precision)

df=pd.DataFrame()
df['Class']=Class_list
#df['Sensitivity']=sensitivity_list
#df['Specificity']=specificity_list
df['AUC']=AUC_list
df['F1-score']=f1_list
df['Precision']=Precision_list
#
# df.to_csv('/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/GPCR_xgboost_results.csv')
