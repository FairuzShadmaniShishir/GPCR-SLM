import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer, accuracy_score,classification_report
# Define the parent directory containing subfolders
parent_dir = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta"
acc_list = []
sensitivity_list = []
specificity_list = []
Class_list = []
AUC_list = []
f1_list =[]
Precision_list=[]

# Iterate through subfolders
for folder in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder)

    # Ensure it's a directory
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".out"):
                file_path = os.path.join(folder_path, file)
                colnames = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart',
                            'send', 'evalue', 'bitscore']
                df = pd.read_csv(file_path,sep='\t',header=None, names=colnames)
                df = df.sort_values(by=['evalue'])
                df = df.drop_duplicates(subset=['qseqid'])
                actual = df['sseqid'].apply(lambda r: '_'.join(r.split('_')[:-1]))
                predicted = df['sseqid'].apply(lambda r: '_'.join(r.split('_')[:-1]))
                actual = [int(label) for label in actual]
                predicted = [int(label) for label in predicted]
                #print(len(df))
                #print(f"Loaded {file} from {folder}")
                #print(df.head())
                print(accuracy_score(actual, predicted))
                weighted_f1 = f1_score(actual, predicted, average='weighted')
                print(f"Weighted F1-score: {weighted_f1:.4f}")
                precision = precision_score(actual, predicted)

                acc_list.append((accuracy_score))
                cm = confusion_matrix(actual, predicted)
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                sensitivity_list.append(sensitivity)
                specificity_list.append(specificity)
                #print(file)
                #parts = file.split("_")  # Split by underscore (_)
                extracted_number = file.split("_")[1].split(".")[0]
                print(extracted_number)
                Class_list.append(extracted_number)
                #AUC_list.append(xgb_auc)
                f1_list.append(weighted_f1)
                Precision_list.append(precision)


df=pd.DataFrame()
df['Class']=Class_list
#df['Sensitivity']=sensitivity_list
#df['Specificity']=specificity_list
#df['AUC']=AUC_list
df['F1-score']=f1_list
df['Precision']=Precision_list
df.to_csv('/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/GPCR_Blast_results.csv')
