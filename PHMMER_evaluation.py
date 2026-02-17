import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc as sk_auc
import numpy as np

colnames = ['target_name', 'target_accession', 'query_name', 'query_accession',
            'full_evalue', 'full_score', 'full_bias',
            'domain_evalue', 'domain_score', 'domain_bias',
            'exp', 'reg', 'clu', 'ov', 'env', 'dom', 'rep', 'inc',
            'description']

file_paths = glob.glob('/home/f087s426/GPCR_PHMMER/*/GPCR_*_phmmer.out')
records = []

for path in file_paths:
    class_name = os.path.basename(os.path.dirname(path))  # e.g., 'GPCR_0', 'GPCR_1'
    class_label = ''.join(filter(str.isdigit, class_name))  # extract numeric part only, e.g., '0'
    print(class_name)

    test_family = pd.read_csv(path, sep='\s+',  # Use whitespace as separator
                              comment='#',  # Skip lines starting with #
                              header=None, names=colnames,low_memory=False)
    test_family['full_evalue'] = pd.to_numeric(test_family['full_evalue'], errors='coerce')
    test_family = test_family.dropna()
    filtered_df = test_family[test_family['query_name'].astype(str).str.startswith('1')]

    auc_list = []
    f1_list = []
    accuracy_list = []
    aupr_list = []

    for i in filtered_df['query_name'].unique():
        df = filtered_df[filtered_df['query_name'] == i]
        sorted_evalue = list(df.full_evalue.sort_values(ascending=False))
        actual = df['query_name'].apply(lambda r: '_'.join(r.split('_')[:-1])).tolist()
        predicted = df['target_name'].apply(lambda r: '_'.join(r.split('_')[:-1])).tolist()

        TP_list, FP_list, FN_list, TN_list = [], [], [], []
        precision_list = []
        recall_list = []

        for j in range(len(sorted_evalue)):
            TP = FP = FN = TN = 0
            for k in range(len(sorted_evalue)):
                if sorted_evalue[j] <= sorted_evalue[k]:
                    TP += predicted[k] == '1'
                    FP += predicted[k] == '0'
                else:
                    FN += predicted[k] == '1'
                    TN += predicted[k] == '0'

            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)
            TN_list.append(TN)

            # Calculate precision and recall for AUPR
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            precision_list.append(precision)
            recall_list.append(recall)

        # Calculate TPR and FPR for ROC-AUC
        tpr_list = [TP / (TP + FN) if (TP + FN) > 0 else 0 for TP, FN in zip(TP_list, FN_list)]
        fpr_list = [FP / (FP + TN) if (FP + TN) > 0 else 0 for FP, TN in zip(FP_list, TN_list)]

        # ROC-AUC calculation
        if len(fpr_list) > 1 and len(tpr_list) > 1:
            fpr_sorted, tpr_sorted = zip(*sorted(zip(fpr_list, tpr_list)))
            roc_auc = sk_auc(fpr_sorted, tpr_sorted)
            auc_list.append(roc_auc)

        # AUPR calculation
        if len(recall_list) > 1 and len(precision_list) > 1:
            # Sort by recall for AUPR calculation
            recall_sorted, precision_sorted = zip(*sorted(zip(recall_list, precision_list)))
            aupr = sk_auc(recall_sorted, precision_sorted)
            aupr_list.append(aupr)

        # Find optimal threshold for F1 and Accuracy
        best_f1 = 0
        best_accuracy = 0

        for j in range(len(sorted_evalue)):
            TP, FP, FN, TN = TP_list[j], FP_list[j], FN_list[j], TN_list[j]

            # F1 Score
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Accuracy
            accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
            if accuracy > best_accuracy:
                best_accuracy = accuracy

        f1_list.append(best_f1)
        accuracy_list.append(best_accuracy)

    # Calculate averages for this class
    if auc_list:
        avg_auc = sum(auc_list) / len(auc_list)
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
        avg_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0
        avg_aupr = sum(aupr_list) / len(aupr_list) if aupr_list else 0

        records.append({
            'class': int(class_label),
            'average_auc': avg_auc,
            'average_f1': avg_f1,
            'average_accuracy': avg_accuracy,
            'average_aupr': avg_aupr
        })

        print(f"Class {class_label}:")
        print(f"  Average AUC: {avg_auc:.4f}")
        print(f"  Average F1: {avg_f1:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"  Average AUPR: {avg_aupr:.4f}")
        print()

# Create DataFrame from records
metrics_dff = pd.DataFrame(records)
print("Summary Metrics DataFrame:")
print(metrics_dff)

# # Optional: Create a visualization of the metrics
# if len(metrics_df) > 0:
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     fig.suptitle('Performance Metrics by Class', fontsize=16)

#     # AUC plot
#     axes[0, 0].bar(metrics_df['class'], metrics_df['average_auc'], color='skyblue')
#     axes[0, 0].set_title('Average AUC')
#     axes[0, 0].set_xlabel('Class')
#     axes[0, 0].set_ylabel('AUC')
#     axes[0, 0].set_ylim(0, 1)

#     # F1 plot
#     axes[0, 1].bar(metrics_df['class'], metrics_df['average_f1'], color='lightgreen')
#     axes[0, 1].set_title('Average F1 Score')
#     axes[0, 1].set_xlabel('Class')
#     axes[0, 1].set_ylabel('F1 Score')
#     axes[0, 1].set_ylim(0, 1)

#     # Accuracy plot
#     axes[1, 0].bar(metrics_df['class'], metrics_df['average_accuracy'], color='lightcoral')
#     axes[1, 0].set_title('Average Accuracy')
#     axes[1, 0].set_xlabel('Class')
#     axes[1, 0].set_ylabel('Accuracy')
#     axes[1, 0].set_ylim(0, 1)

#     # AUPR plot
#     axes[1, 1].bar(metrics_df['class'], metrics_df['average_aupr'], color='gold')
#     axes[1, 1].set_title('Average AUPR')
#     axes[1, 1].set_xlabel('Class')
#     axes[1, 1].set_ylabel('AUPR')
#     axes[1, 1].set_ylim(0, 1)

#     plt.tight_layout()
#     plt.show()

# Optional: Save results to CSV
metrics_dff.to_csv('gpcr_performance_metrics_hmmer.csv', index=False)
# print(f"\nResults saved to 'gpcr_performance_metrics.csv'")