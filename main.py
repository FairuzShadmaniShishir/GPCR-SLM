from collections import Counter

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
#from transformers imp

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer, accuracy_score
import pickle

df=pd.read_csv(r'/home/f087s426/jupyter notebook/all_family_samples.csv')
pfam=np.load('/home/f087s426/Research/Bio-informatics Collaboration/Embeddings/pfam_esm_embedding.npy')
pfam_df=pd.DataFrame(pfam)
full_df=pd.concat([df, pfam_df], axis=1)
class_df=full_df.where(full_df['family_accession']=='PF13649.6').dropna()
miscellaneous_df=full_df.where(full_df['family_accession']!='PF13649.6').dropna().head(8000)
class_df['family_accession']=0
miscellaneous_df['family_accession']=1
final_df=pd.concat([class_df,miscellaneous_df])
final_df=final_df.sample(frac=1)
final_df.iloc[:,7:]
X_train, X_test, y_train, y_test = train_test_split(final_df.iloc[:,7:] ,final_df['family_accession'].astype('int'), test_size=0.2)
print('Process Terminated')

import imblearn
# from imblearn.over_sampling import RandomOverSampler
#
# ros = RandomOverSampler(random_state=42)
#
# x_ros, y_ros = ros.fit_resample(X_train, y_train)
#
# print('Original dataset shape', Counter(y_train))
# print('Resample dataset shape', Counter(y_ros))

# from imblearn.over_sampling import SMOTE
# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)
# print(y_train.value_counts())

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='macro')}

#print(loaded_model)

ratio = (y_train.value_counts()[0]) / y_train.value_counts()[1]
model = xgboost.XGBClassifier(scale_pos_weight=ratio)
model.fit(X_train, y_train)
import time
start = time.time()

xgb_prob=model.predict_proba(X_test)[:, 1]
predict=model.predict(X_test)

end = time.time()

print(end - start)
print(predict)
xgb_auc = roc_auc_score(y_test, xgb_prob)
print('ROC AUC=%.3f' % (xgb_auc))
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)
plt.plot(xgb_fpr, xgb_tpr, marker='.', label='xgboost')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

print(classification_report(y_test, predict))