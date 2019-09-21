# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:27:41 2019

@author: medha
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt


lb_enc = LabelEncoder()
data = pd.read_csv('indian_liver_patient.csv')
data["Gender"] = lb_enc.fit_transform(data["Gender"])
cols = ['Dataset']
X = data.drop(cols, axis=1).fillna(0).values
Y = data['Dataset'].fillna(0).values
y = label_binarize(Y, classes=[1, 2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
n_classes = 2
rf_lm = LogisticRegression(solver='lbfgs', max_iter=10000)
m = rf_lm.fit(X_train, y_train)
y_score = rf_lm.fit(X_train, y_train).decision_function(X_test)
#rf_enc.fit(rf.apply(X_train))
#rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:], y_score[:])
    roc_auc[i] = auc(fpr[i], tpr[i])
#############For individual class ROC plot############
plt.figure()
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#y_pred_rf_lm = rf_lm.predict_proba(X_test)[:, 1]
#fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
