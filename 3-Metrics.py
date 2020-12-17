#!/usr/bin/env python
# coding: utf-8

# ## Metrics of baseline and Convolutional Sequence-to-sequence

# Based on the OED implementation: https://github.com/tungk/OED  

# In[1]:


import pathlib
import sys
from os import listdir
from os.path import isfile, join

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import *
import csv


# In[2]:


def OCSVMModel(_abnormal_data, _kernel='rbf', _nu=0.5):
    clf = svm.OneClassSVM(kernel=_kernel, nu=_nu, gamma='auto')
    clf.fit(_abnormal_data)
    score = clf.decision_function(_abnormal_data)
    y_pred = clf.predict(_abnormal_data)
    return score, y_pred


def RunSVMModel(_file_name, _kernel='rbf', _nu=0.5):
    #_file_name = os.path.join("./SMDTesting/"+ _file_name)
    abnormal_data, abnormal_label = ReadSMDDataset(_file_name)

    score, y_pred = OCSVMModel(abnormal_data, _kernel, _nu)
    score_pred_label = np.c_[score, y_pred, abnormal_label]

    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)

    fpr, tpr, roc_auc = CalculateROCAUCMetrics(abnormal_label, score)

    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(abnormal_label, score)

    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
    return precision, recall, f1, roc_auc, average_precision, cks

def IFModel(_abnormal_data, _estimators=100, _contamination=0.1):
    clf = IsolationForest(n_estimators=_estimators, contamination=_contamination,behaviour="new")
    clf.fit(_abnormal_data)
    score = clf.decision_function(_abnormal_data)
    y_pred = clf.predict(_abnormal_data)
    return score, y_pred

def RunIFModel(_file_name, _estimators=100, _contamination=0.1):
    #_file_name = os.path.join("./SMDTesting/"+ _file_name)
    abnormal_data, abnormal_label = ReadSMDDataset(_file_name)
    score, y_pred = IFModel(abnormal_data, _estimators, _contamination)
    score_pred_label = np.c_[score, y_pred, abnormal_label]

    x = abnormal_label[np.where(abnormal_label == -1)]
    y = y_pred[np.where(y_pred == -1)]

    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)

    fpr, tpr, roc_auc = CalculateROCAUCMetrics(abnormal_label, score)

    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(abnormal_label, score)
    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)

    return precision, recall, f1, roc_auc, average_precision, cks

def LOFModel(_abnormal_data, _neighbor=5, _job=2, _contamination=0.05, _metric='euclidean'):
    clf = LocalOutlierFactor(n_neighbors=_neighbor, n_jobs=_job, metric=_metric, contamination=0.05)
    y_pred = clf.fit_predict(_abnormal_data)
    score = clf.negative_outlier_factor_
    return score, y_pred

def RunLOFModel(_file_name, _neighbor=5, _job=2, _contamination=0.1, _metric='euclidean'):
    #_file_name = os.path.join("./SMDTesting/"+ _file_name)
    abnormal_data, abnormal_label = ReadSMDDataset(_file_name)
    score, y_pred = LOFModel(abnormal_data, _neighbor, _job, _metric)
    score_pred_label = np.c_[score, y_pred, abnormal_label]
    x = abnormal_label[np.where(abnormal_label == -1)]
    y = y_pred[np.where(y_pred == -1)]

    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
    fpr, tpr, roc_auc = CalculateROCAUCMetrics(abnormal_label, score)

    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(abnormal_label, score)

    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
    return precision, recall, f1, roc_auc, average_precision, cks

def NNModel(resultsNN):
    scoreA = []
    y_predA = []
    
    with open(resultsNN) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            scoreA.append(float(row[0]))
            y_predA.append(int(row[1]))
    
    score = np.array(scoreA)
    y_pred = np.array(y_predA)
    return score, y_pred

def RunCNNModel(file_name):
    #_file_name = os.path.join("./SMDTesting/"+ file_name)
    abnormal_data, abnormal_label = ReadSMDDataset(file_name)

    for file in sorted(os.listdir("./SMDOutput/")):
        if file.startswith(file_name):
            score, y_pred = NNModel(os.path.join("./SMDOutput/", file))
            score_pred_label = np.c_[score, y_pred, abnormal_label]

            x = abnormal_label[np.where(abnormal_label == -1)]
            y = y_pred[np.where(y_pred == -1)]

            precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
            fpr, tpr, roc_auc = CalculateROCAUCMetrics(abnormal_label, score)

            precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(abnormal_label, score)

            cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)
            
            print(',' + str(precision), end='')
            print(',' + str(recall), end='')
            print(',' + str(f1), end='')
            print(',' + str(roc_auc), end='')
            print(',' + str(average_precision), end='')
            print(',' + str(cks), end='')

def RunRNNModel(file_name):
    #_file_name = os.path.join("./SMDTesting/"+ file_name)
    abnormal_data, abnormal_label = ReadSMDDataset(file_name)

    score, y_pred = NNModel(os.path.join("./SMDRNN/", "Results_"+file_name))
    score_pred_label = np.c_[score, y_pred, abnormal_label]

    x = abnormal_label[np.where(abnormal_label == -1)]
    y = y_pred[np.where(y_pred == -1)]

    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, y_pred)
    fpr, tpr, roc_auc = CalculateROCAUCMetrics(abnormal_label, score)

    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(abnormal_label, score)

    cks = CalculateCohenKappaMetrics(abnormal_label, y_pred)

    print(',' + str(precision), end='')
    print(',' + str(recall), end='')
    print(',' + str(f1), end='')
    print(',' + str(roc_auc), end='')
    print(',' + str(average_precision), end='')
    print(',' + str(cks), end='')


# In[3]:


if __name__ == '__main__':
    filesPath = "./SMDTesting/"
    onlyfiles = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]

    for file_name in onlyfiles:
        #file_name = "real_9.csv"
        print(file_name,end='')
        precision, recall, f1, roc_auc, pr_auc, cks = RunSVMModel(file_name)
        #print('One Class SVM metrics:')
        print(',' + str(precision), end='')
        print(',' + str(recall), end='')
        print(',' + str(f1), end='')
        print(',' + str(roc_auc), end='')
        print(',' + str(pr_auc), end='')
        print(',' + str(cks), end='')
        #print('')
        #print('Isolation Forest metrics:')
        precision, recall, f1, roc_auc, pr_auc, cks = RunIFModel(file_name)
        print(',' + str(precision), end='')
        print(',' + str(recall), end='')
        print(',' + str(f1), end='')
        print(',' + str(roc_auc), end='')
        print(',' + str(pr_auc), end='')
        print(',' + str(cks), end='')
        #print('')
        #print('Local Outlier Factor metrics:')
        precision, recall, f1, roc_auc, pr_auc, cks = RunLOFModel(file_name)
        print(',' + str(precision), end='')
        print(',' + str(recall), end='')
        print(',' + str(f1), end='')
        print(',' + str(roc_auc), end='')
        print(',' + str(pr_auc), end='')
        print(',' + str(cks), end='')
        #RunRNNModel(file_name)
        RunCNNModel(file_name)
        print('')


# In[ ]:




