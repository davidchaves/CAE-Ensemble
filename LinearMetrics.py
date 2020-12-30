#!/usr/bin/env python
# coding: utf-8

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

def IFModel(_abnormal_data, _estimators=100, _contamination=0.1):
    clf = IsolationForest(n_estimators=_estimators, contamination=_contamination)
    clf.fit(_abnormal_data)
    score = clf.decision_function(_abnormal_data)
    y_pred = clf.predict(_abnormal_data)
    return score, y_pred

def LOFModel(_abnormal_data, _neighbor=5, _job=2, _contamination=0.05, _metric='euclidean'):
    clf = LocalOutlierFactor(n_neighbors=_neighbor, n_jobs=_job, metric=_metric, contamination=_contamination)
    y_pred = clf.fit_predict(_abnormal_data)
    score = clf.negative_outlier_factor_
    return score, y_pred


# In[3]:


filesPath = "../Yahoo/Test/"  #"../Donut/" #"../ECG/" #"../SMD/SMDSeries/" #"../Yahoo/Test/" 
typeThreshold = 'Fixed'
onlyfiles = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]

for file_name in onlyfiles:
    #abnormal_data, abnormal_label = ReadDonutDataset(filesPath + file_name)
    #abnormal_data, abnormal_label = ReadECGDataset(filesPath + file_name)
    #abnormal_data, abnormal_label = ReadSMDDataset(file_name)
    abnormal_data, abnormal_label = ReadYahooDataset(filesPath + file_name)

    
    for i in [0.5,1,5,10]:
        score, y_pred = OCSVMModel(abnormal_data, _nu=i/100)
        print("SVM,", end='')
        print(file_name + ",", end='')
        print(typeThreshold, end='')
        print(',' + str(i), end='')

        precision, recall, f1 = CalculatePrecisionRecallF1Metrics(y_pred, abnormal_label)

        print(',' + str(precision), end='')
        print(',' + str(recall), end='')
        print(',' + str(f1), end='')
        print('') 
        
        score, y_pred = IFModel(abnormal_data, _contamination=i/100)
        print("IF,", end='')
        print(file_name + ",", end='')
        print(typeThreshold, end='')
        print(',' + str(i), end='')

        precision, recall, f1 = CalculatePrecisionRecallF1Metrics(y_pred, abnormal_label)

        print(',' + str(precision), end='')
        print(',' + str(recall), end='')
        print(',' + str(f1), end='')
        print('') 
        
        score, y_pred = LOFModel(abnormal_data, _contamination=i/100)
        print("LOF,", end='')
        print(file_name + ",", end='')
        print(typeThreshold, end='')
        print(',' + str(i), end='')

        precision, recall, f1 = CalculatePrecisionRecallF1Metrics(y_pred, abnormal_label)

        print(',' + str(precision), end='')
        print(',' + str(recall), end='')
        print(',' + str(f1), end='')
        print('') 


# In[ ]:




