#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys
from os import listdir
from os.path import isfile, join

from utils import *
import csv


# In[2]:


def CNNModel(resultsCNN):
    scoreA = []
    y_predA = []
    
    threshold = 0
    with open("./Thresholds.txt") as thresholds:
        thresholdReader = csv.reader(thresholds)
        
        for row in thresholdReader:
            if resultsCNN[12:-42] == row[0][:-52]:
                threshold = float(row[1])

    with open(resultsCNN) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            scoreA.append(float(row[0]))
            if float(row[0]) > threshold:
                y_predA.append(-1)
            else:
                y_predA.append(1)
    
    score = np.array(scoreA)
    y_pred = np.array(y_predA)
    return score, y_pred


def RunCNNModel(file_name):
    _file_name = os.path.join("./SMD/SMDSeries/"+ file_name)
    abnormal_data, abnormal_label = ReadSMDDataset(file_name)
    
    for file in sorted(os.listdir("./SMDReview/")):
        if file.startswith(file_name):
            print(file, end='')
            score, y_pred = CNNModel(os.path.join("./SMDReview/", file))
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
            print('')


# In[3]:


if __name__ == '__main__':
    filesPath = "./SMD/SMDSeries/"
    onlyfiles = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]

    for file_name in onlyfiles:
        RunCNNModel(file_name)


