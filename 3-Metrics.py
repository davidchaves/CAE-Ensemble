#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys
from os import listdir
from os.path import isfile, join

from utils import *
import csv
import logging
from sklearn import preprocessing


# In[2]:


def ReadFiles(scoresPath, method, dataset):
    onlyfiles = [f for f in listdir(scoresPath) if isfile(join(scoresPath, f))]

    for file_name in onlyfiles:
        if dataset == 'Yahoo':
            abnormal_data, abnormal_label = ReadYahooDataset(scoresPath + file_name)
        elif dataset == 'Donut':
            abnormal_data, abnormal_label = ReadDonutDataset(scoresPath + file_name)
        elif dataset == 'ECG':     
            abnormal_data, abnormal_label = ReadECGDataset(scoresPath + file_name)
        elif dataset == 'SMD': 
            abnormal_data, abnormal_label = ReadSMDDataset(file_name)
        
        if method == 'RAEEnsemble':
            reviewPath = './OEDSixEnsemble/Scores/' + dataset + '/'
        elif method == 'DAGMM':
            reviewPath = './DAGMMReview/Scores/' + dataset + '/'
        elif method == 'RAE':     
            reviewPath = './RNNReview/Scores/' + dataset + '/'
        elif method == 'MSCRED': 
            reviewPath = './MSCREDReview/' + dataset + '/Extended/'

        
        anomalySegment = np.argwhere(np.diff(abnormal_label.squeeze()))

        if (len(anomalySegment)%2 != 0):
            anomalySegment = np.append(anomalySegment,len(abnormal_data))

        segmentsOutlier = anomalySegment.squeeze().reshape(-1, 2)
        
        for file in sorted(os.listdir(reviewPath)):
            if file.startswith("complete_" + file_name[:-3]):
                CalculateMetrics(reviewPath,file, method, dataset, abnormal_label,segmentsOutlier,'No')
                CalculateMetrics(reviewPath,file, method, dataset, abnormal_label,segmentsOutlier,'Yes')


# In[3]:


def CalculateMetrics(reviewPath, resultsFile, method, dataset, abnormal_label, segmentsOutlier, adjust='No'):    
    with open("./ThresholdCalculation/Baselines/" + method + dataset + ".txt") as thresholds:
        thresholdReader = csv.reader(thresholds)
        arrayThreshold = np.empty([0, 3])

        for row in thresholdReader:

            if resultsFile == row[0]:
                current = np.array([[float(row[1]), float(row[2]),float(row[3])]]) #,float(row[3])
                arrayThreshold = np.concatenate((arrayThreshold,current))

        for thresholds in arrayThreshold:
            scoreArray = []
            predictedArray = []

            with open(os.path.join(reviewPath, resultsFile)) as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    scoreArray.append(float(row[0]))
                    if float(row[0]) > thresholds[2]: #1
                        predictedArray.append(-1)
                    else:
                        predictedArray.append(1)

            score = np.array(scoreArray)
            predictedA = np.array(predictedArray)

            predicted = pd.DataFrame(predictedA, columns=['Score'])

            if (adjust=='Yes'):
                for segment in segmentsOutlier:
                    predictedSegment = predicted.iloc[segment[0]+1:segment[1]+1]
                    try:
                        if (predictedSegment['Score'].value_counts()[1] != segment[1] - segment[0]):
                            predicted.iloc[segment[0]+1:segment[1]+1] = -1
                    except:
                        pass

            
            precision, recall, f1 = CalculatePrecisionRecallF1Metrics(abnormal_label, predicted)
            fpr, tpr, roc_auc = CalculateROCAUCMetrics(abnormal_label, score)
            precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(abnormal_label, score)

            print(resultsFile, end='')
            print(',' +method, end='')
            print(',' +dataset, end='')
            print(',' +adjust, end='')
            print(',' + str(thresholds[0]), end='') #q
            print(',' + str(thresholds[1]), end='') #level
            print(',' + str(precision), end='')
            print(',' + str(recall), end='')
            print(',' + str(f1), end='')
            print(',' + str(roc_auc), end='')
            print(',' + str(average_precision), end='')
            print('')


# In[4]:


datasets = ['ECG','SMD','Yahoo','Donut']
for dataset in datasets:
    if dataset == 'Yahoo':
        filesPath = "./Yahoo/Test/"
        for method in ['RAEEnsemble','RAE']: #,'DAGMM'
            ReadFiles(filesPath, method, dataset) 
    elif dataset == 'Donut':
        filesPath = "./Donut/"
        for method in ['RAEEnsemble','RAE']: #,'DAGMM'
            ReadFiles(filesPath, method, dataset) 
    elif dataset == 'ECG':     
        filesPath = "./ECG/"
        for method in ['RAEEnsemble','RAE','MSCRED']: #,'DAGMM'
            ReadFiles(filesPath, method, dataset)     
    elif dataset == 'SMD': 
        filesPath = "./SMD/SMDSeries/"
        for method in ['RAEEnsemble','RAE','MSCRED']: #,'DAGMM'
            ReadFiles(filesPath, method, dataset)  


# In[ ]:




