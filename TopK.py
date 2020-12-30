import pathlib
import sys
from os import listdir
from os.path import isfile, join

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import *
import csv

filesPath = "../ECG/"  #"../Donut/" #"../ECG/" #"../SMD/SMDSeries/" #"../Yahoo/Test/" 
reviewPath = "../ECGReview/" #"../DonutReview/" #"../ECGReview/" #"../SMDReview/" #"../YahooReview/"
typeThreshold = 'Automatic'
onlyfiles = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]

for file_name in onlyfiles:
    #abnormal_data, abnormal_label = ReadDonutDataset(filesPath + file_name)
    abnormal_data, abnormal_label = ReadECGDataset(filesPath + file_name)
    #abnormal_data, abnormal_label = ReadSMDDataset(file_name)
    #abnormal_data, abnormal_label = ReadYahooDataset(filesPath + file_name)
    
    for file in sorted(os.listdir(reviewPath)):
        if file.startswith(file_name):
            
            results = pd.read_csv(reviewPath+file,header=None)

            score = results.iloc[0:abnormal_data.shape[0],0]
            
            threshold = 0
            if typeThreshold == 'Automatic':
                with open("../ThresholdsNames.txt") as thresholds:
                    thresholdReader = csv.reader(thresholds)
                    for row in thresholdReader:
                        if file_name == row[0]:
                            threshold = float(row[1])
            else:
                threshold = 2.5
                
            y_predicted = []
            
            for row in score:
                if row > threshold:
                    y_predicted.append(-1)
                else:
                    y_predicted.append(1)   

            #y_pred = results.iloc[0:abnormal_data.shape[0],1]
            y_pred = pd.DataFrame(y_predicted, columns=['Score']).iloc[0:abnormal_data.shape[0],0]

            for i in [0.5,1,5,10]:
                print(reviewPath[3:-7] + ",", end='')
                print(file + ",", end='')
                print(typeThreshold, end='')
                print(',' + str(i), end='')
                rangeTop = int(-1 * len(score) * i / 100)
                indices = np.argpartition(score, rangeTop)[rangeTop:]
                k_truth = abnormal_label[indices]
                k_predicted = y_pred[indices]

                precision, recall, f1 = CalculatePrecisionRecallF1Metrics(k_predicted, k_truth)

                print(',' + str(precision), end='')
                print(',' + str(recall), end='')
                print(',' + str(f1), end='')
                print('') 