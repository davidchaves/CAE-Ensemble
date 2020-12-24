import numpy as np
import os
import csv
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score


def precisionK(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def recallK(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result

numberLine = 0
ground = []
file = 'machine-311'
with open("./SMD/label/"+ file +".txt", "r") as f:
        reader = csv.reader(f)
        for line in reader:
                if numberLine % 10 == 0 and numberLine > 0 and numberLine < 22901:
                        ground.append(float(line[0]))
                        #print(str(int(numberLine/10))+","+line[3])
                numberLine += 1

numberLine = 1
predictionScores = []
predictionValues = []
with open("./SMD/results/"+ file +"_result.csv", "r") as f:
	reader = csv.reader(f)
	for line in reader:
		#print(str(numberLine)+","+str(line[0]))
		predictionScores.append(int(line[0]))
		if int(line[0]) > 3:
			predictionValues.append(1)
		else:
			predictionValues.append(0)
		numberLine += 1

precision, recall, f_score, _ = prf(ground, predictionValues, average='binary', pos_label=1)
precision_k = precisionK(ground, predictionScores,10)
print("Precision: {:0.8f}".format(precision))
#print("Precision@k: {:0.8f}".format(precision_k))
print("Recall: {:0.8f}".format(recall)) 
print("F1: {:0.8f}".format(f_score))
ROC = roc_auc_score(ground, predictionScores)
#if ROC < 0.5:
#	ROC = 1 - ROC  
print('ROC-AUC: {:0.8f}'.format(ROC))
average_PR = average_precision_score(ground, predictionScores)
#if average_PR < 0.5:
#	average_PR = 1 - average_PR  
print('PR-AUC: {:0.8f}'.format(average_PR))

