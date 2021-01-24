#!/usr/bin/env python
# coding: utf-8

# In[1]:


from spot import SPOT
from os import listdir
from os.path import isfile, join
import csv
import numpy as np
import logging
import asyncio
import time
from sklearn import preprocessing
import pandas as pd


# In[2]:


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


# In[3]:


@background
def parallelSpot(score,filename, q=1e-4, level=0.01):
    s = SPOT(q)  # SPOT object
    s.fit(score, score)  # data import
    s.initialize(level=level)  # initialization step
    ret = s.run()  # run
    pot_th = np.mean(ret['thresholds'])
    
    logging.info(filename + "," + str(q) + "," + str(level) + "," +str(pot_th))


# In[4]:


def ReadScores(resultsCNN):
    scoreA = []
    y_predA = []
    
    with open(resultsCNN) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            scoreA.append(float(row[0]))
            y_predA.append(int(row[1]))
    
    score = np.array(scoreA)
    
    x = score.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataset = pd.DataFrame(x_scaled).squeeze()
    return normalizedDataset


# In[5]:


dataset = 'Donut'
method = 'RAEEnsemble'
logging.basicConfig(level=logging.DEBUG, filename="./Baselines/"+ method + dataset +".txt", filemode="a+",format="%(message)s")

filesPath = "../OEDSixEnsemble/Scores/"+dataset+"/"
onlyfiles = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]

for file_name in onlyfiles:
    score = ReadScores(filesPath + file_name)
    parallelSpot(score,file_name,0.01,0.04)


# In[ ]:




