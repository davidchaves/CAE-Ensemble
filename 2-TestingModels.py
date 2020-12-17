#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from datetime import datetime
from statistics import mean, stdev
import os
from os import listdir
from os.path import isfile, join
import logging

from torch.utils.data import DataLoader
import torch.utils.data as data_utils

import matplotlib.pyplot as plt

import numpy as np

import random
import math
import time

import pandas as pd
from scipy.stats import zscore
import csv

seed = int(time.time())

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


class encoder(nn.Module):

    def __init__(self):
        super(encoder, self).__init__()
        self.linear = nn.Linear(in_features=numberColumns, out_features=linearChannels, bias=True).to(device)
        self.position = nn.Linear(1, linearChannels).to(device)
        self.convolutions = nn.ModuleList([nn.Conv1d(in_channels = convolutionalChannels, 
                                                 out_channels = convolutionalChannels*2, kernel_size = kernelSize, 
                                                 padding = (kernelSize - 1) // 2) for _ in range(10)]).to(device)
        self.toHidden = nn.Linear(linearChannels, convolutionalChannels).to(device)
        self.fromHidden = nn.Linear(convolutionalChannels, linearChannels).to(device)
        self.dropout = nn.Dropout().to(device)
        
    def forward(self, seriesInput, batch):
        linearOutput = torch.unsqueeze(self.linear(seriesInput.to(device)), 2)
        positionSeries = torch.arange((batch-1) * linearOutput.size()[0], 
                                      batch * linearOutput.size()[0]).unsqueeze(1).float().to(device)
        positionEmbedded = self.position(positionSeries/len(normalizedDataset)).unsqueeze(1).permute(0, 2, 1)
        embedded = self.dropout(linearOutput + positionEmbedded).permute(0, 2, 1)
        convolutionInput = torch.unsqueeze(self.toHidden(embedded.squeeze()), 2)
    
        for i, convolution in enumerate(self.convolutions):
            convolutionHidden = F.glu(convolution(self.dropout(convolutionInput)), dim = 1)
            convolutionHidden = convolutionHidden + convolutionInput
            convolutionInput = convolutionHidden
        
        convolutionHidden = self.fromHidden(convolutionHidden.permute(0, 2, 1))
        residualConnection = convolutionHidden + embedded
        return convolutionHidden, residualConnection


# In[3]:


class decoder(nn.Module):

    def __init__(self):
        super(decoder, self).__init__()
        self.linear = nn.Linear(in_features=linearChannels, out_features=attentionChannels, bias=True).to(device)
        self.position = nn.Embedding(linearChannels, attentionChannels).to(device)
        self.convolutions = nn.ModuleList([nn.Conv1d(in_channels = convolutionalChannels, 
                                                     out_channels = convolutionalChannels*2, 
                                                     kernel_size = kernelSize) for _ in range(10)]).to(device)
        self.toHidden = nn.Linear(attentionChannels, convolutionalChannels).to(device)
        self.fromHidden = nn.Linear(convolutionalChannels, attentionChannels).to(device)
        self.attentiontoHidden = nn.Linear(attentionChannels, convolutionalChannels).to(device)
        self.attentionfromHidden = nn.Linear(convolutionalChannels, attentionChannels).to(device)
        self.dropout = nn.Dropout().to(device)
        self.output = nn.Linear(attentionChannels, numberColumns).to(device)
        
    def calculateAttention(self, embedded, convolutionHidden, encoderConvolutionHidden, encoderResidualConnection):
        residualConnection = self.attentionfromHidden(convolutionHidden.permute(0, 2, 
                                                                                1)).squeeze() + embedded.squeeze()
        attention = F.softmax(torch.matmul(residualConnection.permute(1, 0), 
                                           encoderConvolutionHidden.squeeze()), dim=1)
        attentionEncoded = self.attentiontoHidden(torch.matmul(attention, encoderResidualConnection.squeeze()
                                                               .permute(1,0)).permute(1,0))
        attentionConnection = convolutionHidden.squeeze() + attentionEncoded
        return attention, attentionConnection

    def forward(self, hiddenEncoder, residualEncoder):
        linearInput = torch.unsqueeze(self.linear(torch.squeeze(hiddenEncoder)), 2)
        positionSeries = torch.arange(0, hiddenEncoder.shape[1]).unsqueeze(0).repeat(linearInput.size()[0], 
                                                                                     1).to(device)
        positionEmbedded = self.position(positionSeries).permute(0, 2, 1)
        embedded = self.dropout(linearInput + positionEmbedded)
        convolutionInput = self.toHidden(embedded.squeeze())
        
        for i, convolution in enumerate(self.convolutions):
            convolutionInput = self.dropout(torch.unsqueeze(convolutionInput, 2))
            padding = torch.zeros(convolutionInput.size()[0], 512, 3-1).fill_(1).to(device)
            paddedConvolutionInput = torch.cat((padding, convolutionInput), dim = 2)
            convolutionHidden = F.glu(convolution(paddedConvolutionInput), dim = 1)
            attention, convolutionHidden = self.calculateAttention(embedded, convolutionHidden, 
                                                                    hiddenEncoder, residualEncoder)
            convolutionHidden = convolutionHidden + convolutionInput.squeeze()
            convolutionInput = convolutionHidden
            
        output = self.output(self.dropout(self.fromHidden(convolutionHidden)))
        stdMean = torch.std_mean(output)
        normalizedDecoder = ((output - stdMean[1])/stdMean[0])
        return normalizedDecoder


# In[4]:


lossFunction1 = nn.BCEWithLogitsLoss(reduction='mean')
lossFunction = nn.L1Loss()
lossFunction3 = nn.PoissonNLLLoss()
lossFunction4 = nn.MSELoss()

def diversityLoss(input, target, iteration):
    sizeInput = input.size()[0]
    #rangeEnsemble = torch.tensor([outputCycle[k] for k in range((iteration-1) * sizeInput,iteration * sizeInput)])
    #lossDiversity = loss_function(input, target) #- diversityFactor * torch.norm(input - rangeEnsemble, 2)/sizeInput
    normResults = torch.zeros(1, 1).to(device)
    for i in range(0,cycle):
        #indices = torch.arange(iteration * sizeInput, (iteration+1) * sizeInput).to(device)
        #rangeEnsemble = torch.index_select(outputCycle[i], 0, indices).to(device)
        
        #rangeEnsemble = torch.tensor([outputCycle[i][k] for k in range((iteration-1) * sizeInput,iteration * sizeInput)])
        #normResults += torch.norm(input - rangeEnsemble, 2).to(device)/sizeInput
        normResults += torch.norm(input - outputCycle[i], 2).to(device)/sizeInput
        #print(normResults)
        
    lossDiversity = lossFunction(input, target) - diversityFactor * normResults

    return lossDiversity


# In[5]:


def VariableLR(initialLR, currentEpoch, epochPerCycle):
    return initialLR * (np.cos(np.pi * currentEpoch / epochPerCycle) + 1) / 2


# In[6]:


def pot_eval(init_score, score, q=1e-3, level=0.01):
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level)  # initialization step
    ret = s.run()  # run
    pot_th = np.mean(ret['thresholds'])
   
    print('POT result: ', pot_th)
    return pot_th


# In[7]:


def loading(encoderName,decoderName):

    encoder_eval = encoder()
    decoder_eval = decoder()

    encoder_eval.load_state_dict(torch.load(os.path.join("/mnt/Models", encoderName)))
    decoder_eval.load_state_dict(torch.load(os.path.join("/mnt/Models", decoderName)))

    data = autograd.Variable(torchDataset)

    encoder_eval.eval()
    decoder_eval.eval()

    a, b = encoder_eval(data,1)
    reconstruction = decoder_eval(a, b)

    normDifferences = torch.norm(reconstruction - torchDataset, dim=1)

    score = zscore(normDifferences.tolist())
    outlierPred_Normal = []
    outlierPred_Automatic = []

    thresholdNormal = 2.5 
    for i in range (len(score)):
        if score[i] < thresholdNormal:
            outlierPred_Normal.append(1)
        else:
            outlierPred_Normal.append(-1)

    with open("./Output/"+encoderName[8:-4]+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(score, outlierPred_Normal))
        
    #thresholdAutomatic = pot_eval(score,score)
    #for i in range (len(score)):
    #    if score[i] < thresholdAutomatic:
    #        outlierPred_Automatic.append(1)
    #    else:
    #        outlierPred_Automatic.append(-1)
    #
    #with open("./Results/Threshold/xmitdb_x108_0_Automatic.csv", 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(zip(score, outlierPred_Automatic))


# In[9]:


numberEpochs = 10
cycles = 20
beta = 0.9
diversityFactor = 0.8
learningRate = 0.001
linearChannels = 64
convolutionalChannels = 512
attentionChannels = 256
kernelSize = 3

filesPath = "./Testing/"
onlyfiles = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]

for nameFile in onlyfiles:
    importedData = pd.read_csv(filesPath + nameFile, skiprows=1, header=None, sep= " ")
    importedData = importedData.iloc[:, :-1]
    numberColumns = len(importedData.columns)
    
    values = importedData + 1e-7
    meanImported = np.mean(values)
    stdImported = np.std(values)
    normalizedDataset = ((values - meanImported)/stdImported).fillna(0)
    torchDataset = torch.from_numpy(normalizedDataset.values).float().to(device)
    dataloader = DataLoader(torchDataset, batch_size=len(normalizedDataset), shuffle=True)
    
    for file in sorted(os.listdir("/mnt/Models/")):
        if file.startswith("Encoder_"+nameFile[:-8]):
            loading(file,"De"+file[2:])


# In[ ]:




