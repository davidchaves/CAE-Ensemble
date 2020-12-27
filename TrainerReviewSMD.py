#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import autograd
from torch.utils.data import DataLoader

import os
from os import listdir
from os.path import isfile, join
import logging
import csv
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import random

import math
import time
from datetime import datetime


# In[2]:


seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


class encoder(nn.Module):

    def __init__(self):
        super(encoder, self).__init__()
        self.linear = nn.Linear(in_features=numberColumns, out_features=linearChannels, bias=True).to(device)
        self.position = nn.Linear(1, linearChannels).to(device)
        self.convolutions = nn.ModuleList([nn.Conv1d(in_channels = convolutionalChannels, 
                                                 out_channels = convolutionalChannels*2, kernel_size = kernelSize, 
                                                 padding = (kernelSize - 1) // 2) for _ in range(5)]).to(device)
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


# In[4]:


class decoder(nn.Module):

    def __init__(self):
        super(decoder, self).__init__()
        self.linear = nn.Linear(in_features=linearChannels, out_features=attentionChannels, bias=True).to(device)
        self.position = nn.Embedding(linearChannels, attentionChannels).to(device)
        self.convolutions = nn.ModuleList([nn.Conv1d(in_channels = convolutionalChannels, 
                                                     out_channels = convolutionalChannels*2, 
                                                     kernel_size = kernelSize) for _ in range(5)]).to(device)
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
            padding = torch.zeros(convolutionInput.size()[0], 128, 3-1).fill_(1).to(device)
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


# In[5]:


lossFunction1 = nn.BCEWithLogitsLoss(reduction='mean')
lossFunction = nn.L1Loss()
lossFunction3 = nn.PoissonNLLLoss()
lossFunction4 = nn.MSELoss()


# In[6]:


def diversityLoss(input, target, iteration):
    sizeInput = input.size()[0]
    normResults = torch.zeros(1, 1).to(device)
    for i in range(0,cycle):
        normResults += math.sqrt(2)/2 * torch.norm(input - outputCycle[i], 2).to(device)/sizeInput

    lossDiversity = abs(lossFunction(input, target) - diversityFactor * normResults)

    return lossDiversity


# In[7]:


def VariableLR(initialLR, currentEpoch, epochPerCycle):
    return initialLR * (np.cos(np.pi * currentEpoch / epochPerCycle) + 1) / 2


# In[8]:


numberEpochs = 10
cycles = 10
beta = 0.9
diversityFactor = 0.7
learningRate = 0.001
linearChannels = 64
convolutionalChannels = 128
attentionChannels = int(convolutionalChannels/2)
kernelSize = 3

filesPath = "./SMD/SMDSeries/"
onlyFiles = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]

for nameFile in onlyFiles:
    importedData = pd.read_csv(filesPath + nameFile,header=None)
    numberColumns = len(importedData.columns)
    
    values = importedData + 1e-7
    meanImported = np.mean(values)
    stdImported = np.std(values)
    normalizedDataset = ((values - meanImported)/stdImported).fillna(0)
    torchDataset = torch.from_numpy(normalizedDataset.values).float().to(device)
    dataloader = DataLoader(torchDataset, batch_size=len(normalizedDataset), shuffle=True)

    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    logging.basicConfig(level=logging.DEBUG, filename="./ModelsSMD/Log_"+ now +".txt", 
                        filemode="a+",format="%(message)s")

    encoderTrain = encoder()
    decoderTrain = decoder()
    encoderOptimizer = torch.optim.Adam(encoderTrain.parameters(), lr=learningRate)
    decoderOptimizer = torch.optim.Adam(decoderTrain.parameters(), lr=learningRate)
    
    dataComplete = autograd.Variable(torchDataset).to(device)
    outputCycle = torch.empty(cycles,torchDataset.size()[0],torchDataset.size()[1],requires_grad=False).to(device)
    
    for cycle in range(cycles):
        logging.info("Cycle # " + str(cycle + 1) + " " + nameFile)

        parametersEncoder = len(list(encoderTrain.parameters()))
        parametersDecoder = len(list(decoderTrain.parameters()))
        genericFeaturesEncoder = random.sample(range(1, parametersEncoder), math.floor(parametersEncoder*beta))
        genericFeaturesDecoder = random.sample(range(1, parametersDecoder), math.floor(parametersDecoder*beta))
        lossesCycle = []
        
        for epoch in range(numberEpochs):
            lr = VariableLR(learningRate, epoch, numberEpochs)
            encoderOptimizer.param_groups[0]['lr'] = lr
            decoderOptimizer.param_groups[0]['lr'] = lr

            index = 0

            for parameterEncoder in encoderTrain.parameters():
                index += 1
                if index in genericFeaturesEncoder:
                    parameterEncoder.requires_grad = True
                else:
                    parameterEncoder.requires_grad = False

            index = 0
            
            for parameterDecoder in decoderTrain.parameters():
                index += 1
                if index in genericFeaturesDecoder:
                    parameterDecoder.requires_grad = True
                else:
                    parameterDecoder.requires_grad = False
                    
            if ((epoch > 3) and (abs(lossesCycle[-1]/lossesCycle[-2])> 10)):
                logging.info("Loss is growing very fast, discard model")
                break
            else: 
                batchCounter = 0

                if(torch.backends.cudnn.version() != None) and (device == 'cuda'):
                    encoderTrain.cuda()
                    decoderTrain.cuda()

                encoderTrain.train()
                decoderTrain.train()

                for dataBatch in dataloader:

                    batchCounter += 1
                    dataBatchTorch = autograd.Variable(dataBatch)
                    convolutionHiddenBatch, residualConnectionBatch = encoderTrain(dataBatchTorch,batchCounter) 
                    batchReconstruction = decoderTrain(convolutionHiddenBatch, residualConnectionBatch)

                    if cycle == 0:
                        reconstruction_loss = lossFunction(torch.squeeze(batchReconstruction), 
                                                            torch.squeeze(dataBatchTorch))
                    else:
                        reconstruction_loss = diversityLoss(torch.squeeze(batchReconstruction), 
                                                             torch.squeeze(dataBatchTorch),batchCounter)

                    decoderOptimizer.zero_grad()
                    encoderOptimizer.zero_grad()
                    reconstruction_loss.backward()
                    decoderOptimizer.step()
                    encoderOptimizer.step()

                encoderTrain.to(device).eval()
                decoderTrain.to(device).eval()

                convolutionHiddenComplete, residualConnectionComplete = encoderTrain(dataComplete.to(device),1)
                reconstructionComplete = decoderTrain(convolutionHiddenComplete, residualConnectionComplete)

                if cycle == 0:
                    lossComplete = lossFunction(torch.squeeze(reconstructionComplete), 
                                                            torch.squeeze(dataComplete))
                else:
                    lossComplete = diversityLoss(torch.squeeze(reconstructionComplete), 
                                                            torch.squeeze(dataComplete),1)

                logging.info('Epoch {} | Learning Rate {:.6f} | Loss {:.6f}'.format(epoch + 1, lr, 
                                                                                    lossComplete.item()))

                lossesCycle.extend([lossComplete.item()])

                if ((epoch + 1) == numberEpochs):
                    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
                    cycleFormat = '{:03}'.format(cycle + 1)
                    epochFormat = '{:03}'.format(epoch + 1)
                    modelName = "{}_cycle_{}_epoch_{}_{}".format(nameFile,cycleFormat,epochFormat,now)
                    encoderName = "Encoder_"+ modelName + ".pth"
                    decoderName = "Decoder_"+ modelName + ".pth"
                    torch.save(encoderTrain.state_dict(), os.path.join("./ModelsSMD", encoderName))
                    torch.save(decoderTrain.state_dict(), os.path.join("./ModelsSMD", decoderName))

        with torch.no_grad():
            outputCycle[cycle] = reconstructionComplete #.squeeze()

