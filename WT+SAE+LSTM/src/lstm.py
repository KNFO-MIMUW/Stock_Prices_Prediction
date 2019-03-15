import torch
import torch.nn as nn
from initializer import Initializer
import numpy as np

class LSTM_model(nn.Module):
    def __init__(self, ninp, nhid, nlayers):
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(ninp, nhid, nlayers)

    def forward(self, input):
        output, _ = self.lstm(input)
        return output[-1]

# jakieś syfy testowe
# def prepareData (data, windowSize, batchSize):
#     test = data[:, :800]
#     target = data[:, 800:1000]
#
#     testSeqs = np.empty([800-windowSize+1, 1, windowSize])
#     testTargets = np.empty([800-windowSize+1, 1, 1])
#
#     for i in range(800-windowSize+1):
#         testSeqs[i] = test[3,i:i+windowSize]
#         testTargets[i] = test[3, i+windowSize-1]
#
#     for i in range(1000/batchSize):
#         for j in range(batchSize):
#
#
#
# Initializer.load("test.csv")
# data = Initializer.as_matrix()
#
# net = LSTM_model(1, 1, 1)
# sample = torch.from_numpy(data[:,:4])
# test = torch.from_numpy(data[:,1:5])
# sample = sample.contiguous().view(4, -1, 13)
# print(sample)
# output = net(sample.float())
# print(output)
#
# prepareData(data, 20, 1)