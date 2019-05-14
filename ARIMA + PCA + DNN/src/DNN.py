# Authors: Damian Burczyk, Grzegorz Pielot
# Based on Based on "Stock prediction using deep learning" by Ritika Singh and Shashi Srivastava
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import sys

import DataPrep

# constants
data_path = 'F.csv'
close_id = 3
bias_5_id = 33
bias_10_id = 34

# input
result_path = sys.stdin.readline().strip()
reduced_dimension = int(sys.stdin.readline().strip())
frame_size = int(sys.stdin.readline().strip())
adadelta_rho = float(sys.stdin.readline().strip())
adadelta_eps = float(sys.stdin.readline().strip())
layer1_size = int(sys.stdin.readline().strip())
layer2_size = int(sys.stdin.readline().strip())
test_ratio = float(sys.stdin.readline().strip())
epoch_amount = int(sys.stdin.readline().strip())
train_on_tests = sys.stdin.readline().strip() == 'True'
relative_line = sys.stdin.readline().strip() == 'True'
randomize_indices = sys.stdin.readline().strip() == 'True'
predict_by_rate = sys.stdin.readline().strip() == 'True'
date_from = sys.stdin.readline().strip()
date_to = sys.stdin.readline().strip()

if predict_by_rate:
    # id of Close
    predict_id = 4
else:
    # id of Close rate
    predict_id = 3


# Applies PCA twice on a matrix
def pca(matrix):
    pca_model = PCA(n_components=reduced_dimension)
    first_reduction = pca_model.fit_transform(np.transpose(matrix))
    return pca_model.fit_transform(np.transpose(first_reduction))


# Normalizes a matrix
def normalize(matrix):
    dim = matrix.shape
    for i in range(0, dim[0]):
        minimum, maximum = np.amin(matrix[i]), np.amax(matrix[i])
        for j in range(0, dim[1]):
            matrix[i][j] = (matrix[i][j] - minimum) / (maximum - minimum)
    return matrix


# Neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if relative_line:
            self.fully_connected_1 = nn.Linear(reduced_dimension * (1 + reduced_dimension), layer1_size)
        else:
            self.fully_connected_1 = nn.Linear(reduced_dimension * reduced_dimension, layer1_size)
        self.fully_connected_2 = nn.Linear(layer2_size, 1)

    def forward(self, x):
        if relative_line:
            x = x.view(reduced_dimension * (1 + reduced_dimension)).float()
        else:
            x = x.view(reduced_dimension * reduced_dimension).float()
        x = torch.tanh(self.fully_connected_1(x))
        x = self.fully_connected_2(x)
        return x


# Prepares properly transformed window of data
def prepare_data(data_table, index):
    data_frame = np.nan_to_num(data_table[index - frame_size: index])
    normalize(data_frame)
    data_frame = pca(np.nan_to_num(data_frame))
    normalize(data_frame)
    if relative_line:
        data_frame = np.append(data_frame, data_table[index - reduced_dimension: index, predict_id])
    return torch.tensor(np.nan_to_num(data_frame))


# One epoch calculation
def train(train_model, data_table, train_bias, train_split, train_optimizer, train_epoch):
    print("Training epoch {}".format(train_epoch))

    train_model.train()
    if randomize_indices:
        indices = np.random.permutation(range(frame_size, train_split))
    else:
        indices = range(frame_size, train_split)

    for i in indices:
        train_optimizer.zero_grad()

        data_frame = prepare_data(data_table, i)
        output = train_model(data_frame)
        target = data_table[i][predict_id]
        prediction = output.item()

        if predict_by_rate:
            predicted_close = data_table[i - 1][close_id] * prediction + data_table[i - 1][close_id]
        else:
            predicted_close = prediction

        loss = func.mse_loss(target, output)
        loss.backward()
        train_optimizer.step()

        new_bias = predicted_close - target
        data_table[i][bias_5_id] += (new_bias - train_bias[i])/5
        data_table[i][bias_10_id] += (new_bias - train_bias[i])/10
        train_bias[i] = new_bias

    return


# Creates forecast, doesn't train on tests
def test_eval(test_model, data_table, test_bias, test_split):
    print("Testing")

    test_model.eval()
    test_forecast = []
    indices = range(test_split, data_table.shape[0])

    with torch.no_grad():
        for i in indices:
            data_frame = prepare_data(data_table, i)
            output = test_model(data_frame)
            target = data_table[i][predict_id]
            prediction = output.item()

            if predict_by_rate:
                predicted_close = data_table[i - 1][close_id] * prediction + data_table[i - 1][close_id]
            else:
                predicted_close = prediction

            test_forecast.append(predicted_close)

            new_bias = predicted_close - target
            data_table[i][bias_5_id] += (new_bias - test_bias[i]) / 5
            data_table[i][bias_10_id] += (new_bias - test_bias[i]) / 10
            test_bias[i] = new_bias

    return test_forecast


# Creates forecast, trains on tests
def test_train(test_model, data_table, test_bias, test_split):
    print('Testing')

    test_model.train()
    test_forecast = []
    indices = range(test_split, data_table.shape[0])

    for i in indices:
        optimizer.zero_grad()

        data_frame = prepare_data(data_table, i)
        output = test_model(data_frame)
        target = data_table[i][predict_id]
        prediction = output.item()

        if predict_by_rate:
            predicted_close = data_table[i - 1][close_id] * prediction + data_table[i - 1][close_id]
        else:
            predicted_close = prediction

        test_forecast.append(predicted_close)

        loss = func.mse_loss(target, output)
        loss.backward()
        optimizer.step()

        new_bias = predicted_close - target
        data_table[i][bias_5_id] += (new_bias - test_bias[i])/5
        data_table[i][bias_10_id] += (new_bias - test_bias[i])/10
        test_bias[i] = new_bias

    return test_forecast


# input data
raw = pd.read_csv(data_path, header=0, parse_dates=True, index_col=0, squeeze=True, date_parser=pd.to_datetime)
data = DataPrep.datatable(data_path)

# variables
split = int(data.shape[0] * (1.0 - test_ratio))
model = Net()
optimizer = optim.Adadelta(model.parameters(), rho=adadelta_rho, eps=adadelta_eps)
bias = np.zeros(data.shape[0], dtype=int)

# Model training
for epoch in range(1, epoch_amount + 1):
    train(model, data, bias, split, optimizer, epoch)

# Making a forecast
if train_on_tests:
    forecast = test_train(model, data, bias, split)
else:
    forecast = test_eval(model, data, bias, split)

test_set = data[split:, close_id]
mse = mean_squared_error(test_set, forecast)


# output
result = pd.DataFrame({'Close': test_set, 'Forecast': forecast},
                      index=raw.truncate(before=date_from, after=date_to).index[-len(forecast):])
result.to_csv(path_or_buf=result_path + '.csv')

report = open(result_path + '.txt', 'w+')

report.write('Model = PCA+DNN\n')

report.write('\nINPUT:\n')
report.write('date from = %s\n' % date_from)
report.write('date to = %s\n' % date_to)
report.write('test ratio = %f\n' % test_ratio)
report.write('number of epochs = %d\n' % epoch_amount)
report.write('frame size = %d\n' % frame_size)
report.write('matrix dimensions after PCA = %d x %d\n' % (reduced_dimension, reduced_dimension))
report.write('train on tests = %s\n' % str(train_on_tests))
report.write('use relative line in processed data frame = %s\n' % str(relative_line))
report.write('randomize training indices = %s\n' % str(randomize_indices))
report.write('predict by rate of close instead = %s\n' % str(predict_by_rate))
report.write('ADADELTA parameters (rho, epsilon) = (%f, %e)\n' % (adadelta_rho, adadelta_eps))
report.write('layer 1 size = %s\n' % str(layer1_size))
report.write('layer 2 size = %s\n' % str(layer2_size))

report.write('\nOUTPUT:\n')
report.write('mean square error = %f\n' % mse)

report.close()
