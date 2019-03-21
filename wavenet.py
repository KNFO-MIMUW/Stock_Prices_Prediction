"""
    Time series forecasting based on:
    1. A. Borovykh, S. Bohte, and C. W. Oosterlee, "Conditional Time Series Forecasting with Convolutional
    Neural Networks", arXiv:1703.04691, Mar. 2017.
    2. Google DeepMind, "Wavenet: A generative model for raw audio", arXiv:1609.03499, Sep. 2016.
    3.  K. Papadopoulos, "SeriesNet: A Dilated Casual Convolutional Neural Network for Forecasting" Apr. 2018

    Requirements:
         tensorflow==1.10.1
         Keras==2.2.2
         numpy==1.14.5
         matplotlib==3.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from keras.layers import Conv1D, Input, Add, Activation, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from keras import optimizers

###
### DATA PREPARATION
###

# Read ford's data.
ford_data = pd.read_csv('f_us_d.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
ford_data = ford_data.truncate(before="2012-01-01", after="2015-12-31")
series_ford = ford_data[['Open']]

# Read s&p's data
# stock_data = pd.read_csv('s&p_d.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# stock_data = stock_data.truncate(before="2000-01-01", after="2003-12-31")
# series_stock_high = stock_data[['High']]
# series_stock_low = stock_data[['Low']]


TEST_RATIO = 0.80

vseries = series_ford.values
split_val = int(TEST_RATIO * vseries.size)
dataset, tests = vseries[:split_val], vseries[split_val:]
dataset = [x[0] for x in dataset]
tests = [x[0] for x in tests]
tests = np.array(tests)

# v_high, v_low = series_stock_high.values, series_stock_low.values
# split_val_h, split_val_l = int(TEST_RATIO * v_high.size), int(TEST_RATIO * v_low.size)
# dataset_high, dataset_low = v_high[:split_val_h], v_low[:split_val_l]
# dataset_high, dataset_low = [x[0] for x in dataset_high], [x[0] for x in dataset_low]


###
### BUILD MODEL
###

def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        residual = input_

        layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length,
                           dilation_rate=dilation,
                           activation='relu', padding='causal', use_bias=False,
                           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                            seed=11), kernel_regularizer=l2(l2_layer_reg))(input_)

        layer_out = Activation('relu')(layer_out)

        skip_out = Conv1D(1, 1, activation='relu', use_bias=False,
                          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                            seed=11), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_in = Conv1D(1, 1, activation='relu', use_bias=False,
                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                            seed=11), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_out = Add()([residual, network_in])

        return network_out, skip_out

    return f


def DC_CNN_Model(length):
    input = Input(shape=(length, 1))

    l1a, l1b = DC_CNN_Block(32, 2, 1, 0.001)(input)
    l2a, l2b = DC_CNN_Block(32, 2, 2, 0.001)(l1a)
    l3a, l3b = DC_CNN_Block(32, 2, 4, 0.001)(l2a)
    l4a, l4b = DC_CNN_Block(32, 2, 8, 0.001)(l3a)
    l5a, l5b = DC_CNN_Block(32, 2, 16, 0.001)(l4a)
    l6a, l6b = DC_CNN_Block(32, 2, 32, 0.001)(l5a)
    l7a, l7b = DC_CNN_Block(32, 2, 64, 0.001)(l6a)
    # l7b = Dropout(0.1)(l7b)
    l8a, l8b = DC_CNN_Block(32, 2, 128, 0.001)(l7a)
    # l8b = Dropout(0.2)(l8b)
    # l9a, l9b = DC_CNN_Block(32, 2, 256, 0.001)(l8a)
    # l9b = Dropout(0.3)(l9b)
    # l10a, l10b = DC_CNN_Block(32, 2, 512, 0.001)(l9a)
    # l10b = Dropout(0.6)(l10b)

    lx = Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b, l8b])#, l9b])#, l10b])

    lf = Activation('relu')(lx)

    output_l = Conv1D(1, 1, activation='relu', use_bias=False,
                 kernel_initializer=TruncatedNormal(mean=0, stddev=0.05, seed=11),
                 kernel_regularizer=l2(0.001))(lf)

    model = Model(input=input, output=output_l)

    adam = optimizers.Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None,
                           decay=0.0, amsgrad=False)

    model.compile(optimizer=adam, loss='mae', metrics=['mse'])

    return model


def evaluate_timeseries(timeseries, predict_size):

    timeseries = timeseries[~pd.isna(timeseries)]

    length = len(timeseries) - 1

    timeseries = np.atleast_2d(np.asarray(timeseries))
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T

    model = DC_CNN_Model(length)
    print('\nInput size: {}\nOutput size: {}'.
          format(model.input_shape, model.output_shape))

    model.summary()

    X = timeseries[:-1].reshape(1, length, 1)
    y = timeseries[1:].reshape(1, length, 1)

    model.fit(X, y, epochs=1000)

    pred_array = np.zeros(predict_size).reshape(1, predict_size, 1)
    observation = np.zeros(predict_size).reshape(1, predict_size, 1)
    X_test_initial = timeseries[1:].reshape(1, length, 1)
    for i in range(predict_size):
        observation[:, i, :] = tests[i]

    pred_array[:, 0, :] = model.predict(X_test_initial)[:, -1:, :]

    # Forecast is based on the observations up to previous day.
    for i in range(predict_size - 1):
        pred_array[:, i + 1:, :] = model.predict(
            np.append(X_test_initial[:, i + 1:, :], observation[:, :i + 1, :])
                .reshape(1, length, 1), batch_size=32)[:,-1:, :]

        # stddev = (dataset_high[i + 1] - dataset_low[i + 1]) / dataset_high[i + 1]
        # pred_array[:, i + 1:, :] = model.predict(np.append(X_test_initial[:, i + 1:, :], np.random.normal(observation[:, :i + 1, :], stddev)).reshape(1, length, 1))[:, -1:, :]

        # Forecast for the next day based on the predictions.
        #
        # pred_array[:, i + 1:, :] = model.predict(np.append(X_test_initial[:, i + 1:, :],
        #                                                   pred_array[:, :i + 1, :])
        #                                                   .reshape(1, length, 1))[:, -1:, :]

    return pred_array.flatten()


predictions = evaluate_timeseries(np.array(dataset), len(tests))

mae = mean_absolute_error(tests, predictions)

print('Loss (mean absolute error): ', mae)

# plt.plot(dataset + tests, color='yellow')
# plt.plot(dataset, color='orange')

plt.plot(tests, color='orange', label='True data')
plt.plot(predictions, label= 'Predictions')
plt.legend(loc='upper left')
plt.xlabel('Days')
plt.ylabel('Price')

plt.show()
