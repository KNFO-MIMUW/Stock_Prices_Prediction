# Authors: Damian Burczyk, Grzegorz Pielot
# Based on Based on "Stock prediction using deep learning" by Ritika Singh and Shashi Srivastava
import pandas as pd
import numpy as np
from stockstats import StockDataFrame

pd.options.mode.chained_assignment = None


def datatable(data_path):
    # input data
    raw = pd.read_csv(data_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=pd.to_datetime)
    size = raw.shape[0]

    # I28
    i28 = ((np.array(raw['Close'][1:size]) - np.array(raw['Close'][0:size - 1])) / np.array(raw['Close'][0:size - 1]))
    raw['I28'] = np.concatenate(([0], i28))

    # MA5, MA10, MA20
    raw['MA5'] = raw['Close'].rolling(window=5).mean()
    raw['MA10'] = raw['Close'].rolling(window=10).mean()
    raw['MA20'] = raw['Close'].rolling(window=20).mean()

    # Exponential Moving Average
    ema12 = raw['Close'].ewm(span=12, min_periods=0, adjust=False, ignore_na=False).mean()
    ema26 = raw['Close'].ewm(span=26, min_periods=0, adjust=False, ignore_na=False).mean()
    raw['DIFF'] = ema12 - ema26

    # Boolinger
    std20 = raw.Close.rolling(window=20).std()
    bub = raw['MA20'] + (std20 * 2)
    bul = raw['MA20'] - (std20 * 2)

    raw['BU'] = (raw['Open'] - bub) / bub
    raw['BL'] = (raw['Open'] - bul) / bul

    # Stochastic Fast
    window_size = 20

    raw['K'] = ((raw['Close'] - raw['Low'].rolling(window=window_size).min()) / (
            raw['High'].rolling(window=window_size).max() - raw['Low'].rolling(window=window_size).min())) * 100

    raw['D'] = raw['K'].rolling(window=3).mean()

    # Price Rate of Change
    raw['ROC'] = pd.Series(((raw['Close'].diff(20 - 1) / raw['Close'].shift(20 - 1)) * 100), name='ROC_' + str(20))

    # Momentum
    momentum6 = np.sign(np.array(raw['Close'][6:size]) - np.array(raw['Close'][0:size - 6]))
    raw['MTM6'] = np.concatenate(([0, 0, 0, 0, 0, 0], momentum6))

    momentum12 = np.sign(np.array(raw['Close'][12:size]) - np.array(raw['Close'][0:size - 12]))
    raw['MTM12'] = np.concatenate(([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], momentum12))

    # Williams Index
    stock = StockDataFrame.retype(pd.read_csv(data_path))
    raw['WR5'] = stock['wr_5']
    raw['WR10'] = stock['wr_10']

    # True Range
    raw['TR'] = stock['tr']

    # RSI
    raw['RSI6'] = stock['rsi_6']
    raw['RSI12'] = stock['rsi_12']

    # Oscillator
    raw['OSC6'] = raw['Close'] - raw['Close'].rolling(window=6).mean()
    raw['OSC12'] = raw['Close'] - raw['Close'].rolling(window=12).mean()

    # I26
    i26 = np.array(raw['K'][1:size]) - np.array(raw['K'][0:size - 1])
    raw['I26'] = np.concatenate(([0], i26))

    # I27
    i27 = np.array(raw['D'][1:size]) - np.array(raw['D'][0:size - 1])
    raw['I27'] = np.concatenate(([0], i27))

    # I29
    raw['I29'] = (raw['Close'] - raw['Open']) / raw['Open']

    # I30
    raw['I30'] = (raw['Close'] - raw['Low']) / (raw['High'] - raw['Low'])

    # I31
    i31 = ((np.array(raw['MA5'][1:size]) - np.array(raw['MA5'][0:size - 1])) / np.array(raw['MA5'][0:size - 1]))
    raw['I31'] = np.concatenate(([0], i31))

    # I32
    i32 = ((np.array(raw['MA20'][1:size]) - np.array(raw['MA20'][0:size - 1])) / np.array(raw['MA20'][0:size - 1]))
    raw['I32'] = np.concatenate(([0], i32))

    # I33
    size = raw.shape[0]
    i33 = ((np.array(raw['MA5'][1:size]) - np.array(raw['MA20'][0:size - 1])) / np.array(raw['MA20'][0:size - 1]))
    raw['I33'] = np.concatenate(([0], i33))

    # I34
    raw['I34'] = (raw['Close'] - raw['MA20']) / raw['MA20']

    # I35
    d = raw['Close']
    i35 = [0]
    minimum = d[0]
    for i in range(1, len(d)):
        i35.append((d[i] - minimum) / (min(minimum, d[i])))
        minimum = min(minimum, d[i])
    raw['I35'] = i35

    # I36
    d = raw['Close']
    i36 = [0]
    maximum = d[0]
    for i in range(1, len(d)):
        i36.append((d[i] - maximum) / (max(maximum, d[i])))
        maximum = max(maximum, d[i])
    raw['I36'] = i36

    # BIAS
    raw['BIAS5'] = 0
    raw['BIAS10'] = 0

    # formatting
    raw = raw.truncate(before='1988-01-04', after='2019-06-30')
    raw.drop(columns=['Volume'], inplace=True)
    raw.drop(columns=['Adj Close'], inplace=True)
    return np.array(raw)
