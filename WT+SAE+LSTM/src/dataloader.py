import pandas as pd
import torch


class DataLoader:
    def __init__(self, ts_name):
        path = '../data/' + ts_name
        self._ts = pd.read_csv(path)
        self._fill()

    def _fill(self):
        # MACD index
        self._ts['MACD'] = self._ts["Adj Close"].ewm(span=26).mean() - self._ts['Adj Close'].ewm(
            span=12).mean()

        # CCI index
        constant = 0.015
        period = 20
        tp = (self._ts['High'] + self._ts['Low'] + self._ts['Close']) / 3
        self._ts['CCI'] = (tp - tp.rolling(period).mean()) / (constant * tp.rolling(period).std())

        # ATR index
        period = 14
        ch_cl = self._ts['High'] - self._ts['Low']
        ch_pc = self._ts['High'] - self._ts['Close'].shift(periods=-1)
        cl_pc = self._ts['Low'] - self._ts['Close'].shift(periods=-1)
        tr = pd.concat([ch_cl, ch_pc, cl_pc], axis=1).max(axis=1)
        self._ts['ATR'] = tr.rolling(period).mean()

        # BOLL_H, BOLL_L indices (Bollinger Bands)
        sma = self._ts['Close'].rolling(20).mean()
        std = self._ts['Close'].rolling(20).std()
        self._ts['BOLL_H'] = sma + 2 * std
        self._ts['BOLL_L'] = sma - 2 * std

        # EMA20 index
        self._ts['EMA20'] = self._ts['Close'].ewm(span=20).mean()

        # MA5/10 indices
        self._ts['MA5'] = self._ts['Close'].rolling(5).mean()
        self._ts['MA10'] = self._ts['Close'].rolling(10).mean()

    def as_matrix(self):
        # drop unwanted columns and NaN values (from rolling indices)
        trimmed = self._ts.drop(['Date', 'Adj Close'], axis=1)
        trimmed.drop(trimmed.index[:30], inplace=True)
        #        trimmed.drop(trimmed.index[-30:], inplace=True)

        return trimmed.values.transpose()

    def prepare_dataset_sae(self, data, t, b):
        i = 0
        dataset = []
        _, series_length = data.shape
        while i + t <= series_length:
            full_data = torch.unsqueeze(torch.tensor(data[:, i:i + t]).float().t(),  dim=0)
            price_data = torch.unsqueeze(torch.tensor(data[3, i:i + t]).float(), dim=0)
            dataset.append((full_data, price_data))
            i += b
        return dataset
