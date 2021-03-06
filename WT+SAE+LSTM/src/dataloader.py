import pandas as pd
import torch
import numpy as np


class DataLoader:
    PRICE_INDEX = 3

    def __init__(self, ts_name, last_days=-1, debug=False):
        self.max = []
        self.min = []
        self.debug = debug

        path = '../data/' + ts_name
        self._ts = pd.read_csv(path)

        if last_days != -1:
            self._ts = self._ts.tail(last_days)

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

        print("[DATA LOADER] Filled dataset.")

    def as_matrix(self):
        # drop unwanted columns and NaN values (from rolling indices)
        trimmed = self._ts.drop(['Date', 'Adj Close'], axis=1)
        trimmed.drop(trimmed.index[:30], inplace=True)

        if self.debug:
            time_frames, daily_features = self._ts.shape #o 2 za duzo
            print("""[DATA LOADER] Trimmed dataset created:
            time_frames     {}
            daily features  {}""".format(time_frames, daily_features-2))
        return trimmed.values.transpose()

    def _normalize_data(self, data):
        ndata = []
        for index in data:
            maxi = max(index)
            mini = min(index)
            self.max.append(maxi)
            self.min.append(mini)
            if maxi == mini:
                nindex = [0 for _ in index]
            else:
                nindex = [(x - mini) / (maxi - mini) for x in index]
            ndata.append(nindex)
        return np.array(ndata)

    def prepare_dataset_sae(self, data, t, b):
        i = 0
        dataset = []
        data = self._normalize_data(data)
        _, series_length = data.shape
        while i + t + 1 <= series_length:
            full_data = torch.unsqueeze(torch.tensor(data[:, i:i + t]).float().t(), dim=0)
            price_data = torch.unsqueeze(
                torch.tensor(data[self.PRICE_INDEX, i + 1:i + t + 1]).float(), dim=0)
            dataset.append((full_data, price_data))
            i += b
        return dataset

    def denormalize_data(self, data):
        ndata = []
        for ctr, index in enumerate(data):
            nindex = [self.to_dolar(x, ctr) for x in index]
            ndata.append(nindex)
        return np.array(ndata)

    def to_dolar(self, x, idx=PRICE_INDEX):
        maxi = self.max[idx]
        mini = self.min[idx]

        if maxi == mini:
            return maxi
        else:
            return x * (maxi - mini) + mini
