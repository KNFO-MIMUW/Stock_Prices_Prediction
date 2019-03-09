import pandas as pd

class Initializer:
    _ts = None

    @staticmethod
    def load(ts_name):
        path = '../data/' + ts_name
        Initializer._ts = pd.read_csv(path)
        Initializer._fill()

    @staticmethod
    def _fill():
        # MACD index
        Initializer._ts['MACD'] = Initializer._ts["Adj Close"].ewm(span=26).mean() - Initializer._ts['Adj Close'].ewm(span=12).mean()

        # CCI index
        constant = 0.015
        period = 20
        tp = (Initializer._ts['High'] + Initializer._ts['Low'] + Initializer._ts['Close']) / 3
        Initializer._ts['CCI'] = (tp - tp.rolling(period).mean()) / (constant * tp.rolling(period).std())

        # ATR index
        period = 14
        ch_cl = Initializer._ts['High'] - Initializer._ts['Low']
        ch_pc = Initializer._ts['High'] - Initializer._ts['Close'].shift(periods=-1)
        cl_pc = Initializer._ts['Low'] - Initializer._ts['Close'].shift(periods=-1)
        tr = pd.concat([ch_cl, ch_pc, cl_pc], axis=1).max(axis=1)
        Initializer._ts['ATR'] = tr.rolling(period).mean()

        # BOLL_H, BOLL_L indices (Bollinger Bands)
        sma = Initializer._ts['Close'].rolling(20).mean()
        std = Initializer._ts['Close'].rolling(20).std()
        Initializer._ts['BOLL_H'] = sma + 2 * std
        Initializer._ts['BOLL_L'] = sma - 2 * std

        # EMA20 index
        Initializer._ts['EMA20'] = Initializer._ts['Close'].ewm(span=20).mean()

        # MA5/10 indices
        Initializer._ts['MA5'] = Initializer._ts['Close'].rolling(5).mean()
        Initializer._ts['MA10'] = Initializer._ts['Close'].rolling(10).mean()

    @staticmethod
    def as_matrix():
        return Initializer._ts.drop(['Date', 'Adj Close'], axis=1).values.transpose()







