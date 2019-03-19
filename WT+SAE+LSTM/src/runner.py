from dataloader import DataLoader
from wt_denoiser import denoise
from sae_model import StackedAutoencoder
import numpy as np
from lstm_model import TsLSTM
import torch.optim as optim
from cross_validator import CrossValidator
import torch
from matplotlib import pyplot as plt


def test():
    lvl = 2
    wavelet = 'Haar'
    ts_file_name = 'ford_ts.csv'
    last_days = 600
    time_frame = 30
    time_bias = 1

    data_loader = DataLoader(ts_file_name, last_days, debug=True)

    raw_data = data_loader.as_matrix()
    ts_data = denoise(raw_data, lvl, wavelet)

    plt.plot(raw_data[3])
    plt.show()
    plt.plot(ts_data[3])
    plt.show()

    daily_features, _ = np.shape(ts_data)
    dataset = data_loader.prepare_dataset_sae(ts_data, time_frame, time_bias)

    runner = Runner(daily_features, hidden_layers_sizes=[10, 10], debug=True)

    cross_validator = CrossValidator()
    validation_loss = cross_validator.run_validation(runner, dataset, sae_epoch=4, lstm_epoch=2)
    print("[RUNNER] Dollars lost={}".format(data_loader.to_dolar(validation_loss)))



class Runner:
    def __init__(self, daily_features,
                 hidden_layers_sizes=[10, 10, 10, 10, 10],
                 gamma=0.2,
                 beta=0,
                 hidden_nodes_activation_rate=1,
                 sae_lr=0.001,
                 delay=4,
                 lstm_lr=0.05,
                 debug=False):

        hidden_layers_sizes.insert(0, daily_features)
        self.sae = StackedAutoencoder(hidden_layers_sizes, gamma=gamma,
                                      beta=beta,
                                      hidden_nodes_activation_rate=hidden_nodes_activation_rate,
                                      lr=sae_lr,
                                      debug=debug)
        self.lstm = TsLSTM(delay, hidden_layers_sizes[-1])
        self.lstm_optimizer = optim.SGD(self.lstm.parameters(), lr=lstm_lr)
        self.debug = debug

    def _train_sae_epoch(self, dataset):
        for x, _ in dataset:
            self.sae(x)

    def _train_lstm_epoch(self, dataset, debug=False):
        total_loss = 0
        for x, target in dataset:
            self.lstm_optimizer.zero_grad()
            res = self.lstm(x)
            loss = self.lstm.criterion(res, target)
            total_loss += loss
            loss.backward()
            self.lstm_optimizer.step()
        if debug:
            print("[LSTM LOSS] average lstm loss on dataset = {}".format(total_loss / len(dataset)))

    def _train_sae(self, dataset, epoch=50):
        self.sae.train()
        for _ in range(epoch):
            self._train_sae_epoch(dataset)

    def _train_lstm(self, dataset, epoch=50):
        self.lstm.train()
        for e in range(epoch):
            if self.debug:
                self._train_lstm_epoch(dataset, debug=True)
            else:
                self._train_lstm_epoch(dataset, debug=False)

    def train(self, dataset, sae_epoch=100, lstm_epoch=50):
        if len(dataset) == 0:
            print("[LSTM LOSS] Empty dataset exiting training!")
            return
        print("[RUNNER] SAE training started")
        self._train_sae(dataset, sae_epoch)
        if self.debug:
            print("[RUNNER] SAE training finished")
        lstm_dataset = []
        self.sae.eval()
        for data, target in dataset:
            compressed_data = self.sae(data)
            lstm_dataset.append((compressed_data, target))
        print("[RUNNER] LSTM training started")
        self._train_lstm(lstm_dataset, lstm_epoch)
        if self.debug:
            print("[RUNNER] LSTM training finished")

    @torch.no_grad()
    def eval_single(self, x):
        self.sae.eval()
        self.lstm.eval()
        x = self.sae(x)
        x = self.lstm(x)
        return x


if __name__ == '__main__':
    test()
