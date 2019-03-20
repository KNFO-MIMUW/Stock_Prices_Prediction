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
    lvl = 1
    wavelet = 'db4'  # Haar'
    ts_file_name = 'ford_ts.csv'
    last_days = 1200
    time_frame = 60
    time_bias = 1

    data_loader = DataLoader(ts_file_name, last_days, debug=True)

    raw_data = data_loader.as_matrix()
    ts_data = denoise(raw_data, lvl, wavelet)

    # plt.plot(raw_data[3])
    # plt.show()
    # plt.plot(ts_data[3])
    # plt.show()

    daily_features, _ = np.shape(ts_data)
    dataset = data_loader.prepare_dataset_sae(ts_data, time_frame, time_bias)

    runner = Runner(daily_features, lstm_layers=1, gamma=0.005, delay=4, sae_lr=0.01, beta=0,
                    hidden_nodes_activation_rate=0.9, hidden_layers_sizes=[8], debug=True)

    cross_validator = CrossValidator()
    pred_target = cross_validator.run_validation(runner, dataset, sae_epoch=1, lstm_epoch=1)
    pred_target_dollars = [(data_loader.to_dolar(x), data_loader.to_dolar(y)) for x, y in pred_target]
    dollars_loss = sum([abs(x - y) for x, y in pred_target_dollars])
    print("[RUNNER] Dollars lost={}".format(dollars_loss))


class Runner:
    def __init__(self, daily_features,
                 hidden_layers_sizes=[10, 10, 10, 10, 10],
                 gamma=0.2,
                 beta=0.0,
                 hidden_nodes_activation_rate=0.999,
                 sae_lr=0.001,
                 delay=4,
                 lstm_lr=0.05,
                 lstm_layers=1,
                 debug=False):

        hidden_layers_sizes.insert(0, daily_features)
        self.sae = StackedAutoencoder(hidden_layers_sizes, gamma=gamma,
                                      beta=beta,
                                      hidden_nodes_activation_rate=hidden_nodes_activation_rate,
                                      lr=sae_lr,
                                      debug=debug)
        self.lstm = TsLSTM(delay, hidden_layers_sizes[-1], lstm_layers)
        self.lstm_optimizer = optim.SGD(self.lstm.parameters(), lr=lstm_lr)
        self.debug = debug

    def _train_sae_epoch(self, dataset):
        for x, _ in dataset:
            self.sae(x)

    def _train_lstm_epoch(self, dataset, epoch_number=-1, debug=False):
        total_loss = 0
        for x, target in dataset:
            self.lstm_optimizer.zero_grad()
            res = self.lstm(x)
            loss = self.lstm.criterion(res, target)
            total_loss += loss
            loss.backward()
            self.lstm_optimizer.step()
        if debug:
            print("[LSTM LOSS] Epoch {}: average lstm loss on dataset = {}".format(epoch_number,
                                                                                   total_loss / len(
                                                                                       dataset)))

    def _train_sae(self, dataset, epoch=50):
        self.sae.train()
        for _ in range(epoch):
            self._train_sae_epoch(dataset)

    def _train_lstm(self, dataset, epoch=50):
        self.lstm.train()
        for e in range(epoch):
            if self.debug:
                self._train_lstm_epoch(dataset, epoch_number=e, debug=True)
            else:
                self._train_lstm_epoch(dataset, epoch_number=e, debug=False)

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
