import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence


# TODO starting weights
# Based on:
# https://github.com/ShayanPersonal/stacked-autoencoder-pytorch


class Autoencoder(nn.Module):
    LOSS_TRD = 1000

    def __init__(self, input_size,
                 hidden_size,
                 gamma=0.2,
                 beta=0,
                 hidden_nodes_activation_rate=1,
                 lr=0.001,
                 debug=False):
        super(Autoencoder, self).__init__()
        self.iter_ctr = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.beta = beta
        self.debug = debug
        self.hidden_nodes_distribution = Bernoulli(torch.tensor([hidden_nodes_activation_rate]))

        self.forward_pass = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
        )
        self.backward_pass = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def criterion(self, input, target, hidden):
        out_diff = (1 / 2) * torch.norm(target - input, dim=2).sum()

        w1 = self.forward_pass[0].weight
        w2 = self.backward_pass[0].weight
        weight_decay = (1 / 2) * (
                torch.norm(w1, p='fro') ** 2 + torch.norm(w2, p='fro') ** 2)
        weight_decay *= self.gamma

        current_hidden_nodes_activation_rate = torch.mean(hidden, dim=1)
        sparse_penalty_term = torch.tensor([0.0])

        if self.beta != 0:
            for batch in current_hidden_nodes_activation_rate:
                for node in batch:
                    node_distribution = Bernoulli(node)
                    sparse_penalty_term += kl_divergence(node_distribution,
                                                         self.hidden_nodes_distribution)
            sparse_penalty_term *= self.beta

        if self._is_debug():
            print(
                """[LOSS INFO] Autoencoder [i:{}, h:{}] (iter: {}th):
            out_diff                {}
            weight_decay            {}
            sparse_penalty_term     {}\n""".format(
                    self.input_size,
                    self.hidden_size,
                    self.iter_ctr,
                    out_diff,
                    weight_decay,
                    sparse_penalty_term[0]  # TODO remove [0]
                )
            )

        return out_diff + weight_decay + sparse_penalty_term

    def forward(self, x):
        """
        :param x: tensor of size N x T x P  where N is batch size, P are daily features, T is time
        series length
        :return: tensor of size N x T x C where N is batch size, P are compresed daily
        features, T is time series length
        """
        x = x.detach()

        y = self.forward_pass(x)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iter_ctr = self.iter_ctr + 1

        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)

    def _is_debug(self):
        return self.debug and self.iter_ctr % self.LOSS_TRD == 1


class StackedAutoencoder(nn.Module):
    def __init__(self, layers_sizes,
                 gamma=0.2,
                 beta=0,
                 hidden_nodes_activation_rate=1,
                 lr=0.001,
                 debug=False):
        super(StackedAutoencoder, self).__init__()
        self.debug = debug
        self.layers_sizes_pairs = list(zip(layers_sizes, layers_sizes[1:]))

        if not all(e >= l for e, l in self.layers_sizes_pairs[1:]):
            raise Exception("Hidden layers sizes aren't non-increasing!")

        self.layers = nn.ModuleList()
        for input, hidden in self.layers_sizes_pairs:
            layer = Autoencoder(input, hidden, debug=debug, gamma=gamma, beta=beta,
                                hidden_nodes_activation_rate=hidden_nodes_activation_rate, lr=lr)
            self.layers.append(layer)
        if self.debug:
            print("-" * 70)
            print("""[INITIALIZATION] StackedAutoencoder with parameters: 
            layers                         {}
            gamma                          {}
            beta                           {}
            hidden_nodes_activation_rate   {}
            debug                          {}""".format(
                self.layers_sizes_pairs,
                gamma,
                beta,
                hidden_nodes_activation_rate,
                debug))
            print("-" * 70)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reconstruct(self, x):
        for layer in reversed(list(self.layers)):
            x = layer.reconstruct(x)
        return x


def testAE():
    epoch = 100000
    tensor_length = 12
    batches = 10
    ae = Autoencoder(tensor_length, 7, debug=True)

    ae.train()
    x = torch.randn(batches, 1, tensor_length)
    for _ in range(epoch):
        ae.forward(x)

    ae.eval()
    y = ae.forward(x)
    x_recon = ae.reconstruct(y)

    print("-" * 6 + "AE test" + "-" * 6)
    print('x: {}'.format(x))
    print('x_recon: {}'.format(x_recon))


def testSAE():
    epoch = 100000
    tensor_length = 12
    batches = 10
    sae = StackedAutoencoder([tensor_length, 7, 6], debug=True)

    sae.train()

    x = torch.randn(batches, 1, tensor_length)
    for _ in range(epoch):
        sae.forward(x)

    sae.eval()
    y = sae.forward(x)
    x_recon = sae.reconstruct(y)
    print("-" * 6 + "SAE test" + "-" * 6)
    print('x: {}'.format(x))
    print('x_recon: {}'.format(x_recon))


def testAE_series():
    epoch = 20000
    tensor_length = 12
    batches = 1
    series_lenghth = 15
    ae = Autoencoder(tensor_length, 5, debug=True)

    ae.train()

    x = torch.randn(1, tensor_length, 1)
    d = x
    for _ in range(series_lenghth - 1):
        d = torch.cat([x, d], 2)
    x = torch.transpose(d, 1, 2)
    for _ in range(epoch):
        ae.forward(x)

    ae.eval()
    y = ae.forward(x)
    x_recon = ae.reconstruct(y)
    print("-" * 6 + "AE series test" + "-" * 6)
    print('x: {}'.format(x))
    print('x_recon: {}'.format(x_recon))


def testAE_divergence():
    epoch = 20000
    tensor_length = 12
    batches = 1
    series_lenghth = 15
    ae = Autoencoder(tensor_length, 5, debug=True, hidden_nodes_activation_rate=0.1, gamma=0.1,
                     beta=100)

    ae.train()

    x = torch.randn(1, tensor_length, 1)
    d = x
    for _ in range(series_lenghth - 1):
        d = torch.cat([x, d], 2)
    x = torch.transpose(d, 1, 2)
    for _ in range(epoch):
        ae.forward(x)

    ae.eval()
    y = ae.forward(x)
    x_recon = ae.reconstruct(y)
    print("-" * 6 + "AE divergence test" + "-" * 6)
    print('x: {}'.format(x))
    print('x_recon: {}'.format(x_recon))


def testSAE_final():
    """ In this test we want to detect pattern [0,x,2x,...,nx]"""
    epoch = 5000
    tensor_length = 12
    batches = 3
    series_length = 15
    hnar = 0.2
    h1 = int(2 / hnar)
    h2 = int(2 / hnar)
    ae = StackedAutoencoder([tensor_length, h1, h2], debug=True, hidden_nodes_activation_rate=hnar,
                            gamma=1, beta=1)

    ae.train()

    for i in range(batches):
        for j in range(series_length - 1):
            x = torch.randn(1)
            x = torch.tensor([[abs(x[0]) * j for j in range(tensor_length)]])
            x = x[:, :, None]

            if j != 0:
                d = torch.cat([x, d], 2)
            else:
                d = x

        x = torch.transpose(d, 1, 2)
        if i != 0:
            e = torch.cat([e, x], 0)
        else:
            e = x
    for _ in range(epoch):
        ae.forward(e)

    ae.eval()

    x = torch.randn(1)
    x = torch.tensor([[abs(x[0]) * j for j in range(tensor_length)]])
    x = x[:, :, None]
    x = torch.transpose(x, 1, 2)

    y = ae.forward(x)
    x_recon = ae.reconstruct(y)
    print("-" * 6 + "SAE final test" + "-" * 6)
    print('x: {}'.format(x))
    print('x_recon: {}'.format(x_recon))


if __name__ == '__main__':
    testSAE_final()
