import torch
import torch.nn as nn
from torch import optim


class TsLSTM(nn.Module):
    def __init__(self, delay, daily_features, nlayers=1):
        super(TsLSTM, self).__init__()
        self.delay = delay
        self.lstm = nn.LSTM(daily_features, 1, nlayers, batch_first=True)

    def criterion(self, input, target):
        input = torch.squeeze(input) #TODO batching
        target = torch.squeeze(target)
        return torch.norm(input[self.delay:] - target[self.delay:])

    def forward(self, input):
        output, _ = self.lstm(input)
        return output

def test():
    n = 40
    epoch = 1000
    x = torch.tensor([[-2 for _ in range(n)]])
    x = torch.unsqueeze(x, 2).float()

    z = x

    lstm = TsLSTM(2, 1, 2)
    optimizer = optim.SGD(lstm.parameters(), lr=0.1)
    lstm.train()
    for e in range(epoch):
        optimizer.zero_grad()
        y = lstm(x)
        loss = lstm.criterion(y, z)
        loss.backward()
        if e % 100 == 0:
            for p in lstm.parameters():
                    print('===========\ngradient:{}\n'.format, p.grad)
            print("loss={}".format(loss))
        optimizer.step()

    lstm.eval()
    y = lstm(x)
    print(y)



if __name__ == "__main__":
    test()