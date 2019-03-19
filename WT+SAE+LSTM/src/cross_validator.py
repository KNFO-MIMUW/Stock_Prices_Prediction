import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class CrossValidator:
    def __init__(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        assert (train_ratio + valid_ratio + test_ratio == 1)

    def _get_training(self, dataset):
        new_length = int(len(dataset) * self.test_ratio)
        return dataset[:new_length]

    def _get_validation(self, dataset):
        ldata = len(dataset)
        train_end = int(ldata * self.train_ratio) #zmieniłem bo te końce poprzednie to były złe, przynajmniej tak mi sie wydaje xD
        validation_end = int(ldata * (self.train_ratio + self.valid_ratio))

        return dataset[train_end:validation_end]

    def run_validation(self, runner, dataset, epoch_train=100):
        training_dataset = self._get_training(dataset)
        runner.train(training_dataset, epoch_train)

        validation_dataset = self._get_validation(dataset)
        errors = []
        pred_values = []
        target_values = []
        for x, t in validation_dataset:
            pred_t = runner.eval_single(x)
            t = t.squeeze()
            target_values.append(t[-1])
            pred_t = pred_t.squeeze()
            pred_values.append(pred_t[-1])
            errors.append(t[-1] - pred_t[-1])

        comparation = np.array([pred_values, target_values])
        comparation = pd.DataFrame(data=comparation.transpose(), columns=['predicted', 'target'])
        comparation.plot()
        plt.show()

        return np.linalg.norm(np.array(errors))
