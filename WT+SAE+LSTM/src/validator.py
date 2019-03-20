import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Validator:
    def __init__(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        assert (train_ratio + valid_ratio + test_ratio == 1)

    def _get_training(self, dataset):
        new_length = int(len(dataset) * self.train_ratio)
        return dataset[:new_length]

    def _get_validation(self, dataset):
        ldata = len(dataset)
        train_end = int(ldata * self.train_ratio) #zmieniłem bo te końce poprzednie to były złe, przynajmniej tak mi sie wydaje xD
        validation_end = int(ldata * (self.train_ratio + self.valid_ratio))

        return dataset[train_end:validation_end]

    def _get_test(self, dataset):
        ldata = len(dataset)
        test_beg = int(ldata * (self.train_ratio + self.valid_ratio))

        return dataset[test_beg:]

    #TODO temporary function for debug
    def _eval_and_plot(self, dataset, runner):
        errors = []
        pred_values = []
        target_values = []

        for x, t in dataset:
            pred_t = runner.eval_single(x)

            pred_t = pred_t.squeeze()
            t = t.squeeze()

            errors.append(t[-1] - pred_t[-1])

            target_values.append(t[-1])
            pred_values.append(pred_t[-1])

        comparation = np.array([pred_values, target_values])
        comparation = pd.DataFrame(data=comparation.transpose(), columns=['predicted', 'target'])
        comparation.plot()
        plt.show()

        pred_target = zip(pred_values, target_values)

        return errors, pred_target

    def train_and_eval_hard(self, runner, training_dataset, evaluation_dataset, sae_epoch, lstm_epoch):
        runner.train(training_dataset, sae_epoch, lstm_epoch)

        _ = self._eval_and_plot(training_dataset, runner)
        errors, pred_target = self._eval_and_plot(evaluation_dataset, runner)

        cross_errors_square_norm = np.linalg.norm(np.array(errors))
        print(
            """[CROSS-VALIDATION] Loss on validation set cross_errors_square_norm={}""".format(
                cross_errors_square_norm))

        return pred_target

    def run_validation(self, runner, dataset, sae_epoch=100, lstm_epoch=50):
        training_dataset = self._get_training(dataset)
        validation_dataset = self._get_validation(dataset)

        return self.train_and_eval_hard(runner, training_dataset, validation_dataset, sae_epoch, lstm_epoch)

    def run_test(self, runner, dataset, sae_epoch, lstm_epoch):
        training_dataset = self._get_training(dataset) + self._get_validation(dataset)
        test_dataset = self._get_test(dataset)

        return self.train_and_eval_hard(runner, training_dataset, test_dataset, sae_epoch, lstm_epoch)
