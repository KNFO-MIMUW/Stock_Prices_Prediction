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
        new_length = int(len(dataset) * self.train_ratio)
        return dataset[:new_length]

    def _get_validation(self, dataset):
        ldata = len(dataset)
        train_end = int(ldata * self.train_ratio) #zmieniłem bo te końce poprzednie to były złe, przynajmniej tak mi sie wydaje xD
        validation_end = int(ldata * (self.train_ratio + self.valid_ratio))

        return dataset[train_end:validation_end]

    #TODO temporary function for debug
    def _eval_plot(self, dataset, runner):
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

        return errors

    def run_validation(self, runner, dataset, sae_epoch=100, lstm_epoch=50):
        training_dataset = self._get_training(dataset)
        runner.train(training_dataset, sae_epoch, lstm_epoch)

        validation_dataset = self._get_validation(dataset)

        _ = self._eval_plot(training_dataset, runner)
        errors = self._eval_plot(validation_dataset, runner)

        cross_errors_sum = sum([abs(x) for x in errors])
        cross_errors_square_norm = np.linalg.norm(np.array(errors))
        print(
            """[CROSS-VALIDATION] Loss on validation set 
                cross_errors_sum              {}
                cross_errors_square_norm      {}""".format(
                cross_errors_sum,
                cross_errors_square_norm))

        return cross_errors_square_norm, cross_errors_sum