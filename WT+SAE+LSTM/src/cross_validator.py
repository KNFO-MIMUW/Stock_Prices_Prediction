import numpy as np


class CrossValidator:
    def __init__(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        assert (train_ratio + valid_ratio + test_ratio == 1)

    def _get_training(self, dataset):
        new_lenght = int(len(dataset) * self.test_ratio)
        return dataset[:new_lenght]

    def _get_validation(self, dataset):
        ldata = len(dataset)
        test_end = int(ldata * self.test_ratio)
        validation_end = int(ldata * (self.test_ratio + self.valid_ratio))

        return dataset[test_end:validation_end]

    def run_validation(self, runner, dataset, epoch_train=100):
        trainning_dataset = self._get_training(dataset)
        runner.train(trainning_dataset, epoch_train)

        validation_dataset = self._get_validation(dataset)
        errors = []
        for x, t in validation_dataset:
            pred_t = runner.eval_single(x)
            t = t.squeeze()
            pred_t = pred_t.squeeze()
            errors.append(t[-1] - pred_t[-1])
        return np.linalg.norm(np.array(errors))
