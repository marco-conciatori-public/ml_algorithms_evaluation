import torch
import torchmetrics
import numpy as np
import warnings

import utils
import global_constants


class Confusion_matrix_flexible(torchmetrics.Metric):
    def __init__(self,
                 metrics: list,
                 continuous_values: bool,
                 num_classes: int = None,
                 thresholds: list = None,
                 name: str = 'confusion_matrix',
                 verbose: bool = False,
                 ):
        super().__init__()
        if continuous_values:
            assert thresholds is not None, 'when using continuous values, thresholds must be specified.'
            assert utils.check_strictly_increasing(thresholds), 'thresholds must be ordered from lowest to highest.'
            if num_classes is not None:
                warnings.warn('when using continuous values, num_classes is ignored.')
            self.thresholds = torch.tensor(thresholds)
            self.num_classes = len(self.thresholds) + 1
        else:
            assert num_classes is not None, 'when using discrete values, num_classes must be specified.'
            self.num_classes = num_classes
        self.metrics = metrics
        self.confusion_matrix_name = name
        self.continuous_values = continuous_values
        self.add_state(
            name=name,
            default=torch.zeros(
                size=(self.num_classes, self.num_classes),
                dtype=torch.int32,
            ),
        )
        self.verbose = verbose

    def update(self, predicted_values: torch.Tensor, true_values: torch.Tensor):
        if self.verbose:
            print(f'predicted_values.shape:\n{predicted_values.shape}')
            print(f'predicted_values:\n{predicted_values}')
            print(f'true_values.shape:\n{true_values.shape}')
            print(f'true_values:\n{true_values}')

        if self.continuous_values:
            # each element in these vectors indicate the bucket assigned to the corresponding h_shift
            class_true = torch.bucketize(true_values, boundaries=self.thresholds)
            class_pred = torch.bucketize(predicted_values, boundaries=self.thresholds)
        else:
            class_true = true_values
            class_pred = predicted_values
        if self.verbose:
            print(f'class_pred:\n{class_pred}')
            print(f'class_true:\n{class_true}')

        class_true = class_true.flatten()
        class_pred = class_pred.flatten()
        if self.verbose:
            print(f'class_pred:\n{class_pred}')
            print(f'class_true:\n{class_true}')

        # get the confusion matrix from the string of the name of the class attribute
        confusion_matrix = getattr(self, self.confusion_matrix_name)

        # update the confusion matrix
        for batch_index in range(len(class_true)):
            confusion_matrix[class_true[batch_index]][class_pred[batch_index]] += 1

    def compute(self) -> dict:
        # with these two lines we suppress runtime warnings from the 'compute' method. This is necessary because
        #   when the network learns poorly, the confusion matrix can have all the values in one column and
        #   zeros in all the other cells. In this case, some of the following operations rise 'divide by zero' warnings,
        #   but the program works correctly and assigns 'NaN' to the corresponding metric.
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            confusion_matrix = getattr(self, self.confusion_matrix_name).numpy()
            elements_on_diagonal = np.diag(confusion_matrix)
            false_positives = confusion_matrix.sum(axis=0) - elements_on_diagonal
            false_negatives = confusion_matrix.sum(axis=1) - elements_on_diagonal
            true_positives = elements_on_diagonal
            true_negatives = confusion_matrix.sum() - (false_positives + false_negatives + true_positives)

        result = {
            global_constants.CONFUSION_MATRIX: confusion_matrix,
        }
        for metric in self.metrics:
            result_by_class = metric(true_positives, false_positives, true_negatives, false_negatives)
            averaged_result = result_by_class.mean()
            if np.isnan(averaged_result):
                averaged_result = 0
            result[metric.__name__] = averaged_result

        return result
