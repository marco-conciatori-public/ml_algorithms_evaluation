import torch
import torchmetrics
import numpy as np
import warnings

import utils
import global_constants


class Confusion_matrix_from_continuous_values(torchmetrics.Metric):
    def __init__(self, name: str, meta_data: dict):
        super().__init__()
        self.min_val = meta_data[global_constants.GLOBAL_MIN_VALUE]
        self.max_val = meta_data[global_constants.GLOBAL_MAX_VALUE]
        self.thresholds = torch.tensor(meta_data[global_constants.THRESHOLDS_NORMALIZED])
        self.num_classes = len(self.thresholds) + 1
        self.confusion_matrix_name = name

        self.name = 'confusion_matrix_' + name
        assert utils.check_strictly_increasing(meta_data[global_constants.THRESHOLDS_NORMALIZED]), \
            'ERROR: thresholds should be ordered from lowest to highest'
        self.add_state(
            name=name,
            default=torch.zeros(
                size=(self.num_classes, self.num_classes),
                dtype=torch.int32,
            ),
        )

    def update(self, predicted_values, true_values):
        # send to cpu
        if hasattr(predicted_values, 'cpu'):
            predicted_values = predicted_values.cpu()
            true_values = true_values.cpu()
        # remove unused extra dimension
        if len(predicted_values.shape) == 3:
            true_values = true_values.squeeze(-2)
            predicted_values = predicted_values.squeeze(-2)

        # print(f'predicted_values.shape: {predicted_values.shape}')
        # print(f'predicted_values.shape: {len(predicted_values.shape)}')
        # print(f'true_values.shape: {true_values.shape}')

        if len(predicted_values.shape) == 2:
            if true_values.shape[-1] == 2:
                print('2D case')
                # evaluate h_shift from x_shift and y_shift for both predictions and targets
                norm_h_true = torch.hypot(true_values[:, 0::2], true_values[:, 1::2])
                norm_h_predicted = torch.hypot(predicted_values[:, 0::2], predicted_values[:, 1::2])
            else:
                print('1D case')
                norm_h_true = true_values
                norm_h_predicted = predicted_values
        elif len(predicted_values.shape) == 1:
            print('batched 0D case')
            norm_h_true = true_values
            norm_h_predicted = predicted_values
        elif len(predicted_values.shape) == 0:
            print('single value 0D case')
            norm_h_true = true_values
            norm_h_predicted = predicted_values
        else:
            raise ValueError('predicted_values has an unexpected shape.')

        print(f'norm_h_true.shape: {norm_h_true.shape}')
        print(f'norm_h_predicted.shape: {norm_h_predicted.shape}')

        # each element in these vectors indicate the bucket assigned to the corresponding h_shift
        class_true = torch.bucketize(norm_h_true, boundaries=self.thresholds)
        class_pred = torch.bucketize(norm_h_predicted, boundaries=self.thresholds)
        class_true = class_true.flatten()
        class_pred = class_pred.flatten()

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

            # Recall: true_positives / all true cases
            # in multi-class: same operation but for each class separately and then averages the results
            recall = true_positives / (true_positives + false_negatives)

            # Precision: true positives / positives
            # in multi-class: same operation but for each class separately and then averages the results
            precision = true_positives / (true_positives + false_positives)

            # Accuracy: all positives / all cases
            # in multi-class: same operation but for each class separately and then averages the results
            accuracy = (true_positives + true_negatives) / \
                       (true_positives + false_positives + false_negatives + true_negatives)

            # F1 score: harmonic mean of precision and recall
            #   the operation below is the equivalent of 2 * precision * recall / (precision + recall)
            #   but is better because it avoids division by zero in certain cases
            f1_score = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

            # change from a value for each class to the average of all classes
            accuracy = accuracy.mean()
            precision = precision.mean()
            recall = recall.mean()
            f1_score = f1_score.mean()

            # substitute NaN with 0
            if np.isnan(accuracy):
                accuracy = 0
            if np.isnan(precision):
                precision = 0
            if np.isnan(recall):
                recall = 0
            if np.isnan(f1_score):
                f1_score = 0

        result = {
            global_constants.CONFUSION_MATRIX: confusion_matrix,
            global_constants.ACCURACY: accuracy,
            global_constants.PRECISION: precision,
            global_constants.RECALL: recall,
            global_constants.F1_SCORE: f1_score,
        }
        return result
