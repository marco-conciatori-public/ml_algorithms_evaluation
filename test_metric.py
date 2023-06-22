import torch

import utils
import config
import base_metrics
import multi_class_confusion_matrix


torch.manual_seed(0)  # comment this line to get different results

# choose metrics functions to compute
# they can be any functions implemented in base_metrics.py or imported (see base_metrics.py for the specifics)
metrics = [base_metrics.accuracy, base_metrics.precision, base_metrics.recall, base_metrics.f1_score]

# REGRESSION TEST
print('REGRESSION TEST')
# generate random input data
regression_predictions = torch.rand(size=(config.BATCH_SIZE, config.NUM_OBSERVATIONS))
regression_targets = torch.rand(size=(config.BATCH_SIZE, config.NUM_OBSERVATIONS))

# instantiate confusion matrix
regression_cm = multi_class_confusion_matrix.Confusion_matrix_flexible(
    metrics=metrics,
    continuous_values=True,
    thresholds=config.THRESHOLDS,
    name='test_regression_cm',
    verbose=config.VERBOSE,
)

# update confusion matrix
# repeat for each input or batch of inputs
regression_cm.update(predicted_values=regression_predictions, true_values=regression_targets)

# extract results up to this point
# it can be done at any stage during training
regression_results = regression_cm.compute()
utils.print_formatted_results(results=regression_results, display_as_percent=True, title='REGRESSION RESULTS')


# CLASSIFICATION TEST
print('\nCLASSIFICATION TEST')
# to test that the confusion matrix works with discrete values, we will use the same data as before
# converting them manually to classes
classification_predictions = torch.zeros_like(regression_predictions, dtype=torch.int32)
for batch_index in range(len(classification_predictions)):
    for prediction_index in range(len(classification_predictions[batch_index])):
        for threshold_index in range(len(config.THRESHOLDS)):
            if regression_predictions[batch_index][prediction_index] < config.THRESHOLDS[threshold_index]:
                classification_predictions[batch_index][prediction_index] = threshold_index
                break
        else:
            classification_predictions[batch_index][prediction_index] = len(config.THRESHOLDS)
if config.VERBOSE:
    print(f'classification_predictions.shape:\n{classification_predictions.shape}')
    print(f'classification_predictions:\n{classification_predictions}')

classification_targets = torch.zeros_like(regression_targets, dtype=torch.int32)
for batch_index in range(len(classification_targets)):
    for target_index in range(len(classification_targets[batch_index])):
        for threshold_index in range(len(config.THRESHOLDS)):
            if regression_targets[batch_index][target_index] < config.THRESHOLDS[threshold_index]:
                classification_targets[batch_index][target_index] = threshold_index
                break
        else:
            classification_targets[batch_index][target_index] = len(config.THRESHOLDS)
if config.VERBOSE:
    print(f'classification_targets.shape:\n{classification_targets.shape}')
    print(f'classification_targets:\n{classification_targets}')

# instantiate confusion matrix
classification_cm = multi_class_confusion_matrix.Confusion_matrix_flexible(
    metrics=metrics,
    continuous_values=False,
    num_classes=len(config.THRESHOLDS) + 1,
    name='test_classification_cm',
    verbose=config.VERBOSE,
)

# update confusion matrix
# repeat for each input or batch of inputs
classification_cm.update(predicted_values=classification_predictions, true_values=classification_targets)

# extract results up to this point
# it can be done at any stage during training
classification_results = classification_cm.compute()
utils.print_formatted_results(
    results=classification_results,
    display_as_percent=True,
    title='CLASSIFICATION RESULTS',
)
