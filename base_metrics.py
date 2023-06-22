from torch import Tensor


# ADD ARBITRARY METRICS HERE
# they must take four arguments:
#     true_positives: torch.Tensor,
#     false_positives: torch.Tensor,
#     true_negatives: torch.Tensor,
#     false_negatives: torch.Tensor
# each tensor is of shape (num_classes)
# and return a torch.Tensor of shape (num_classes)
# which represents the metric calculated separately for each class

# finally, add the metric to the list of metrics in test_metric.py


# Recall: true_positives / all true cases
# in multi-class: same operation but for each class separately and then averages the results
def recall(true_positives: Tensor, false_positives: Tensor, true_negatives: Tensor, false_negatives: Tensor) -> Tensor:
    return true_positives / (true_positives + false_negatives)


# Precision: true positives / positives
# in multi-class: same operation but for each class separately and then averages the results
def precision(true_positives: Tensor, false_positives: Tensor, true_negatives: Tensor, false_negatives: Tensor) -> Tensor:
    return true_positives / (true_positives + false_positives)


# Accuracy: all positives / all cases
# in multi-class: same operation but for each class separately and then averages the results
def accuracy(true_positives: Tensor, false_positives: Tensor, true_negatives: Tensor, false_negatives: Tensor) -> Tensor:
    return (true_positives + true_negatives) / \
                       (true_positives + false_positives + false_negatives + true_negatives)


# F1 score: harmonic mean of precision and recall
#   the operation below is the equivalent of 2 * precision * recall / (precision + recall)
#   but is better because it avoids errors in certain cases
def f1_score(true_positives: Tensor, false_positives: Tensor, true_negatives: Tensor, false_negatives: Tensor) -> Tensor:
    return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
