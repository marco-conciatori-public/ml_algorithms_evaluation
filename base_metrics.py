from torch import Tensor


# ADD ARBITRARY METRICS HERE
# they must take four arguments:
#     tp: torch.Tensor,
#     fp: torch.Tensor,
#     tn: torch.Tensor,
#     fn: torch.Tensor
# each tensor is of shape (num_classes)
# and return a torch.Tensor of shape (num_classes)
# which represents the metric calculated separately for each class

# the abbreviations used in the function names are:
# tp = true positives
# fp = false positives
# tn = true negatives
# fn = false negatives

# finally, add the metric to the list of metrics in test_metric.py


# Recall: true positives / all true cases
# in multi-class: same operation but for each class separately and then averages the results
def recall(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> Tensor:
    return tp / (tp + fn)


# Precision: true positives / positives
# in multi-class: same operation but for each class separately and then averages the results
def precision(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> Tensor:
    return tp / (tp + fp)


# Accuracy: all positives / all cases
# in multi-class: same operation but for each class separately and then averages the results
def accuracy(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> Tensor:
    return (tp + tn) / \
                       (tp + fp + fn + tn)


# F1 score: harmonic mean of precision and recall
#   the operation below is the equivalent of 2 * precision * recall / (precision + recall)
#   but is better because it avoids errors in certain cases
def f1_score(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> Tensor:
    return 2 * tp / (2 * tp + fp + fn)
