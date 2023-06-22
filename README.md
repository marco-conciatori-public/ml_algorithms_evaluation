# ML algorithms evaluation

Algorithm based on the confusion matrix for the evaluation of machine learning
algorithms.

The test_metrics.py file can be run and modified to test the algorithm,
along with values in the config.py file.

It requires:
- python
- pytorch
- torchmetrics
- numpy

To instantiate the algorithm, the following parameters are required:
- a list of metrics to be evaluated from the confusion matrix
- a boolean to distinguish between continuous and discrete values
(respectively from regression and classification models)
- for classification models: the number of classes
- for regression models: a list of thresholds (the threshold will
be used to convert the continuous values to class indices)

Then the update method can be called any number of times on the instance
with a tensor of predictions (normally the output of the ML algorithm to
be evaluated) and a tensor of targets (the ground truth).

Finally with the compute method, the chosen metrics are evaluated and returned,
along with the raw confusion matrix.
The metrics are computed separately for each class and then averaged.

Code available under the GNU General Public License v3.0