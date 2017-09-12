Usage
=====

`train` and `predict` functions are provided as specified, taking a
filename string as the only argument. For `predict` to use a previously
trained model without additional arguments this requires the data reader
and model classes to be defined globally. Example usage is shown in the
`main()` function.

The `predict` function provided maps the predictions of the model, which
are labelled with integers, back to capital letters. The `NaiveBayes`
class method `accuracy(feats, labels)` can be used once trained (using
the `fit(feat, labels)` method), to compute the accuracy without
carrying out the conversion to letters. Example usage is provided in
`NaiveBayes` docstring.

Assumptions
-----------

-   Filenames used will end with either ’.csv’ or ’.json’ (not
    case–sensitive)

-   Data in CSV formats are provided with a header line and first column
    is the label

-   Data in JSON formats have three key levels: line number, label,
    features

-   No classes will appear in the labels of the test set that were not
    observed in the training set (otherwise an assertion error will be
    raised)

Solution
========

Two main problems were identified: firstly, the task of reading data
interchangeably from `csv` or `json` formats; secondly, the
classification task using Naive Bayes. Subsequently the system designed
consists of separate data reader class and Naive Bayes model class.

Data Reader
-----------

The `DataReader` class is initialised with a filename pointing to a
training dataset. The class learns the necessary transformations from
either `csv` or `json` file formats to a filetype–agnostic numeric
representation, a tuple of feature matrix and label vector both coded as
integers. The mapping of feature orders, feature levels, and labels to
integers are stored in the `DataReader` object, which can then be used
to read in test data with the same transformations applied. The storage
of these transformations allow e.g. the test dataset to be read from a
different file format than the training dataset.

For both `csv` and `json` files, at training time an arbitrary ordering
of features is read from the header line of `csv` files or the
lowest–level keys of `json` files is stored. The DataReader produces
based on this ordering a feature matrix with observations as rows and
features as columns, the columns ordered as specified by the stored
`feat_order` attribute. At testing time, any data read by the
`DataReader` method is permuted column–wise to match the ordering of
features in the feature matrix of the training set. This allows
interchangeabiliy of `csv`, which is read using the standard library
with each row as a list (an ordered data structure), and `json`, which
is read using standard library as a dictionary (an unordered data
structure). This was tested using the function `test_format_swaps()`.

Naive Bayes Model
-----------------

The model implemented is a Multinomial Naive Bayes. This was chosen as
the features are all measured discretely over a fixed scale (integers
from 0 to 15 inclusive). Preliminary data analysis, e.g. histograms of
the distributions of each feature, suggested that prior to
discretisation data was most likely continuous and normally distributed,
as would be expected from the feature descriptions available on the data
set website[^1]. Consequently a Gaussian Naive Bayes model was briefly
tested but found to perform worse than the multinomial case.

In the multinomial model, occurrences in the training data of each
combination of class, feature, and feature level are counted, then
normalised to give empirical frequencies. These are collected in a numpy
array private attribute of the `NaiveBayes` class, `_log_freq`. To allow
inference using combinations not seen in the training data,
Laplace/Lidstone smoothing is applied, adding pseudo–counts to each
entry in the table. Class priors (empirical frequency) were used and
calculated, though this was not particularly necessary for this dataset
given that it is very balanced.

The elements of the empirical frequency array can be expressed as
follows: $$\Pr(x_k = j | y=i) =
\theta_{ijk} = \frac{\#_{i}({x_k=j)} + \alpha}{\sum_{j}(\#_{i}(x_k=j) + \alpha)}$$
where $k$ indexes the feature, i.e. column of the feature matrix
consisting of rows $x$, $j$ the level of that feature, an integer from 0
to 15 inclusive, $i$ is the class (from label $y$), $\alpha$ is the
smoothing parameter, and $\#_i(x_k=j)$ denotes the count in class $i$ of
examples where feature $k$ has level $j$.

The total log-likelihood of the feature vector given a class is computed
as the sum of logs of these individual feature probabilities (using the
Naive Bayes assumption) and the class prior. Sum of logs is used rather
than straightforward multiplication to avoid numerical underflow.

Some generalisable elements are included in the Naive Bayes model that
are redundant for the particular datasets provided. In case the training
dataset does not contain all possible feature levels up to the maximum,
the class has a dictionary attribute to map the levels that do exist to
a contiguous range of integers that can be used to index a compact
frequency table. This is also implemented for the labels. This avoids
assuming that all labels or feature levels can be enumerated by e.g.
`range(n_labels)` or `range(n_levels)`.

### Hyperparameter optimisation

A brief hyperparameter optimisation was carried out for the smoothing
parameter $\alpha$ (see function `tune_alpha()`). The dataset of 16989
examples was split into a validation set of 2000 examples and a training
set of 14989 examples (using `tail`, `head` and `cat`). The training and
validation set accuracy was measured for a range of $\alpha$ as shown in
Table \[tab:hyperopt\]. On the basis of highest validation accuracy
$\alpha=0.005$ was chosen.

#### Cross-validation for $\alpha$
  -------- -------- --------
    0.0001   0.7700   0.7355
    0.0010   0.7699   0.7355
    0.0050   0.7694   0.7360
    0.0100   0.7691   0.7345
    0.0500   0.7673   0.7330
    0.1000   0.7659   0.7325
    0.1500   0.7641   0.7330
    0.2000   0.7632   0.7310
    0.2500   0.7619   0.7295
    0.3000   0.7609   0.7280
    0.4000   0.7595   0.7260
    0.6000   0.7567   0.7255
    0.8000   0.7547   0.7225
    1.0000   0.7527   0.7215
  -------- -------- --------



[^1]: <https://archive.ics.uci.edu/ml/datasets/Letter+Recognition>
