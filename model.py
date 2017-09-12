import csv
import json
import numpy as np
import IPython

class NaiveBayes(object):
    """
    Multinomial Naive Bayes classifier.

    Suitable for data provided in separate (n_examples x n_feats) features
    array and (n_examples, ) labels array, with both coded in discrete
    integer levels. Requires the range of levels of each feature to be the
    same.


    Parameters
    ----------
        alpha : float, optional (default=1.0)
            Smoothing prior.


    Attributes
    ----------

        n_classes : int
            Number of classes found in training data during call to fit (number
            of unique labels found in labels array). Allows classes to be
            identified by an integer in range(n_classes) using classes dict

        classes : dict, length=n_classes
            Dictionary mapping from range(n_classes) to the labels found in
            label vector at training time

        n_levels : int
            Number of integer levels of each feature (0-15 in data provided)

        levels : dict, length=n_levels
            Maps each integer level in original features to an integer in
            range(n_levels). Required in case not all integers in
            range(max(feature_column)) are found in data

        n_feats : int
            Number of features found during call to fit (number of columns in
            features array)

        priors : array, shape=[n_classes, ]
            Log prior probabilities of each class, calculated during call to
            fit as empirical frequency of class.


    Examples
    --------

        >>> import numpy as np
        >>> X = np.random.randint(2, size=(5, 16))
        >>> Y = np.array([0,2,4,4,4])
        >>> from model import NaiveBayes
        >>> NB = NaiveBayes(1.0)
        >>> NB.fit(X, Y)
        >>> NB.predict(X)
        array([0, 2, 4, 4, 4])

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, feats, labels):
        """Count number of class/feature/level occurrences to fit Naive Bayes

        Args:
            feats : numpy array, shape=(n_examples, n_feats)
                Matrix of feature vectors as rows. n_examples and n_feats
                are extracted from this matrix at fitting time.

            labels : numpy array, shape=(n_examples, )
                Vector of class labels
        """

        # From labels extract classes vector
        self.classes, self.n_classes = self._extract_classes(labels)

        # Extract number of features and number of discrete feature levels
        self.n_feats, self.levels, self.n_levels = self._extract_feats_and_levels(feats)

        # Define log_probability function based on data
        self._log_prob, self.priors = self._count(feats, labels)

        return self


    def log_likelihood(self, feat_vals, y_idx):
        """Compute the log likelihood of a feature vector for given class"""
        p = self.priors[y_idx]
        for idx, val in enumerate(feat_vals):
            p += self.log_prob(y_idx, idx, val)
        return p

    def log_prob(self, label_idx, feat_idx, feat_val):
        """Extract entry in log frequency table for given class/feature/level"""
        return self._log_prob[label_idx, self.levels[feat_val], feat_idx]

    def _count(self, feats, labels):
        """Fill log frequency table for given class/feature/level

        Args:
            feats : numpy array, shape=(n_examples, n_feats)
            labels : numpy array, shape=(n_examples, )

        Returns:
            Tuple of:
                Normalised log frequency table, array
                Log priors based on empirical class frequencies

        """
        n_examples = feats.shape[0]
        freq = np.ndarray((self.n_classes, self.n_levels, self.n_feats))
        prior = np.ndarray((self.n_classes,))
        z = np.ndarray((self.n_classes, 1, 1))

        # contiguous integer index of class as i, class identity as c
        for i, c in self.classes.items():
            c_feats = feats[np.where(labels == c)]
            num_in_class = c_feats.shape[0]
            prior[i] = num_in_class
            z[i, 0, 0] = num_in_class + self.n_levels * self.alpha

            # contiguous integer index of class as j, level identity as f
            for f, j in self.levels.items():
                p = (np.count_nonzero(c_feats == f, axis=0) + self.alpha)
                freq[i, j, :] = p

        # broadcasted normalisation
        return np.log(freq) - np.log(z), np.log(prior/n_examples)


    def predict(self, feats):
        """
        Predict the labels for a given feature array

        First produces labels using internal integer representations and
        then converts to original integers

        Args:
            feats : numpy array

        Returns:
            A vector of labels
        """

        p = np.zeros((feats.shape[0], self.n_classes))
        for r in range(p.shape[0]):
            for i in self.classes.keys():
                p[r, i] = self.log_likelihood(feats[r], i)

        # Compute argmax and convert back to original label integers
        func = np.vectorize(lambda x: self.classes[x])
        return func(np.argmax(p, axis=1))

    def accuracy(self, feats, labels):
        """Compute the accuracy on the given features"""
        predict = self.predict(feats)
        acc = (predict == labels).sum() / predict.shape[0]
        return acc

    @staticmethod
    def _extract_classes(labels):
        classes = set(labels)
        n_classes = len(classes)
        return dict(zip(range(n_classes), classes)), n_classes

    @staticmethod
    def _extract_feats_and_levels(feats):
        # Take one variable (a column), map levels to contiguous integers
        levels = set(feats[:,0])
        n_levels = len(levels)
        n_feats = feats.shape[1]
        return n_feats, dict(zip(levels, range(n_levels))), n_levels





class DataReader(object):
    """
    Read JSON or CSV files and convert to filetype-agnostic arrays for NB

    Initialise the DataReader object with the filename of the training data,
    saving the order of features and mapping of integers/labels to allow
    consistent import of test/validation sets.

    Parameters
    ----------
        fname : string
            String of filename of training data relative to working
            directory, must be either .csv or .json

    Attributes
    ----------
        feats : numpy array, shape=(n_examples, n_feats)
            Matrix of feature vectors as rows. n_examples and n_feats
            are extracted from this matrix at fitting time.

        labels : numpy array, shape=(n_examples, )
            Vector of class labels

    """
    def __init__(self, fname):
        self._label_to_int = None
        self._feat_order = None
        self.feats, self.labels = self.load_data(fname)

    def load_data(self, fname):
        """Load json or csv from filename string"""

        filetype = fname.lower().split('.')[-1]
        if filetype == 'csv':
            return self.load_csv(fname)
        else:
            assert filetype == 'json'
            return self.load_json(fname)

    def load_csv(self, fname):
        """Convert csv to feature, label arrays

        Unlike json, csv files are read with one function with an if switch.
        A feature ordering is stored at training time based on the header file
        and at testing time the columns of the new feature matrix are permuted
        so that the headers of the new data match the ordering of the headers
        of the old data.

        Args:
            fname, string
                Filepath of csv which must have feature headers as first line

        """
        with open(fname, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        feats = np.array([row[1:] for row in data[1:]], dtype=int)
        raw_labels = [row[0] for row in data[1:]]

        # If initialising the data reader object, i.e. training:
        if self._label_to_int is None:
            #  get the header names
            self._feat_order = data[0][1:]
            #  create the conversion mappings
            label_to_int = dict(zip(set(raw_labels), range(len(set(raw_labels)))))
            self._label_to_int = label_to_int
            self._int_to_label = {v: k for k, v in label_to_int.items()}  # creates the inverse mapping
        else:  # if testing
            assert self._feat_order is not None
            correct_order = dict(zip(self._feat_order, range(len(self._feat_order))))
            current_order_labels = data[0][1:]
            current_order_ints = [correct_order[k] for k in current_order_labels]
            permute = np.argsort(current_order_ints)
            feats = feats[:, permute]

        labels = self.label_to_int(raw_labels)

        return feats, labels

    def load_json(self, fname):
        """Convert json to feature, label arrays"""
        with open(fname, 'r') as f:
            data = json.load(f)

        if self._label_to_int is None:
            return self.load_train_json(data)
        else:
            return self.load_test_json(data)

    def load_train_json(self, data):
        """Extract feature order, label and feature levels from training data"""
        n_examples = len(data.keys())
        labels = np.ndarray((n_examples,))
        label_to_int = dict()
        label_counter = 0

        for i, val in enumerate(data.values()):
            this_label = [x for x in val.keys()][0]

            # Extract headers
            if i == 0:
                self._feat_order = list(val[this_label].keys())
                feats = np.ndarray((n_examples, len(self._feat_order)), dtype=int)

            # If the label hasn't been found yet, add to dictionary
            if this_label not in label_to_int.keys():
                label_to_int[this_label] = label_counter
                label_counter += 1

            labels[i] = label_to_int[this_label]
            for j, o in enumerate(self._feat_order):
                feats[i, j] = val[this_label][o]

        # Create the conversion mappings
        self._label_to_int = label_to_int
        self._int_to_label = {v: k for k, v in label_to_int.items()}
        return feats, labels

    def load_test_json(self, data):
        """Apply feature order, label and feature levels to testing data"""
        n_examples = len(data.keys())
        labels = np.ndarray((n_examples,))
        feats = np.ndarray((n_examples, len(self._feat_order)), dtype=int)
        label_to_int = self._label_to_int

        for i, val in enumerate(data.values()):
            this_label = [x for x in val.keys()][0]
            assert this_label in label_to_int.keys()  # check label in classes from training set
            labels[i] = label_to_int[this_label]
            for j, feat_key in enumerate(self._feat_order):
                feats[i, j] = val[this_label][feat_key]

        return feats, labels

    def label_to_int(self, labels):
        return np.array([self._label_to_int[c] for c in labels])

    def int_to_label(self, ints):
        return np.array([self._int_to_label[c] for c in ints])


def test_format_swaps():
    """Test interchangeability of json and csv at train/test time"""
    train_fnames = ['./data/train.json', './data/train.csv']
    val_fnames = ['./data/validation.json', './data/validation.csv']
    alphas = [0.001, 0.005]

    for alpha in alphas:
        for train in train_fnames:
            print(train)
            reader = DataReader(train)
            nb = NaiveBayes(alpha=alpha)
            nb.fit(reader.feats, reader.labels)
            print('Training set accuracy {:4.2f}%'.format(100*nb.accuracy(reader.feats, reader.labels)))
            for val in val_fnames:
                print(val)
                val_feats, val_labels = reader.load_data(val)
                print('Validation set accuracy {:4.2f}%'.format(100 * nb.accuracy(val_feats, val_labels)))


def tune_alpha():
    """Cross-validate to find best alpha"""
    alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0]

    for alpha in alphas:
        reader = DataReader('./data/train.json')
        nb = NaiveBayes(alpha)
        nb.fit(reader.feats, reader.labels)
        val_feats, val_labels = reader.load_data('./data/validation.json')
        print('{:4.4f} & {:4.4f} & {:4.4f} \\\\'.format(
            alpha,
            nb.accuracy(reader.feats, reader.labels),
            nb.accuracy(val_feats, val_labels)
        ))

def train(input_file):
    """Load data from input_file and train Naive Bayes model"""
    reader = DataReader(input_file)
    nb = NaiveBayes(alpha=0.0001)
    nb.fit(reader.feats, reader.labels)
    print('Trained on {}.\nTraining set accuracy: {:4.4f}%'.format(
        input_file, 100*nb.accuracy(reader.feats, reader.labels)))
    return reader, nb


def predict(input_file):
    """Predict the labels for data loaded from input_file

    Requires training data to have been previously loaded by a DataReader
    object persisting as a global variable READER and a trained Naive Bayes
    model NB.
    Args:
        input_file, string
            Path to file containing data in csv or json format.

    Returns:
        A numpy array of strings (capital letters)
    """
    global READER, NB
    feats, labels = READER.load_data(input_file)
    output = READER.int_to_label(NB.predict(feats))
    return output


def main():
    """Train and calculate training accuracy by default on csv"""
    global READER, NB
    fname = './data/letter-recognition.csv'
    READER, NB = train(fname)
    output = predict(fname)
    print(output)

if __name__ == '__main__':
    main()
