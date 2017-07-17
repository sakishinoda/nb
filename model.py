import csv
import json
import numpy as np
import IPython

class NaiveBayes(object):
    """
    Base class for a Naive Bayes classifier.
    Fitting is significantly simplified for the given dataset because it has discrete data in 15 buckets.
    All features are distributed in the same way, so we can use array operations on a count-per-bin per feature per class matrix.
    Takes features in integer levels from 0 to a maximum number
    Takes labels in integers from 0 to a maximum
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def log_likelihood(self, feat_vals, y):
        p = self.priors[y]
        for idx, val in enumerate(feat_vals):
            p += self.log_prob(y, idx, val)
        return p


    def fit(self, feats, labels):
        # From labels extract classes vector
        self.n_classes = self.count_classes(labels)

        # Extract number of features and number of discrete feature levels
        self.n_feats, self.n_levels = self.count_feats_and_levels(feats)

        # Define log_probability function based on data
        self._log_prob, self.priors = self.get_freq(feats, labels)

        return self

    def log_prob(self, label, feat_idx, feat_val):
        return self._log_prob[label, feat_val, feat_idx]

    def get_freq(self, feats, labels):
        n_examples = feats.shape[0]
        freq = np.ndarray((self.n_classes, self.n_levels, self.n_feats))
        prior = np.ndarray((self.n_classes,))
        z = np.ndarray((self.n_classes, 1, 1))

        # assume classes are indexed by integers 0 to n_classes
        for c in range(self.n_classes):
            c_feats = feats[np.where(labels == c)]
            num_in_class = c_feats.shape[0]
            prior[c] = num_in_class
            z[c, 0, 0] = num_in_class + self.n_levels * self.alpha

            # assuming that the levels are indexed in integers 0 to n_levels
            for f in range(self.n_levels):
                p = (np.count_nonzero(c_feats == f, axis=0) + self.alpha)

                freq[c, f, :] = p

        # broadcasted normalisation
        return np.log(freq) - np.log(z), np.log(prior/n_examples)
        # return np.log(freq)

    @staticmethod
    def count_classes(labels):
        return len(sorted(set(labels)))

    @staticmethod
    def count_feats_and_levels(feats):
        # Take one variable (a column)
        n_levels = len(set(feats[:,0]))
        n_feats = feats.shape[1]
        return n_feats, n_levels

    def predict(self, feats):
        p = np.zeros((feats.shape[0], self.n_classes))
        for r in range(p.shape[0]):
            for c in range(self.n_classes):
                p[r, c] = self.log_likelihood(feats[r], c)

        return np.argmax(p, axis=1)

    def accuracy(self, feats, labels):
        predict = self.predict(feats)
        acc = (predict == labels).sum() / predict.shape[0]
        return acc




class DataReader(object):
    def __init__(self, fname):
        """

        :param fname:
        :param csv:
        """
        self._label_to_int = None
        self.feats, self.labels = self.load_data(fname)

    def load_data(self, fname):
        csv = True if fname.lower().split('.')[-1] == 'csv' else False

        if csv:
            return self.load_csv(fname)
        else:
            return self.load_json(fname)



    def load_csv(self, fname):
        with open(fname, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        feats = np.array([row[1:] for row in data[1:]], dtype=int)
        raw_labels = [row[0] for row in data[1:]]

        # If initialising the data reader object (with the training set) create the conversion mappings
        if self._label_to_int is None:
            label_to_int = dict(zip(set(raw_labels), range(len(set(raw_labels)))))
            self._label_to_int = label_to_int
            self._int_to_label = {v: k for k, v in label_to_int.items()}  # creates the inverse mapping

        labels = self.label_to_int(raw_labels)

        return feats, labels

    def load_json(self, fname):

        with open(fname, 'r') as f:
            data = json.load(f)

        n_examples = len(data.keys())
        labels = np.ndarray((n_examples,))

        label_to_int = dict()

        label_counter = 0

        for i, val in enumerate(data.values()):
            this_label = [x for x in val.keys()][0]

            # If the label hasn't been found yet, add to dictionary
            if this_label not in label_to_int.keys():
                label_to_int[this_label] = label_counter
                label_counter += 1

            labels[i] = label_to_int[this_label]

            # Extract headers
            if i == 0:
                feat_order = list(val[this_label].keys())
                feats = np.ndarray((n_examples, len(feat_order)), dtype=int)

            for j, o in enumerate(feat_order):
                feats[i, j] = val[this_label][o]

        # If initialising the data reader object (with the training set) create the conversion mappings
        if self._label_to_int is None:
            self._label_to_int = label_to_int
            self._int_to_label = {v: k for k, v in label_to_int.items()}  # creates the inverse mapping

        return feats, labels

    def label_to_int(self, labels):
        return np.array([self._label_to_int[c] for c in labels])

    def int_to_label(self, ints):
        return np.array([self._int_to_label[c] for c in ints])

train_json = './data/train.json'
val_json = './data/validation.json'
json_dr = DataReader(train_json)

train_csv = './data/train.csv'
val_csv = './data/validation.csv'
csv_dr = DataReader(train_csv)

# IPython.embed()

val_feats, val_labels = json_dr.load_data(val_json)


alpha = 1.0
nb = NaiveBayes(alpha=alpha)
nb.fit(json_dr.feats, json_dr.labels)

print(alpha, nb.accuracy(json_dr.feats, json_dr.labels))
print(nb.accuracy(val_feats, val_labels))

# Laplace smoothing
# Training error: 75.262
# Validation error 72.150

# Lidstone smoothing
# 0.001 0.769831209554 0.7355
# 0.005 0.769364200414 0.736
# 0.01 0.769097338048 0.7345
# 0.05 0.767229301488 0.733
# 0.1 0.765828274068 0.733
# 0.15 0.76409366869 0.733
# 0.2 0.763293081593 0.731
# 0.25 0.76222563213 0.73
# 0.3 0.760958035893 0.728
# 0.35 0.760157448796 0.727
# 0.4 0.759490292881 0.726
# 0.6 0.756688238041 0.7255
# 0.8 0.754686770298 0.7225
# 1.0 0.752618586964 0.7215

# IPython.embed()
