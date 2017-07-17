import numpy as np
import math
import csv

class DataReader(object):
    """
    Read data from either csv or json file.
    Extracts:
     - an integer array of features
     - an integer vector of class labels
     - a dictionary mapping integer -> class label string
    """
    def __init__(self, fname):

        self.n_examples = None
        self.n_classes = None
        self.n_feats = None
        self.n_feat_levels = None

        self.feats, self.labels, self.classes, self.classes_dict = self.load_data(fname)

    def load_data(self, fname):
        filetype = fname.lower().split('.')[-1]
        if filetype == 'csv':
            feats, labels, classes, classes_dict = self._load_csv(fname)
        elif filetype == 'json':
            feats, labels = self._load_json(fname)


        num_classes = len(classes)
        num_feats = feats.shape[1]
        num_feat_levels = 16
        feat_range = range(num_feat_levels)



    def _load_csv(self, fname):
        with open(fname, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        self.n_examples = len(data)
        header = data[0]
        feats = np.array([row[1:] for row in data[1:]], dtype=int)
        label_chars = [row[0] for row in data[1:]]
        char2int = dict(zip(sorted(set(label_chars)), range(len(set(label_chars)))))
        labels = np.array([char2int[c] for c in label_chars])
        classes = sorted(char2int.values())
        return feats, labels

    def _load_json(self, fname):
        return None, None, None, None