# coding: utf-8
import os
import random
import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger("train_log")


class Dataset(object):
    """Dataset class"""

    def __init__(self, name, data_home="./data", **kwargs):
        self.name = name
        self.seed = kwargs.get("seed")
        self.paths = kwargs.get("paths")
        self.filepath = f"{data_home}/{self.name}"
        self.n_samples = kwargs.get("n_samples")
        self.valid_pct = kwargs.get("valid_pct")
        self.batch_size = kwargs.get("batch_size")
        self.randomize_minibatch = kwargs.get("randomize_minibatch")

    @property
    def names(self):
        return self._names

    @property
    def classes(self):
        return self._classes

    @property
    def counts(self):
        return self._counts

    def print_summary(self):
        pass

    @property
    def n_classes(self):
        return len(self._classes)

    def save_data(self, X, y):
        plt.savefig(self.filepath + ".pdf")
        np.savez(self.filepath + ".npz", X=X, y=y)

    def encode_target(self, target):
        target = np.squeeze(target)
        names, counts = np.unique(target, return_counts=True)
        new_target = np.empty_like(target, dtype=int)
        for i, name in enumerate(names):
            new_target[target == name] = i

        classes = range(len(names))
        return new_target, classes, names, counts

    def read_data(self):
        try:
            X, y = self.load_dataset(self.filepath + ".npz")
            # convert multiclass -> binary labels
            y, self._classes, self._names, self._counts = self.encode_target(y)
            y = self.binarize_target(target=y)
            return X, y

        except:
            print("exception while reading data!")

    def load_dataset(self, filepath):
        if os.path.exists(filepath):
            logger.info("Dataset found ..")
            data = np.load(filepath, allow_pickle=True)
            return data["X"], data["y"]
        else:
            raise ValueError("Please provide valid paths for the datasets.")

    def binarize_target(self, target):
        """Converts a multi-class dataset to binary dataset by assigning majority
        class as "1" and rest of the classes together as "0"
        """
        class_counts = np.bincount(target)
        if np.alen(class_counts) > 2:
            majority = np.argmax(class_counts)
            t = np.zeros_like(target)
            t[target == majority] = 1

        else:
            t = target

        return t

    def prepare_data(self, x, y, training=True):
        """Creates TF Datasets"""
        # no. of samples and features
        n_samples, _ = x.shape
        data = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            data = data.shuffle(
                buffer_size=n_samples, reshuffle_each_iteration=self.randomize_minibatch
            ).batch(
                self.batch_size
            )  # , drop_remainder=True)
        else:
            data = data.batch(self.batch_size)  # no shuffling

        data = data.cache()
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data
