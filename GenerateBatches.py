import sys
import numpy as np
import copy
import pandas as pd
from sklearn.model_selection import train_test_split


class GenerateBatches(object):

    def __init__(self, columns, data):
        lengths = [mat.shape[0] for mat in data]
        self.length = lengths[0]
        self.columns = columns
        self.data = data
        #[0 1 2 3 .... till last row]
        self.index = np.arange(self.length)
        self.random_integer = np.random.randint(10000)
        self.train_data_split = 0.95

    def train_test_split(self):
        train_keys, test_keys = train_test_split(self.index, train_size=self.train_data_split, random_state=self.random_integer)
        train_data = GenerateBatches(copy.copy(self.columns), [mat[train_keys] for mat in self.data])
        test_data = GenerateBatches(copy.copy(self.columns), [mat[test_keys] for mat in self.data])
        return train_data, test_data

    def batch_generator(self, batch_size, total_iterations=10000, allow_smaller_final_batch=False):
        current_iteration = 0
        while current_iteration < total_iterations:
        	np.random.shuffle(self.index)
            for i in range(0, self.length + 1, batch_size):
                batch_index = self.index[i: i + batch_size]
                if not allow_smaller_final_batch and len(batch_index) != batch_size:
                    break
                yield GenerateBatches(columns=copy.copy(self.columns), data=[mat[batch_index].copy() for mat in self.data])
            current_iteration += 1