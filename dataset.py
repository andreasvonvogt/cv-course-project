import os
import math
import cv2
import numpy as np
from sklearn.utils import shuffle
from dataset_location import DatasetLocation

class Dataset:
    VALIDATION_SPLIT = 0.1
    SAMPLE_WIDTH = 160
    SAMPLE_HEIGHT = 67

    def __init__(self):
        path = os.path.join(DatasetLocation.ROOT, '1')
        filenames = shuffle(os.listdir(path))
        val_idx = math.floor(len(filenames) * Dataset.VALIDATION_SPLIT)
        self.filenames_test = filenames[val_idx:]
        self.filenames_val = filenames[:val_idx]

    def get_samples(name, size, width, height):
        img = cv2.imread(os.path.join(DatasetLocation.ROOT, size, name)).astype(np.float32) / 255
        return np.expand_dims(img[:height, :width], axis=0) # TODO get multiple samples if img is larger

    def test(self, batch_size):
        while True:
            filenames = shuffle(self.filenames_test)
            num_samples = len(filenames)
            for start in range(0, num_samples - batch_size + 1, batch_size):
                x_batch = np.empty((0, Dataset.SAMPLE_HEIGHT, Dataset.SAMPLE_WIDTH, 3), np.uint8)
                y_batch = np.empty((0, 2 * Dataset.SAMPLE_HEIGHT, 2 * Dataset.SAMPLE_WIDTH, 3), np.uint8)
                for i in range(0, batch_size):
                    x_batch = np.concatenate((x_batch, Dataset.get_samples(filenames[start+i], '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT)), axis=0)
                    y_batch = np.concatenate((y_batch, Dataset.get_samples(filenames[start+i], '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT)), axis=0)

                yield x_batch, y_batch

    def test_size(self):
        return len(self.filenames_test)

    def validation(self):
        # TODO switch to a sequence/dataset class
        while True:
            for name in self.filenames_val:
                yield (Dataset.get_samples(name, '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT),
                       Dataset.get_samples(name, '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT))

    def validation_size(self):
        return len(self.filenames_val)
