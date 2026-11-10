import os
import math
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from dataset_location import DatasetLocation

def centered_sample(img, width, height):
    h, w, _ = img.shape
    top = (h - height) // 2
    left = (w - width) // 2
    return img[top:top+height, left:left+width]

class Dataset:
    VALIDATION_SPLIT = 0.1
    SAMPLE_WIDTH = 64
    SAMPLE_HEIGHT = 64

    def __init__(self):
        data = pd.read_csv(os.path.join(DatasetLocation.ROOT, 'images.csv')).sample(frac=1)
        val_idx = math.floor(len(data) * Dataset.VALIDATION_SPLIT)
        self.data_test = data[val_idx:]
        self.data_val = data[:val_idx]

    def get_samples(image, size, width, height):
        img = cv2.imread(os.path.join(DatasetLocation.ROOT, size, image['Name'])).astype(np.float32) / 255
        return np.expand_dims(centered_sample(img, width, height), axis=0) # TODO get multiple samples if img is larger

    def test(self, batch_size):
        num_samples = self.test_size()
        while True:
            data = shuffle(self.data_test).sample(frac=1)
            for start in range(0, num_samples - batch_size + 1, batch_size):
                x_batch = np.empty((0, Dataset.SAMPLE_HEIGHT, Dataset.SAMPLE_WIDTH, 3), np.float32)
                y_batch = np.empty((0, 2 * Dataset.SAMPLE_HEIGHT, 2 * Dataset.SAMPLE_WIDTH, 3), np.float32)
                for i in range(0, batch_size):
                    x_batch = np.concatenate((x_batch, Dataset.get_samples(data.iloc[start+i], '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT)), axis=0)
                    y_batch = np.concatenate((y_batch, Dataset.get_samples(data.iloc[start+i], '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT)), axis=0)

                yield x_batch, y_batch

    def test_size(self):
        return len(self.data_test)

    def validation(self):
        # TODO switch to a sequence/dataset class
        while True:
            for name in self.data_val.iloc:
                yield (Dataset.get_samples(name, '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT),
                       Dataset.get_samples(name, '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT))

    def validation_size(self):
        return len(self.data_val)
