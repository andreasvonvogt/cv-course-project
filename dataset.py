import os
import math
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from dataset_location import DatasetLocation

def centered_sample(img, width, height):
    """Extract a single sample from an image, centered on both dimensions."""
    h, w, _ = img.shape
    top = (h - height) // 2
    left = (w - width) // 2
    return img[top:top+height, left:left+width]

class Dataset:
    """Class encapsulating the training dataset."""
    VALIDATION_SPLIT = 0.1
    SAMPLE_WIDTH = 64
    SAMPLE_HEIGHT = 64

    def __init__(self):
        data = pd.read_csv(os.path.join(DatasetLocation.ROOT, 'images.csv')).sample(frac=1)
        data['Samples'] = (data['Width'] // Dataset.SAMPLE_WIDTH) * (data['Height'] // Dataset.SAMPLE_HEIGHT)
        val_idx = math.floor(len(data) * Dataset.VALIDATION_SPLIT)
        self.data_train = data[val_idx:]
        self.data_val = data[:val_idx]

    def get_samples(image, size, width, height):
        """
        Extract (probably multiple) samples from an image with a given name.
        Samples start from the top left corner, as many are extracted as the image size permits.

        Args:
            image: CSV line describing an image from the dataset
            size (string) Image size (one of '1', '2', '4', '8' or '16')
            width (int) Sample width
            height (int) Sample height

        Returns:
            (numpy.ndarray) 4D array of images with shape (index, rows, columns, channels)
        """
        img = cv2.imread(os.path.join(DatasetLocation.ROOT, size, image['Name'])).astype(np.float32) / 255
        h, w, _ = img.shape
        samples = np.empty((0, height, width, 3), np.float32)
        for top in range(0, h-height+1, height):
            for left in range(0, w-width+1, width):
                samples = np.concatenate((samples, np.expand_dims(img[top:top+height, left:left+width], axis=0)), axis=0)
        return samples

    def training(self, batch_size):
        """Generator iterating over the training dataset and yielding batches of a given size."""
        num_samples = self.training_size()
        while True:
            data = shuffle(self.data_train)

            i = 0
            x_batch = np.empty((0, Dataset.SAMPLE_HEIGHT, Dataset.SAMPLE_WIDTH, 3), np.float32)
            y_batch = np.empty((0, 2 * Dataset.SAMPLE_HEIGHT, 2 * Dataset.SAMPLE_WIDTH, 3), np.float32)
            for batch in range(0, num_samples // batch_size):
                # gather samples into a batch, considering that there may be multiple samples from one image
                while len(x_batch) < batch_size:
                    x_batch = np.concatenate((x_batch, Dataset.get_samples(data.iloc[i], '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT)), axis=0)
                    y_batch = np.concatenate((y_batch, Dataset.get_samples(data.iloc[i], '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT)), axis=0)
                    i += 1 # move to the next image in the list

                # yield a complete batch and truncate the arrays, leaving the samples not yielded yet
                yield x_batch[:batch_size], y_batch[:batch_size]
                x_batch = x_batch[batch_size:]
                y_batch = y_batch[batch_size:]

    def training_size(self):
        return np.sum(self.data_train.iloc[:]['Samples'])

    def validation(self):
        """Generator iterating over the validation dataset and yielding batches of a size 1."""
        # TODO switch to a sequence/dataset class
        while True:
            for image in self.data_val.iloc:
                x = Dataset.get_samples(image, '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT)
                y = Dataset.get_samples(image, '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT)
                for i in range(0, len(x)):
                    yield x[i:i+1], y[i:i+1] # yield samples in batches of 1

    def validation_size(self):
        return np.sum(self.data_val.iloc[:]['Samples'])

    def validation_arrays(self):
        """Complete validation dataset returned as 1 batch."""
        x_batch = np.empty((0, Dataset.SAMPLE_HEIGHT, Dataset.SAMPLE_WIDTH, 3), np.float32)
        y_batch = np.empty((0, 2 * Dataset.SAMPLE_HEIGHT, 2 * Dataset.SAMPLE_WIDTH, 3), np.float32)
        for image in self.data_val.iloc:
            x_batch = np.concatenate((x_batch, Dataset.get_samples(image, '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT)), axis=0)
            y_batch = np.concatenate((y_batch, Dataset.get_samples(image, '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT)), axis=0)
        return x_batch, y_batch
