import os
import math
import cv2
from sklearn.utils import shuffle

class Dataset:
    ROOT = './dataset'
    VALIDATION_SPLIT = 0.1
    SAMPLE_WIDTH = 160
    SAMPLE_HEIGHT = 67

    def __init__(self):
        path = os.path.join(Dataset.ROOT, '1')
        filenames = shuffle(os.listdir(path))
        val_idx = math.floor(len(filenames) * Dataset.VALIDATION_SPLIT)
        self.filenames_test = filenames[val_idx:]
        self.filenames_val = filenames[:val_idx]

    def get_samples(name, size, width, height):
        img = cv2.imread(os.path.join(Dataset.ROOT, size, name))
        return img[:height, :width] # TODO get multiple samples if img is larger

    # return pairs of images if batch_size is None or pairs of batches of the given size
    def datagen(filenames, batch_size=None):
        while True:
            filenames = shuffle(filenames)
            if batch_size is None:
                for name in filenames:
                    yield (Dataset.get_samples(name, '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT),
                           Dataset.get_samples(name, '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT))
            else:
                num_samples = len(filenames)
                for start in range(0, num_samples - batch_size + 1, batch_size):
                    x_batch = []
                    y_batch = []
                    for i in range(0, batch_size):
                        x_batch.append(Dataset.get_samples(filenames[start+i], '1', Dataset.SAMPLE_WIDTH, Dataset.SAMPLE_HEIGHT))
                        y_batch.append(Dataset.get_samples(filenames[start+i], '2', 2 * Dataset.SAMPLE_WIDTH, 2 * Dataset.SAMPLE_HEIGHT))
        
                    yield x_batch, y_batch

    def test(self, batch_size):
        return Dataset.datagen(self.filenames_test, batch_size)

    def validation(self):
        return Dataset.datagen(self.filenames_val)