import os
import math
import cv2
from sklearn.utils import shuffle

class Dataset:
    ROOT = './dataset'
    VALIDATION_SPLIT = 0.1

    def __init__(self):
        path = os.path.join(Dataset.ROOT, '1')
        filenames = shuffle(os.listdir(path))
        val_idx = math.floor(len(filenames) * Dataset.VALIDATION_SPLIT)
        self.filenames_test = filenames[val_idx:]
        self.filenames_val = filenames[:val_idx]

    def datagen(filenames, batch_size):
        num_samples = len(filenames)
        while True:
            filenames = shuffle(filenames)
            for start in range(0, num_samples - batch_size + 1, batch_size):
                x_batch = []
                y_batch = []
                for i in range(0, batch_size):
                    x_batch.append(cv2.imread(os.path.join(Dataset.ROOT, '1', filenames[start+i])))
                    y_batch.append(cv2.imread(os.path.join(Dataset.ROOT, '2', filenames[start+i])))
    
                yield x_batch, y_batch

    def test(self, batch_size):
        return Dataset.datagen(self.filenames_test, batch_size)

    def validation(self, batch_size):
        return Dataset.datagen(self.filenames_val, batch_size)