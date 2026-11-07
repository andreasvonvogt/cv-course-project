from abc import ABC, abstractmethod
from dataset import Dataset
import numpy as np
import keras
from keras import Model
from keras.layers import Input
import cv2
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [21, 9]

def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-',
                               epochs, h['val_accuracy'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    print('Train Acc     ', h['accuracy'][-1])
    print('Validation Acc', h['val_accuracy'][-1])

def visualize_result(model, x, y):
    # visualize a single result of the model, comparing it with a reference image and with the result of cv2 resizing
    res = model.predict(np.expand_dims(x, axis=0), batch_size=1)[0]
    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)), plt.title('Source image')
    plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(np.clip(cv2.resize(x, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4), 0, 1), cv2.COLOR_BGR2RGB)), plt.title('Source image, resized')
    plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(y, cv2.COLOR_BGR2RGB)), plt.title('Reference image')
    plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)), plt.title('Model result')

class ModelBase(ABC):
    def __init__(self, name):
        self.name = name
        inputs = Input(shape=(Dataset.SAMPLE_HEIGHT, Dataset.SAMPLE_WIDTH, 3))
        outputs = self.create(inputs)
        self.model = Model(inputs, outputs)

    @abstractmethod
    def create(self, inputs):
        pass

    def summary(self):
        self.model.summary()

    def fit(self):
        ds = Dataset()

        epochs = 50
        batch_size = 64
        steps_per_epoch = ds.test_size() // batch_size

        checkpoint = keras.callbacks.ModelCheckpoint(f'{self.name}.keras', monitor='val_loss', save_best_only=True)
        self.model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        history = self.model.fit(ds.test(batch_size),
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=ds.validation(),
                            validation_steps=ds.validation_size(),
                            callbacks=[checkpoint],
                            shuffle=False, verbose=2)
        self.model = keras.models.load_model(f'{self.name}.keras')
        return history

    def predict(self, x, batch_size):
        return self.model.predict(x, batch_size=batch_size)
