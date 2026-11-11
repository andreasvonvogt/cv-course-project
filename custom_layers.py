import keras
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Concatenate, MaxPooling2D

@keras.saving.register_keras_serializable(package='super_resolution')
class Conv2Dx2Block(keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.c1 = Conv2D(self.filters, kernel_size=(self.kernel_size, self.kernel_size), padding='same')
        self.c2 = Conv2D(self.filters, kernel_size=(self.kernel_size, self.kernel_size), padding='same')
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.relu1 = Activation("relu")
        self.relu2 = Activation("relu")

    def build(self, input_shape):
        self.c1.build(input_shape)
        shape = self.c1.compute_output_shape(input_shape)

        self.bn1.build(shape)
        self.relu1.build(shape)

        self.c2.build(shape)
        shape = self.c2.compute_output_shape(shape)

        self.bn2.build(shape)
        self.relu2.build(shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.filters)

    def call(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        }
        return {**base_config, **config}

@keras.saving.register_keras_serializable(package='super_resolution')
class DownscaleBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.cx2 = Conv2Dx2Block(self.filters, self.kernel_size)
        self.pool = MaxPooling2D(pool_size=(2, 2))

    def build(self, input_shape):
        self.cx2.build(input_shape)
        shape = self.cx2.compute_output_shape(input_shape)
        self.pool.build(shape)

        super().build(input_shape)

    def call(self, x):
        x = self.cx2(x)
        p = self.pool(x)
        return x, p

    def get_config(self):
        base_config = super().get_config()
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        }
        return {**base_config, **config}

@keras.saving.register_keras_serializable(package='super_resolution')
class UpscaleBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, concat=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.concat = concat
        self.kernel_size = kernel_size

        self.ct = Conv2DTranspose(self.filters, self.kernel_size, strides=2, padding='same')
        self.bn = BatchNormalization()
        self.relu = Activation("relu")
        self.cat = Concatenate()
        self.cx2 = Conv2Dx2Block(self.filters, self.kernel_size)

    def build(self, input_shape):
        self.ct.build(input_shape)
        shape = self.ct.compute_output_shape(input_shape)

        self.bn.build(shape)
        self.relu.build(shape)
        if self.concat:
            self.cat.build([shape, shape])
            shape = self.cat.compute_output_shape([shape, shape])

        self.cx2.build(shape)

        super().build(input_shape)

    def call(self, x, skip=None):
        x = self.ct(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.concat and skip is not None:
            x = self.cat([x, skip])
        x = self.cx2(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "filters": self.filters,
            "concat": self.concat,
            "kernel_size": self.kernel_size,
        }
        return {**base_config, **config}
