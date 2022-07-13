import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import math
import einops

from typing import Union, Sequence


class PoNo(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-5):
        super(PoNo, self).__init__()
        self.epsilon = tf.Variable(
            epsilon,
            trainable=False,
            dtype=tf.float32,
            name='epsilon'
        )

    def call(self, inputs, *args, **kwargs):
        mean, var = tf.nn.moments(inputs, axis=-1)
        stddev = tf.sqrt(var + self.epsilon)
        return (inputs - mean) / stddev, mean, stddev


class SNConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            kernel_size: Union[Sequence[int], int],
            strides: Sequence[int] = (1, 1),
            padding: str = 'SAME',
            bias: bool = True
    ):
        super(SNConv2D, self).__init__()
        self.n_filters = n_filters
        if len(kernel_size) == 1:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bias = bias

        self.forward = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(
                self.n_filters,
                self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                use_bias=self.bias
            )
        )

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class PoNoResBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            act=tf.keras.layers.LeakyReLU(alpha=.2),
            downsample: bool = False
    ):
        super(PoNoResBlock, self).__init__()
        self.n_filters = n_filters
        self.act = act
        self.downsample = downsample

        self.pono = PoNo()
        self.conv1 = SNConv2D(
            self.n_filters,
            (3, 3)
        )
        self.conv2 = SNConv2D(
            self.n_filters,
            (3, 3)
        )
        self.norm1 = tfa.layers.InstanceNormalization(
            center=False, scale=False
        )
        self.norm2 = tfa.layers.InstanceNormalization(
            center=False, scale=False
        )
        if self.downsample:
            self.conv_skip = SNConv2D(
                self.n_filters,
                (1, 1),
                padding='VALID',
                bias=False
            )

    def forward_shortcut(self, x):
        if self.downsample:
            return K.pool2d(
                self.conv_skip(x), pool_size=(2, 2), strides=(2, 2),
                padding='VALID', pool_mode='avg'
            )
        else:
            return x

    def forward_residual(self, x):
        stats = []

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x, mean1, stddev1 = self.pono(x)
        stats.append((mean1, stddev1))

        x = K.pool2d(
            x, pool_size=(2, 2), strides=(2, 2),
            padding='VALID', pool_mode='avg'
        )
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        x, mean2, stddev2 = self.pono(x)
        stats.append((mean2, stddev2))
        return x, stats

    def call(self, x, *args, **kwargs):
        x_res, stats = self.forward_residual(x)
        x = (x_res + self.forward_shortcut(x)) / math.sqrt(2)
        return x, stats


class MS(tf.keras.layers.Layer):
    def __init__(self):
        super(MS, self).__init__()
        self.conv = SNConv2D(
            n_filters=2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME'
        )

    def call(self, inputs, beta, gamma, *args, **kwargs):
        h = tf.concat([beta, gamma], axis=-1)
        h = self.conv(h)
        mean, stddev = tf.split(h, num_or_size_splits=2, axis=1)
        inputs = (inputs * stddev) + mean
        return inputs


class SPAdaIn(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            norm=tfa.layers.InstanceNormalization
    ):
        super(SPAdaIn, self).__init__()
        self.n_filters = n_filters

        self.norm = norm()
        self.conv_weight = tf.keras.layers.Conv2D(
            self.n_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='VALID'
        )
        self.conv_bias = tf.keras.layers.Conv2D(
            self.n_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='VALID'
        )

    def call(self, x, lr, *args, **kwargs):
        x = self.norm(x)
        weight = self.conv_weight(lr)
        bias = self.conv_bias(lr)
        x = weight * x + bias
        return x


class SPAdaInResBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            act=tf.keras.layers.LeakyReLU(alpha=.2),
            upsample: bool = True
    ):
        super(SPAdaInResBlock, self).__init__()
        self.n_filters = n_filters
        self.act = act
        self.upsample = upsample

        self.conv1 = SNConv2D(
            self.n_filters,
            (3, 3)
        )
        self.conv2 = SNConv2D(
            self.n_filters,
            (3, 3)
        )
        self.norm1 = SPAdaIn(self.n_filters)
        self.norm2 = SPAdaIn(self.n_filters)
        self.ms1 = MS()
        self.ms2 = MS()

        if self.upsample:
            self.conv_skip = SNConv2D(
                self.n_filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='VALID',
                bias=False
            )
            self.bilinear_upsample = tf.keras.layers.UpSampling2D(
                size=(2, 2), interpolation='bilinear'
            )

    def forward_shorcut(self, x):
        if self.upsample:
            return self.bilinear_upsample(self.conv_skip(x))
        else:
            return x

    def forward_residusl(self, x, lr, stats):
        x = self.norm1(x, lr)
        x = self.act(x)
        x = self.conv1(x)
        x = self.ms1(x, stats[1])

        if self.upsample:
            x = self.bilinear_upsample(x)
            lr = self.bilinear_upsample(lr)

        x = self.norm2(x, lr)
        x = self.act(x)
        x = self.conv2(x)
        x = self.ms2(x, stats[0])
        return x

    def call(self, inputs, lr, stats, *args, **kwargs):
        return (self.forward_residusl(inputs, lr, stats) + self.forward_shorcut(x)) / math.sqrt(2)
