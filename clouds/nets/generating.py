import tensorflow as tf

from keras.layers import *
from keras.layers.merge import *


class Generator(object):
    def __init__(self):
        self.reuse = False

    def __call__(self, inputs, training=False):
        with tf.variable_scope('g', reuse=self.reuse):

            # 2D model
            # x = Conv2D(12, (3, 3), padding='same', use_bias=True)(inputs)
            # x = LeakyReLU(alpha=0.2)(x)
            # x = Conv2D(24, (3, 3), padding='same', use_bias=True)(x)
            # x = LeakyReLU(alpha=0.2)(x)
            # x = Conv2D(48, (3, 3), padding='same', use_bias=True)(x)
            # x = LeakyReLU(alpha=0.2)(x)
            # x = Conv2D(48, (3, 3), padding='same', use_bias=True)(x)
            # x = LeakyReLU(alpha=0.2)(x)
            # x = Conv2D(24, (3, 3), padding='same', use_bias=True)(x)
            # x = LeakyReLU(alpha=0.2)(x)
            # x = Conv2D(12, (3, 3), padding='same', use_bias=True)(x)
            # x = LeakyReLU(alpha=0.2)(x)
            # outputs = Conv2D(3, (3, 3), padding='same', use_bias=True)(x)

            # 3D model
            x = Conv3D(3, (3, 3, 3), padding='same', use_bias=True)(inputs)
            x = LeakyReLU(alpha=0.2)(x)
            x = MaxPool3D(pool_size=(4, 2, 2))(x)

            x = Conv3D(3, (1, 3, 3), padding='same', activation='relu', use_bias=True)(x)

            x = Conv3D(3, (1, 3, 3), padding='same', use_bias=True)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = MaxPool3D(pool_size=(1, 2, 2))(x)

            x = Conv3D(3, (1, 3, 3), padding='same', activation='relu', use_bias=True)(x)

            x = Conv3D(3, (1, 3, 3), padding='same', use_bias=True)(x)
            x = LeakyReLU(alpha=0.2)(x)

            x = Conv3D(3, (1, 3, 3), padding='same', activation='relu', use_bias=True)(x)

            x = UpSampling3D(size=(1, 2, 2))(x)
            x = Conv3D(3, (1, 3, 3), padding='same', use_bias=True)(x)
            x = LeakyReLU(alpha=0.2)(x)

            x = UpSampling3D(size=(1, 2, 2))(x)
            x = Conv3D(3, (1, 3, 3), padding='same', use_bias=True)(x)
            x = LeakyReLU(alpha=0.2)(x)

            # no rgb-split model
            # outputs = Conv3D(3, (1, 3, 3), padding='same', activation='relu')(x)
            #
            r = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid', use_bias=True)(x)
            g = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid', use_bias=True)(x)
            b = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid', use_bias=True)(x)
            merge_rgb = concatenate([r, g, b], axis=-1)
            outputs = merge_rgb

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs
