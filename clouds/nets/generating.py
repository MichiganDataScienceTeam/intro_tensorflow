import os

import tensorflow as tf

from keras import backend as K
sess = tf.Session()
K.set_session(sess)

from keras.layers import *


class Generator(object):
    def __init__(self):
        self.reuse = False

    def __call__(self, inputs, training=False):
        with tf.variable_scope('g', reuse=self.reuse):

            x = Conv3D(64, (3, 3, 3), dilation_rate=(3, 3, 3), padding='same')(inputs)
            x = BatchNormalization(scale=False)(x)
            # x = Dropout(0.1)(x)
            x = SpatialDropout3D(0.1)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv3D(64, (3, 3, 3), strides=(2, 1, 1), padding='same', use_bias=False)(x)
            x = BatchNormalization(scale=False)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv3D(64, (3, 3, 3), strides=(2, 1, 1), padding='same', use_bias=False)(x)  # 2 -> 1
            x = BatchNormalization(scale=False)(x)
            x = LeakyReLU(alpha=0.2)(x)
            # x = Conv2D(64, (3, 3), dilation_rate=(3, 3), activation='relu')(inputs)  # 22
            # x = Conv2D(128, (3, 3), dilation_rate=(3, 3), activation='relu')(x)  # 16
            # x = Conv2DTranspose(128, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)  # 32
            # x = Convolution2D(64, (3, 3), activation='relu')(x)
            # x = Convolution2D(32, (3, 3), activation='relu')(x)
            outputs = Conv3D(3, (3, 3, 3), activation='sigmoid', padding='same', use_bias=True)(x)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs
