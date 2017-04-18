import os
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
sess = tf.Session()
K.set_session(sess)

from keras.layers import *
from keras.metrics import *
from keras.objectives import binary_crossentropy, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))

def make_noisy_batch(batch):
    noise_factor = 0.5
    noisy = batch + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=batch.shape)
    noisy = np.clip(noisy, 0., 1.)
    return noisy


if __name__ == '__main__':

    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

    img = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

    # model 1 (
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # encoded = MaxPooling2D((2, 2), padding='same')(x)
    #
    # # at this point the representation is (7, 7, 32)
    #
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # preds = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # model 2
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
    # x = MaxPooling2D()(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(3, 3), activation='relu')(x)
    # x = Conv2DTranspose(32, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)
    # x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Convolution2D(32, (3, 3), dilation_rate=(3, 3), activation='relu')(x)
    # preds = Conv2D(1, (3, 3), activation='sigmoid')(x)  # todo rename?

    # model 3
    x = Conv2D(64, (3, 3), dilation_rate=(3, 3), activation='relu')(img)  # 22
    x = Conv2D(128, (3, 3), dilation_rate=(3, 3), activation='relu')(x)  # 16
    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)  # 32
    x = Convolution2D(64, (3, 3), activation='relu')(x)
    x = Convolution2D(32, (3, 3), activation='relu')(x)
    preds = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    # preds = Conv2D(2, (3, 3), activation='relu', padding='same')(x)

    # loss = tf.reduce_mean(binary_crossentropy(labels, preds))
    # loss = tf.reduce_mean(mean_absolute_error(labels, preds))
    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels, preds, weights=labels/2 + 0.5))

    # define performance metrics
    # acc_value = tf.reduce_mean(tf.reshape(tf.cast(tf.equal(tf.round(labels), tf.round(preds)), tf.float32), (-1, 1)))
    # acc_value = tf.reduce_mean(tf.reshape(tf.cast(tf.less(tf.abs(labels-preds), 0.01), tf.float32), (-1, 1)))
    # tf.equal(tf.greater(preds, 0.5), labels
    acc_value = tf.reduce_mean(tf.reshape(tf.cast(tf.equal(
        tf.where(tf.greater(preds, 0.5), tf.ones_like(preds), tf.zeros_like(preds)), labels), tf.float32), (-1, 1)))

    # define which variables to capture in summary
    loss_summary = tf.summary.scalar('loss', loss)
    acc_value_summary = tf.summary.scalar('acc', acc_value)
    # add images to the summary
    display_batch_size = 4
    display_img_trio = tf.placeholder(tf.float32, shape=(display_batch_size, 28, 3*28, 1))
    summary_op = tf.summary.merge_all()

    # choose optimizer
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_step = tf.train.AdadeltaOptimizer().minimize(loss)   #GradientDescentOptimizer(0.1).minimize(loss)
    # train_step = tf.train.MomentumOptimizer(.01, .9).minimize(loss)

    # initialize all variables
    log_path = os.path.join(PATH, 'logs20')
    batch_size = 20
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    train_writer = tf.summary.FileWriter(log_path + '/train')
    valid_writer = tf.summary.FileWriter(log_path + '/valid')
    # test_writer = tf.summary.FileWriter(log_path + '/test')  # too much mem...
    img_writer = tf.summary.FileWriter(log_path + '/images')

    samples_per_epoch = mnist_data.train.labels.shape[0]
    epochs = 50
    assert samples_per_epoch % batch_size == 0, \
        'batch size {} does not divide epoch size {}'.format(batch_size, samples_per_epoch)

    # run training loop
    with sess.as_default():
        epoch = 0
        while epoch < epochs:
            print('Starting epoch {}'.format(epoch))
            sampled = 0
            while sampled < samples_per_epoch:
                # break  # todo deb
                # train
                batch = mnist_data.train.next_batch(batch_size)
                summary, _ = sess.run([summary_op, train_step], feed_dict={
                    img: np.reshape(make_noisy_batch(batch[0]), (-1, 28, 28, 1)),
                    labels: np.reshape(batch[0], (-1, 28, 28, 1))
                })
                train_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

                # validation
                valid_batch = mnist_data.validation.next_batch(batch_size)
                summary, _, _ = sess.run([summary_op, acc_value, loss], feed_dict={
                    img: np.reshape(make_noisy_batch(valid_batch[0]), (-1, 28, 28, 1)),
                    labels: np.reshape(valid_batch[0], (-1, 28, 28, 1))
                })
                valid_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

                sampled += batch_size
            epoch += 1

            # test
            # summary, _, _ = sess.run([summary_op, acc_value, loss], feed_dict={
            #     img: np.reshape(make_noisy_batch(mnist_data.test.images), (-1, 28, 28, 1)),
            #     labels: np.reshape(mnist_data.test.images, (-1, 28, 28, 1))
            # })
            # test_writer.add_summary(summary, epoch*samples_per_epoch)

            display_img_summary = tf.summary.image(
                'Epoch {:03d}'.format(epoch), display_img_trio, max_outputs=display_batch_size)
            print(mnist_data.test.images.shape)
            test_image = mnist_data.test.images[0:display_batch_size, :]
            print('~~')
            print(test_image.shape)
            display_input = np.reshape(make_noisy_batch(test_image), (-1, 28, 28, 1))
            display_label = np.reshape(test_image, (-1, 28, 28, 1))
            predictions = sess.run(preds, feed_dict={img: display_input})
            # print(predictions[0])
            # print(display_input.shape)
            print(predictions[0].shape)
            print(len(predictions))
            pred_array = np.vstack([p[np.newaxis, :] for p in predictions])
            print(pred_array.shape)
            # print(display_label.shape)
            print('=')
            display_trio = np.concatenate((display_input, pred_array, display_label), axis=2)

                # display_trio = np.concatenate((display_input, np.expand_dims(predictions[0], axis=0), display_label), axis=2)
                # print(display_trio.shape)

            img_summary = sess.run([display_img_summary], feed_dict={display_img_trio: display_trio})[0]
            img_writer.add_summary(img_summary, epoch*samples_per_epoch)  # deb, add real epoch+steps in later

            # exit(1) # todo deb



# tensorboard --logdir=`pwd`/mnist/logs
# localhost:6006
