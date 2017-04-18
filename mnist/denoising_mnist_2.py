import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
sess = tf.Session()
K.set_session(sess)

from keras.layers import *
from keras.objectives import binary_crossentropy, mean_absolute_error

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

    # model 1
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # encoded = MaxPooling2D((2, 2), padding='same')(x)
    # # at this point the representation is (7, 7, 32)
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
    # preds = Conv2D(1, (3, 3), activation='sigmoid')(x)

    # model 3
    x = Conv2D(64, (3, 3), dilation_rate=(3, 3), activation='relu')(img)  # 22
    x = Conv2D(128, (3, 3), dilation_rate=(3, 3), activation='relu')(x)  # 16
    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)  # 32
    x = Convolution2D(64, (3, 3), activation='relu')(x)
    x = Convolution2D(32, (3, 3), activation='relu')(x)
    preds = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    loss = binary_crossentropy(labels, preds)  # ok
    # loss = mean_absolute_error(labels, preds)  # bad
    # loss = tf.losses.mean_pairwise_squared_error(labels, preds)
    # loss = tf.losses.absolute_difference(labels, preds, weights=labels/1 + 0.1)
    disp_loss = tf.reduce_mean(loss)

    acc_value = tf.reduce_mean(tf.reshape(tf.cast(tf.equal(
        tf.where(tf.greater(preds, 0.5), tf.ones_like(preds), tf.zeros_like(preds)), labels), tf.float32), (-1, 1)))

    train_step = tf.train.AdadeltaOptimizer().minimize(loss)

    # define which variables to capture in summary
    loss_summary = tf.summary.scalar('loss', disp_loss)
    acc_value_summary = tf.summary.scalar('acc', acc_value)
    summary_op = tf.summary.merge_all()

    # save `batch size` images each epoch
    display_batch_size = 4
    display_img_trio = tf.placeholder(tf.float32, shape=(display_batch_size, 28, 3*28, 1))

    # initialize all variables
    log_path = os.path.join(PATH, 'logs3')
    batch_size = 20
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    train_writer = tf.summary.FileWriter(log_path + '/train')
    valid_writer = tf.summary.FileWriter(log_path + '/valid')
    img_writer = tf.summary.FileWriter(log_path + '/images')

    samples_per_epoch = mnist_data.train.labels.shape[0]/10
    print(samples_per_epoch)
    epochs = 100
    assert samples_per_epoch % batch_size == 0, \
        'batch size {} does not divide epoch size {}'.format(batch_size, samples_per_epoch)

    with sess.as_default():
        epoch = 0
        while epoch < epochs:
            print('Starting epoch {}'.format(epoch))
            sampled = 0
            while sampled < samples_per_epoch:
                # train
                batch = mnist_data.train.next_batch(batch_size)
                summary, _ = sess.run([summary_op, train_step], feed_dict={
                    img: np.reshape(make_noisy_batch(batch[0]), (-1, 28, 28, 1)),
                    labels: np.reshape(batch[0], (-1, 28, 28, 1))
                })
                train_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

                # validation
                valid_batch = mnist_data.validation.next_batch(batch_size)
                summary = sess.run(summary_op, feed_dict={
                    img: np.reshape(make_noisy_batch(valid_batch[0]), (-1, 28, 28, 1)),
                    labels: np.reshape(valid_batch[0], (-1, 28, 28, 1))
                })
                valid_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

                sampled += batch_size
            epoch += 1

            # save images per epoch as: input | prediction | groundtruth
            display_img_summary = tf.summary.image(
                'Epoch {:03d}'.format(epoch), display_img_trio, max_outputs=display_batch_size)
            test_image = mnist_data.test.images[0:display_batch_size, :]
            display_input = np.reshape(make_noisy_batch(test_image), (-1, 28, 28, 1))
            display_label = np.reshape(test_image, (-1, 28, 28, 1))
            predictions = sess.run(preds, feed_dict={img: display_input})
            pred_array = np.vstack([p[np.newaxis, :] for p in predictions])
            display_trio = np.concatenate((display_input, pred_array, display_label), axis=2)

            img_summary = sess.run([display_img_summary], feed_dict={display_img_trio: display_trio})[0]
            img_writer.add_summary(img_summary, epoch*samples_per_epoch)


# tensorboard --logdir=`pwd`/mnist/logs
# localhost:6006
