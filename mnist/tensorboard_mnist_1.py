import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
sess = tf.Session()
K.set_session(sess)

from keras.layers import *
from keras.metrics import *
from keras.objectives import categorical_crossentropy


PATH = os.path.dirname(os.path.abspath(__file__))

# save downloaded data to a directory named MNIST_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# mnist images are 28**2 = 784 pixels, and 10 classes
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

# make model
x = Dense(32, activation='relu')(img)
preds = Dense(10, activation='softmax')(x)

# define loss
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# define performance metrics
acc_value = tf.reduce_mean(categorical_accuracy(labels, preds))

# define which variables to capture in summary
loss_summary = tf.summary.scalar('loss', loss)
acc_value_summary = tf.summary.scalar('acc', acc_value)
summary_op = tf.summary.merge_all()

# choose optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# initialize all variables
log_path = os.path.join(PATH, 'logs')
batch_size = 50
init_op = tf.global_variables_initializer()
sess.run(init_op)
train_writer = tf.summary.FileWriter(log_path + '/train', graph=sess.graph)
valid_writer = tf.summary.FileWriter(log_path + '/valid', graph=sess.graph)
test_writer = tf.summary.FileWriter(log_path + '/test', graph=sess.graph)

samples_per_epoch = mnist_data.train.labels.shape[0]  # 1 pass over dataset
epochs = 10
assert samples_per_epoch % batch_size == 0, \
    'batch size {} does not divide epoch size {}'.format(batch_size, samples_per_epoch)

# run training loop
with sess.as_default():
    epoch = 0
    while epoch < epochs:
        print('Starting epoch {}'.format(epoch))
        sampled = 0
        while sampled < samples_per_epoch:
            # train
            batch = mnist_data.train.next_batch(batch_size)
            summary, _ = sess.run([summary_op, train_step], feed_dict={
                img: batch[0],
                labels: batch[1]
            })
            train_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

            # validation
            valid_batch = mnist_data.validation.next_batch(batch_size)
            summary, _, _ = sess.run([summary_op, acc_value, loss], feed_dict={
                img: valid_batch[0],
                labels: valid_batch[1]
            })
            valid_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

            sampled += batch_size
        epoch += 1

        # test
        summary, _, _ = sess.run([summary_op, acc_value, loss], feed_dict={
            img: mnist_data.test.images,
            labels: mnist_data.test.labels
        })
        test_writer.add_summary(summary, epoch*samples_per_epoch)


# tensorboard --logdir=`pwd`/mnist/logs
# localhost:6006
