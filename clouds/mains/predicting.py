import os
import numpy as np

import tensorflow as tf

from keras import backend as K
sess = tf.Session()
K.set_session(sess)

from clouds.nets import prediction_training
from clouds.dataset import sequence_finder, link_batch_loading, image_batch_loading, batch_assembling
from clouds.augment import link_augment, image_augment

PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    seq_root_dir = '/Users/CyrusAnderson/Movies/clouds'
    shape_target = (128, 128)
    batch_size = 10
    input_frame_length = 4  # 1 for 2D model
    label_frame_length = 1
    input_data = tf.placeholder(tf.float32, shape=(None, input_frame_length, *shape_target, 3))
    labels = tf.placeholder(tf.float32, shape=(None, label_frame_length, *shape_target, 3))
    # input_data = tf.placeholder(tf.float32, shape=(None, *shape_target, 3), name='input_data')  # 2D model
    # labels = tf.placeholder(tf.float32, shape=(None, *shape_target, 3), name='labels')

    sequence_locator = sequence_finder.NestedSequenceLocator()
    link_augmenter = link_augment.Augmenter()
    image_augmenter = image_augment.Augmenter(shape_target)
    link_batch_loader = link_batch_loading.RandomLinkLoader(
        input_frame_length, label_frame_length, sequence_locator, link_augmenter)
    link_batch_loader = link_batch_loading.SampledRandomLinkLoader(
        input_frame_length, label_frame_length, 30, sequence_locator, link_augmenter
    )
    image_batch_loader = image_batch_loading.ImageLoader(image_augmenter)
    batch_assembler = batch_assembling.BatchAssembler()
    sequence_locator.load_sequences(seq_root_dir)

    def batch_generator():
        while True:
            link_batch = link_batch_loader.get_batch(batch_size)
            image_batch = image_batch_loader.process_link_batch(link_batch)
            train, label = batch_assembler.assemble(image_batch)
            yield train, label

    data_generator = batch_generator()
    generator = prediction_training.Predictor()
    feedforward = generator.feedforward(input_data)
    loss = generator.loss(feedforward, labels)
    train_op = generator.train(loss)

    # summary setup
    loss_summary = tf.summary.scalar('loss_summary', tf.reduce_mean(loss))
    summary_op = tf.summary.merge_all()

    # save `batch size` images each epoch
    display_batch_size = 1
    display_img_trio = tf.placeholder(
        tf.float32, shape=(display_batch_size, shape_target[0], 3*shape_target[1], 3), name='image_display')

    # initialize loggers
    log_path = os.path.join(PATH, 'logs2')
    train_writer = tf.summary.FileWriter(log_path + '/train')
    valid_writer = tf.summary.FileWriter(log_path + '/valid')
    img_writer = tf.summary.FileWriter(log_path + '/images')

    epochs = 2000
    samples_per_epoch = 40  # 100
    assert samples_per_epoch % batch_size == 0, \
            'batch size {} does not divide epoch size {}'.format(batch_size, samples_per_epoch)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # for graph vis
        graph_writer = tf.summary.FileWriter(log_path + '/graph', sess.graph)
        graph_writer.add_graph(sess.graph)
        graph_writer.close()

        epoch = 0
        sampled = 0
        while epoch < epochs:
            print('Starting epoch {}'.format(epoch))
            sampled = 0
            while sampled < samples_per_epoch:

                train_data_batch, label_batch = data_generator.__next__()
                print(train_data_batch[:, 0:1, :, :, :].shape)

                summary, loss_val, _ = sess.run([summary_op, loss, train_op], feed_dict={
                    input_data: train_data_batch,  # [:, 0, :, :, :], for 2D model
                    labels: label_batch,  # [:, 0, :, :, :],
                    K.learning_phase(): 1,
                })
                train_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

                sampled += batch_size
            epoch += 1

            # save images per epoch as: input | prediction | groundtruth
            display_img_summary = tf.summary.image(
                'Epoch {:03d}'.format(epoch), display_img_trio, max_outputs=display_batch_size)
            generated, loss_val = sess.run([feedforward, loss], feed_dict={
                input_data: train_data_batch[0:1, ...],  # [0:1, 0, :, :, :],
                labels: label_batch[0:1, ...],  # [0:1, 0, :, :, :],
                K.learning_phase(): 0,
            })
            pred_array = np.vstack([p[np.newaxis, ...] for p in generated])
            pred_array = np.clip(pred_array, 0, 1)

            display_trio = np.concatenate((
                np.expand_dims(train_data_batch[0, 0, ...], axis=0),
                np.expand_dims(pred_array[0, 0, ...], axis=0),
                np.expand_dims(label_batch[0, 0, ...], axis=0)
            ), axis=2)
            # 2D model
            # display_trio = np.concatenate((
            #     np.expand_dims(train_data_batch[0, ...], axis=0),
            #     np.expand_dims(pred_array[0, ...], axis=0),
            #     np.expand_dims(label_batch[0, ...], axis=0),
            # ), axis=2)
            display_trio = (display_trio*255).astype(np.uint8)

            img_summary = sess.run([display_img_summary], feed_dict={display_img_trio: display_trio})[0]
            img_writer.add_summary(img_summary, epoch*samples_per_epoch)


