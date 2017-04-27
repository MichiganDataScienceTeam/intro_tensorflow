import tensorflow as tf

from clouds.nets import generating, training


class Predictor(training.Trainer):

    def __init__(self):
        self.g = generating.Generator()
        self.learning_rate = 0.001
        self.beta1 = 0.9

    def feedforward(self, inputs):
        return self.g(inputs, training=True)

    def loss(self, feedforward_output, training_labels):
        tf.add_to_collection(
            'g_loss',
            tf.reduce_mean(tf.squared_difference(feedforward_output, training_labels), axis=0),
        )
        return tf.add_n(tf.get_collection('g_loss'), name='total_loss')

    def train(self, loss):
        g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, epsilon=.01)
        return g_opt.minimize(loss, var_list=self.g.variables)
