import tensorflow as tf

from clouds.nets import generating, training


class Predictor(training.Trainer):

    def __init__(self):
        self.g = generating.Generator()
        self.learning_rate = 0.01
        self.beta1 = 0.5

    def feedforward(self, inputs):
        return self.g(inputs, training=True)

    def loss(self, training_input, training_labels):
        generated_output = self.g(training_input, training=True)
        tf.add_to_collection(
            'g_loss',
            tf.losses.mean_squared_error(generated_output, training_labels),
        )
        return tf.add_n(tf.get_collection('g_loss'))

    def train(self, loss):
        g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1)
        g_opt_op = g_opt.minimize(loss, var_list=self.g.variables)
        return g_opt_op
