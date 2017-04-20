
class Trainer(object):

    def loss(self, training_input, training_labels):
        raise NotImplementedError

    def train(self, loss):
        raise NotImplementedError
