import numpy as np

class BatchAssembler(object):

    def __init__(self):
        pass

    def assemble(self, batch):
        """

        :param batch: list of example:
        ['train']: stacked training frame clip
        ['label']: stacked label frame clip
        [ ... ]: tags for augments
        :return:
        train [batch_size:train-clip-dims]
        label [batch_size:label-clip-dims]
        """
        train = np.vstack([example['train'][np.newaxis, :] for example in batch])
        label = np.vstack([example['label'][np.newaxis, :] for example in batch])
        return train, label
