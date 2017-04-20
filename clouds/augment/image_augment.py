import numpy as np
import scipy.misc


class Augmenter(object):

    def __init__(self, shape_target):
        self.shape_target = shape_target

    def augment(self, example):
        """

        :param example:
        ['train']: training frame images
        ['label']: label frame images
        [ ... ]: tags for augments
        :return:
        """
        example['train'] = self._resize(example, 'train')
        example['label'] = self._resize(example, 'label')
        return example

    def _resize(self, example, key):  # todo - rescaling as separate
        return [scipy.misc.imresize(img, self.shape_target).astype(np.float32)/255. if img.size > 0
                else np.zeros(self.shape_target) for img in example[key]]
