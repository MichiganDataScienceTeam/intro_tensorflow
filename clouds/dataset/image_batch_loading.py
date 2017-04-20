"""
Augmenter does the resizing+pad-filling
"""
import numpy as np
import skimage.io


class ImageLoader(object):

    def __init__(self, augmenter, padding=-1, seed=0):
        self.padding = padding
        self.augmenter = augmenter
        np.random.seed(seed)

    def process_links(self, example):
        """

        :param example
        ['train']: training frame links
        ['label']: label frame links
        [ ... ]: tags for augments
        :return: augmented_example
        ['train']: stacked training frame clip
        ['label']: stacked label frame clip
        [ ... ]: tags for augments
        """
        example['train'] = self._load_images(example['train'])
        example['label'] = self._load_images(example['label'])
        augmented_example = self.augmenter.augment(example)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in augmented_images:
        #     plt.imshow(i)
        #     plt.show()

        # note: Tensorflow dim ordering
        augmented_example['train'] = self._stack_images(augmented_example, 'train')
        augmented_example['label'] = self._stack_images(augmented_example, 'label')
        return augmented_example

    def _load_images(self, links):
        return [skimage.io.imread(link) if link != self.padding else np.array([]) for link in links]

    @staticmethod
    def _stack_images(example, key):
        return np.vstack([img[np.newaxis, :] for img in example[key]])

    def process_link_batch(self, batch):
        clip_batch = [self.process_links(links) for links in batch]
        return clip_batch
