import numpy as np


class BaseBatchMaker(object):

    def __init__(self, sequence_getter, augmenter, seed=0):
        self.sequence_getter = sequence_getter
        self.augmenter = augmenter
        np.random.seed(seed)

    def get_example(self):
        raise NotImplementedError

    def get_batch(self, size):
        batch = []
        while len(batch) < size:
            try:
                example = self.get_example()
                augmented_example = self.augmenter.augment(example)
            except Exception as e:
                continue
            batch.append(augmented_example)
        return batch


class RandomLinkLoader(BaseBatchMaker):
    """
    Clips are loaded as a continuous sequence of frames, split into:
    [ input | labels ]
    """

    def __init__(self, input_frame_length, label_frame_length, *args, **kwargs):
        super(RandomLinkLoader, self).__init__(*args, **kwargs)
        self.input_frame_length = input_frame_length
        self.label_frame_length = label_frame_length

    def get_example(self):
        """

        :return: example{}
        ['train']: training frames
        ['label']: label frames
        """
        sequence = self.sequence_getter.get_random_sequence()
        clip_size = self.input_frame_length + self.label_frame_length
        clip_starting_locs = np.arange(len(sequence) - clip_size)
        starting_loc = np.random.choice(clip_starting_locs)
        example = {
            'train': sequence[starting_loc:starting_loc+self.input_frame_length],
            'label': sequence[starting_loc+self.input_frame_length:starting_loc+clip_size],
        }
        return example


class SampledRandomLinkLoader(BaseBatchMaker):
    """
    Clips are loaded as a continuous sequence of frames, split into:
    [ input | labels ]
    """

    def __init__(self, input_frame_length, label_frame_length, sample_rate, *args, **kwargs):
        super(SampledRandomLinkLoader, self).__init__(*args, **kwargs)
        self.input_frame_length = input_frame_length
        self.label_frame_length = label_frame_length
        self.sample_rate = sample_rate

    def get_example(self):
        """

        :return: example{}
        ['train']: training frames
        ['label']: label frames
        """
        sequence = self.sequence_getter.get_random_sequence()
        clip_size = (self.input_frame_length + self.label_frame_length) * self.sample_rate
        clip_starting_locs = np.arange(len(sequence) - clip_size)
        starting_loc = np.random.choice(clip_starting_locs)

        input_frame_end = starting_loc+self.input_frame_length*self.sample_rate
        example = {
            'train': sequence[starting_loc:input_frame_end:self.sample_rate],
            'label': sequence[input_frame_end:starting_loc+clip_size:self.sample_rate],
        }
        return example
