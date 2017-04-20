import os
import glob
import numpy as np


class BaseSequenceLocator(object):

    def load_sequences(self, root):
        raise NotImplementedError

    def get_random_sequence(self):
        raise NotImplementedError


class SequenceLocator(BaseSequenceLocator):
    """
    From root directory, load all sequences
    - root
        - seq1
            - image0
            - image1
            ...
        - seq2
        ...
    """

    def __init__(self, image_ext='png', seed=0):
        self.image_ext = image_ext
        self.sequences = None
        np.random.seed(seed)

    def load_sequences(self, root):
        self.sequences = {}
        for seq_dir in os.listdir(root):
            seq_path = os.path.join(root, seq_dir)
            seq_frames = [img_path for img_path in get_dir_image_paths(seq_path, self.image_ext)]
            self.sequences[seq_dir] = seq_frames

    def get_random_sequence(self):
        assert self.sequences, 'Sequences have not yet been loaded!'
        random_key = np.random.choice(list(self.sequences.keys()))
        return self.sequences[random_key]


class NestedSequenceLocator(BaseSequenceLocator):
    """
    From root directory, load all sequences
    - root
        - folder1
            - seq1
                - image0
                - image1
                ...
            - seq2
                ...
        -folder2
            - seq3
        ...
    """

    def __init__(self, image_ext='png', seed=0):
        self.image_ext = image_ext
        self.sequences = None
        np.random.seed(seed)

    def load_sequences(self, root):
        self.sequences = {}
        for folder in os.listdir(root):
            for seq_dir in os.listdir(os.path.join(root, folder)):
                seq_path = os.path.join(root, folder, seq_dir)
                seq_frames = [img_path for img_path in get_dir_image_paths(seq_path, self.image_ext)]
                self.sequences[seq_path] = seq_frames

    def get_random_sequence(self):
        assert self.sequences, 'Sequences have not yet been loaded!'
        random_key = np.random.choice(list(self.sequences.keys()))
        return self.sequences[random_key]


def get_dir_image_paths(directory, ext):
    for _, file in enumerate(sorted(
            glob.glob(os.path.join(directory, '*.' + ext)),
            key=lambda x: str(x[:-len(ext)-1]))):
        yield file
