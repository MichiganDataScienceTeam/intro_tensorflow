from clouds.dataset import sequence_finder, link_batch_loading, image_batch_loading, batch_assembling
from clouds.augment import link_augment, image_augment


seq_root_dir = '/Users/anders/Movies/clouds'
shape_target = (100, 100)
batch_size = 2
input_frame_length = 4
label_frame_length = 1

sequence_locator = sequence_finder.SequenceLocator()
link_augmenter = link_augment.Augmenter()
image_augmenter = image_augment.Augmenter(shape_target)
link_batch_loader = link_batch_loading.RandomLinkLoader(
    input_frame_length, label_frame_length, sequence_locator, link_augmenter)
image_batch_loader = image_batch_loading.ImageLoader(image_augmenter)
batch_assembler = batch_assembling.BatchAssembler()

sequence_locator.load_sequences(seq_root_dir)
link_batch = link_batch_loader.get_batch(batch_size)
image_batch = image_batch_loader.process_link_batch(link_batch)
train, label = batch_assembler.assemble(image_batch)
print(train.shape)
print(label.shape)

import matplotlib.pyplot as plt
plt.figure()
for i in range(train.shape[0]):
    plt.imshow(train[0, i, :, :, :])
    plt.show()
