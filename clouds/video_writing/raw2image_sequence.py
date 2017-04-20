"""
# skvideo.io : python2.7
"""
from __future__ import print_function
import skvideo.io
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import skimage.io
from functools import partial


is_display = False
input_dict = {
    'is_display': is_display,
    'is_flip2horizontal': True,
    'shape_target': (100, 100),
}

if is_display:
    fig = plt.figure()
    plt.ion()
    display_data = plt.imshow(np.zeros((1080, 1920, 3)))
    input_dict.update({
        'fig': fig,
        'display_data': display_data,
    })
    plt.show()


def pick_every_kth_frame(i, k):
    return i % k == 0

def mkdir_p(d):
    try:
        os.makedirs(d)
    except OSError as e:
        print(e)

def process_video(video_feed, dest_dir=None, is_display=False, display_data=None, fig=None,
                  is_flip2horizontal=True, shape_target=None, frame_chooser=None):
    for i, frame in enumerate(video_feed):

        if is_flip2horizontal:
            # RGB image as (height, width, channels)
            if frame.shape[1] < frame.shape[0]:
                frame = frame.transpose((1, 0, 2))

        if is_display:
            # display integer-valued image
            display_data.set_data(frame/255.)
            fig.canvas.draw()
            time.sleep(.01)

        if shape_target:
            frame = scipy.misc.imresize(frame, shape_target)

        if dest_dir:
            frame_save_name = '{img_num:06d}.png'.format(img_num=i)
            save_path = os.path.join(dest_dir, frame_save_name)
            if frame_chooser:
                if frame_chooser(i):
                    skimage.io.imsave(save_path, frame)
            else:
                skimage.io.imsave(save_path, frame)
            print('.' if i % 100 != 0 else '.\n', end='')
    print('\n', end='')


def process_single_video(src_dir, movie_name, input_dict, dest_dir=dest_dir):
    """
    For seeing or processing just a single video
    :param src_dir:
    :param movie_name:
    :param input_dict:
    :return:
    """
    mov_path = os.path.join(src_dir, movie_name)
    videogen = skvideo.io.vreader(mov_path, verbosity=0)
    process_video(videogen, dest_dir=dest_dir, **input_dict)


def process_folder(src_dir, dest_dir, ext, input_dict):
    """
    For processing all movies in the folders in `src_dir`
    :param src_dir:
    :param dest_dir:
    :param ext: extension of movies
    :param input_dict:
    :return:
    """
    os.chdir(src_dir)
    for folder_name in os.listdir(src_dir):
        if not os.path.isdir(folder_name):
            print('Skipping {}'.format(folder_name))
            continue
        print('Processing folder {}'.format(folder_name))
        for movie in glob.glob(
                os.path.join(src_dir, folder_name, '*.{}'.format(ext))):
            movie_name = os.path.split(movie[:-len(ext)-1])[-1]
            save_dir = os.path.join(dest_dir, folder_name, movie_name)
            mkdir_p(save_dir)
            print('Processing {}'.format(movie_name))
            process_single_video(
                folder_name,
                '{}.{}'.format(movie_name, ext), input_dict, dest_dir=save_dir)



# single video
src_dir = '/Users/anders/Movies/2015-4-13-08'
dest_dir = '/Users/anders/Movies/clouds/2015-4-13-08_1346'
mov = 'IMG_1348.MOV'
# process_single_video(src_dir, mov, input_dict, dest_dir=dest_dir)


# process folders
src_dir = '/Users/anders/Movies/clouds_raw'
dest_dir = '/Users/anders/Movies/clouds'
ext = 'MOV'

third_frames = partial(pick_every_kth_frame, k=3)
input_dict['frame_chooser'] = third_frames
process_folder(src_dir, dest_dir, ext, input_dict)

