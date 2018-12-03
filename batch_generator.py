# This code id taken from https://github.com/JGuillaumin/SuperResGAN-keras
import numpy as np
from scipy.misc import imread, imresize, imsave
# from imageio import imread, imresize, imsave
import threading
import math
import os
import random
import matplotlib.pyplot as plt
from glob import glob


def load_img(path, color_mode='rgb', target_size=(256, 256), downscale_factor=4, crop_mode='fixed_size'):
    
    # load the image with the correct color mode
    img_hr = imread(path, mode='RGB')
    if color_mode == 'bgr':
        # RGB -> BGR
        img_hr = img_hr[...,[2,0,1]]
    
    if crop_mode == 'fixed_size':
        # perform random cropping (256,256)
        shape = img_hr.shape[:2]
        off_set_x = random.randint(0, shape[1]-target_size[1])
        off_set_y = random.randint(0, shape[0]-target_size[0])
        img_hr = img_hr[off_set_y:off_set_y+target_size[0], off_set_x:off_set_x+target_size[1]]
    elif crop_mode == 'random_size':
        shape = img_hr.shape[:2]
        short_axis = np.argmin(shape)
        short_edge = shape[short_axis]
        if short_edge > target_size[0] :
            random_size = random.randint(target_size[0], short_edge)
            off_set_x = random.randint(0, shape[1]-random_size)
            off_set_y = random.randint(0, shape[0]-random_size)
            img_hr = img_hr[off_set_y:off_set_y+random_size, off_set_x:off_set_x+random_size]
            img_hr = imresize(img_hr, size=target_size, interp='bicubic')
        else :
            if shape[0]==shape[1]:
                pass
            else:
                off_set = abs(shape[0] - shape[1])
                off_set = random.randint(0, off_set)
                if short_axis == 0:
                    img_hr = img_hr[:, off_set:off_set + short_edge]
                else:
                    img_hr = img_hr[off_set:off_set + short_edge, :]
            img_hr = imresize(img_hr, size=target_size, interp='bicubic')
        
    # add gaussian noise + resize 
    # TODO : check values for mean + sigma ! 
    mean = 0.
    var = 0.01
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,tuple(img_hr.shape))
    gauss = gauss.reshape(img_hr.shape)
    img_lr = img_hr + gauss
    img_lr = imresize(img_lr, size=(target_size[0]//downscale_factor, target_size[1]//downscale_factor))

    return img_lr, img_hr


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
# without 'followlinks' argument
def _count_valid_files_in_directory(directory, white_list_formats):
    """Count files with extension in `white_list_formats` contained in a directory.
    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
# without 'followlinks' argument
def _list_valid_filenames_in_directory(directory, white_list_formats):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.
    # Returns
        filenames: the path of valid files in `directory`
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    filenames = []
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return filenames


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
# only with resize (and optionally cropping) !
class BatchGenerator(object):

    def __init__(self, directory,
                 target_size=(256,256),
                 downscale_factor=4, 
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 color_mode='rgb',
                 crop_mode='fixed_size'):
        """

        :param directory:
        :param target_size:
        :param reshape_mode:
        :param batch_size:
        :param shuffle:
        :param seed:
        :param color_mode:
        """

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        self.directory = directory
        self.target_size = tuple(target_size)
        self.downscale_factor = downscale_factor

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        if color_mode not in {'rgb', 'bgr'}:
            raise ValueError('Invalid color mode:', color_mode, '; expected "rgb" or "bgr".')
        self.color_mode = color_mode
        
        if crop_mode not in {'fixed_size', 'random_size'}:
            raise ValueError('Invalid crop mode :', crop_mode, '; expected "fixed_size" or "random_size".')
        self.crop_mode = crop_mode
        
        # + (3,) : for RGB channels
        self.image_shape_hr = self.target_size + (3,)
        self.image_shape_lr = (self.target_size[0]//self.downscale_factor,
                               self.target_size[1]//self.downscale_factor) + (3,)

        self.samples = _count_valid_files_in_directory(directory, white_list_formats)
        print("Found {} images".format(self.samples))

        self.filenames = _list_valid_filenames_in_directory(directory, white_list_formats)

        self._batch_index = 0
        self._total_batches_seen = 0
        self._lock = threading.Lock()
        self._index_generator = self._flow_index()

    def _flow_index(self):

        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self._total_batches_seen)
            if self._batch_index == 0:
                index_array = np.arange(self.samples)
                if self.shuffle:
                    index_array = np.random.permutation(self.samples)

            current_index = (self._batch_index * self.batch_size) % self.samples
            if self.samples > current_index + self.batch_size:
                current_batch_size = self.batch_size
                self._batch_index += 1
            else:
                self._batch_index = 0
                continue
            self._total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def reset(self):
        self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self._lock:
            index_array, current_index, current_batch_size = next(self._index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        
        batch_hr = np.zeros((current_batch_size,) + self.image_shape_hr, dtype=np.float32)
        batch_lr = np.zeros((current_batch_size,) + self.image_shape_lr, dtype=np.float32)
        
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            batch_lr[i], batch_hr[i] = load_img(os.path.join(self.directory, fname),
                                                color_mode=self.color_mode,
                                                target_size=self.target_size,
                                                downscale_factor=self.downscale_factor,
                                                crop_mode=self.crop_mode)
        
        
        # minimal pre-processing 
        # LR images : scale lr images to [0,1]
        batch_lr = np.divide(batch_lr, 255.) 
        # HR images : scale to [-1, 1]
        batch_hr = np.divide(batch_hr, 127.5) - np.ones_like(batch_hr,dtype=np.float32) 
        
        return batch_lr, batch_hr
