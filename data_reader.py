import glob
import h5py
import tensorflow as tf
import numpy as np
from img_utils import get_images


class FileDataReader(object):

    def __init__(self, data_dir, height, width):
        self.data_dir = data_dir
        self.height, self.width = height, width
        self.image_files = glob.glob(data_dir+'*')

    def next_batch(self, batch_size):
        sample_files = np.random.choice(self.image_files, batch_size)
        images = get_images(sample_files, None, None, self.height, self.width)
        return images


class H5DataLoader(object):

    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images_train = data_file['train']
        self.images_test = data_file['test']
        self.epoch = 0
        self.gen_indexes()

    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.images_train.shape[0]))
            self.indexes = range(self.images_train.shape[0])
        else:
            self.indexes = np.array(range(self.images_test.shape[0]))
        self.cur_index = 0

    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = set(self.indexes[self.cur_index:next_index])
        self.cur_index = next_index
        if len(cur_indexes) < batch_size and self.is_train:
            self.gen_indexes()
            while len(cur_indexes) < batch_size:
                cur_len = batch_size - len(cur_indexes)
                self.cur_index += cur_len
                self.epoch += 1
                cur_indexes = cur_indexes.union(
                    set(self.indexes[self.cur_index:self.cur_index+cur_len]))
        cur_indexes = sorted(cur_indexes)
        res = self.images_train[cur_indexes] if self.is_train else self.images_test[cur_indexes]
        return res
