import os.path
import os
from os import mkdir, makedirs, rename, listdir
from os.path import join, exists, relpath, abspath
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

class UnpairedNirDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt, seed=100):
        BaseDataset.__init__(self, opt)
        if self.opt.dataset_name == 'casia':
            self.__init_casia()
        else:
            raise NotImplementedError
        random.seed(seed)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, Image.open(self.A_path[0]).convert('RGB').size)
        self.A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        transform_params = get_params(self.opt, Image.open(self.B_path[0]).convert('RGB').size)
        self.B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

    def __init_casia(self):
        dir_A = join(self.root, 'NIR')
        dir_B = join(self.root, 'VIS')
        self.A_path = sorted(make_dataset(dir_A, self.opt.max_dataset_size))
        self.B_path = sorted(make_dataset(dir_B, self.opt.max_dataset_size))
        assert len(self.A_path)>0 and len(self.B_path)>0        
        self.input_nc = 3
        self.output_nc = 3
        self.A_label = []
        self.B_label = []        
        name_list = []
        for p in self.A_path:
            name_list.append(p.split('/')[-1].split('_')[-2])
        for p in self.B_path:
            name_list.append(p.split('/')[-1].split('_')[-2])
        name_list = sorted(list(set(name_list)))
        for p in self.A_path:
            self.A_label.append(name_list.index(p.split('/')[-1].split('_')[-2]))
        for p in self.B_path:
            self.B_label.append(name_list.index(p.split('/')[-1].split('_')[-2]))
        self.A_size = len(self.A_path)
        self.B_size = len(self.B_path)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        index_A = index % self.A_size
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_path[index_A]  # make sure index is within then range
        B_path = self.B_path[index_B]
        A_label = self.A_label[index_A]
        B_label = self.B_label[index_B]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, A.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #transform_params = get_params(self.opt, B.size)
        #B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = self.A_transform(A)
        B = self.B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_label': A_label, 'B_label': B_label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)