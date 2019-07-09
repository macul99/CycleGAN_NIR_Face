import os.path
import os
from os import mkdir, makedirs, rename, listdir
from os.path import join, exists, relpath, abspath
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

class PairedNirDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt, seed=100):
        BaseDataset.__init__(self, opt)
        if self.opt.dataset_name == 'oulu':
            self.__init_oulu()
        else:
            raise NotImplementedError
        random.seed(seed)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, Image.open(self.A_path[0]).convert('RGB').size)
        self.A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        transform_params = get_params(self.opt, Image.open(self.B_path[0]).convert('RGB').size)
        self.B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

    def __init_oulu(self):
        dir_A = join(self.root, 'NI/Strong')
        dir_B = join(self.root, 'VL/Strong')
        ds_A = sorted(make_dataset(dir_A, self.opt.max_dataset_size))
        ds_B = sorted(make_dataset(dir_B, self.opt.max_dataset_size))
        assert len(ds_A)>0 and len(ds_B)>0
        self.input_nc = 3
        self.output_nc = 3        
        self.A_path = []
        self.B_path = []        
        self.label = []

        idx = ds_A[0].split('/').index('NI')
        for imp in ds_A:
            spt = imp.split('/')
            spt[0]='/'
            spt[idx]='VL'
            imp1 = join(*spt)
            if imp1 in ds_B:
                self.A_path.append(imp)
                self.B_path.append(imp1)
                self.label.append(int(spt[idx+2][1:])-1)

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
        A_path = self.A_path[index]
        B_path = self.B_path[index]
        label = self.label[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')        

        # apply the same transform to both A and B

        #transform_params = get_params(self.opt, A.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #transform_params = get_params(self.opt, B.size)
        #B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = self.A_transform(A)
        B = self.B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.label)