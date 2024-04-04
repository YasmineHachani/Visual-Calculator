import os, sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.io import read_image

import ctypes
import multiprocessing as mp
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from .Transforms import TransformsComposer
from .utils import min_max_normalize, init_array


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    
)

from torchvision.transforms import (
    Lambda,
    Compose,
    Resize,
)

from PIL import Image
from torchvision.utils import save_image
class CachingDataset(Dataset):
    """
        Parameters
        ----------
        data_path : str - Path to the csv containing all the images paths and their labels
        img_size : int - Size of each frames of a image
        caching : boolean - Caching if true
        transform : TransformsComposer - all the transformations to apply to the training set 
        (data augmentation + normalization)

        Returns
        ----------
        CachingDataset : Dataset
    """
    def __init__(self, data_path, img_size, caching, transform=None):

        self.data = pd.read_csv(data_path)
        self.nb_samples = len(self.data)
        self.preprocess = Compose([Resize(img_size)]) #Common preprocessing between train, val and test
        self.transform = transform

        self.caching = caching
        if self.caching:
            shared_array_base = mp.Array(ctypes.c_float, self.nb_samples*3*img_size*img_size)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            shared_array = shared_array.reshape(self.nb_samples, 3, img_size, img_size)
            self.shared_array = torch.from_numpy(shared_array) #Array containing all the cached images
        
            cache_array_base = mp.Array(ctypes.c_int, self.nb_samples)
            cache_array = np.ctypeslib.as_array(cache_array_base.get_obj())
            self.cache = torch.from_numpy(cache_array) #Array containing booleans to indicate if a image is cached or not
            init_array(self.cache)



    def loading_image(self, image_path, label, preprocess=None):
        image = read_image(image_path) #image : C,H,W

        if preprocess:
                image = preprocess(image)

        return {'image': image.float(), 'label': torch.as_tensor(label), 'image_name': image_path.split('/')[-1]}
    
    def __getitem__(self, index):
        data_idx = index
        image_path, label = self.data.iloc[data_idx]['image'], self.data.iloc[data_idx]['label']
        image_name = image_path.split('/')[-1]

        if (self.caching and not(bool(self.cache[data_idx]))):
            print('Filling cache for index {}'.format(index))
            image = self.loading_image(image_path, label, self.preprocess)
            self.shared_array[data_idx, :] = image['image']
            self.cache[data_idx] = 1

        elif self.caching:
            image = {'image':self.shared_array[data_idx], 'label': torch.as_tensor(label), 'image_name': image_name}

        else:
            image = self.loading_image(image_path, label, self.preprocess)
        #save_image(image['image'],image['image_name']+".png")
        if type(self.transform) == TransformsComposer and len(self.transform.transfs) != 0:
                image['image'] = self.transform(image)['image']
                #save_image(image['image'],image['image_name']+"_trans.png")
        elif self.transform == min_max_normalize:
            image['image'] = self.transform(image['image'])

                


        return image
    
    def __len__(self):
        return self.nb_samples


class CachingDataModule(pl.LightningDataModule):
    """
        Parameters
        ----------
        dir_path : str - Path to the directory containing the train, val and test folder where the csv's are
        data_name : str - Name of the csv
        __________________________

        Example:
                - Datasplit
                    - 2_classes
                        - train
                            Split.csv
                        - val
                            Split.csv
                        - test
                            Split.csv
        __________________________

        batch_size : int - Size of a batch
        num_workers : int - Number of parallel processes fetching data
        caching : boolean - Caching if true

        Returns
        ----------
        CachingDataModule : pl.LightningDataModule
    """
    def __init__(self, dir_path, data_name, batch_size, num_workers, caching, augmentation, img_size, **kwargs) :
        super().__init__()
        print('Caching ? : ', caching)
        self._DATA_NAME= data_name
        self._DIR_PATH = dir_path
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers
        self.caching = caching
        self._TRANSFORMS = TransformsComposer(augmentation)
        self.img_size = img_size

    def train_dataloader(self):
        """
        Create the train partition from the list of image labels in {self._DATA_PATH}/train
        """
        print('train_dataloader')
        
        data_path =os.path.join(self._DIR_PATH, "train", self._DATA_NAME)
        print(data_path)
        train_dataset = CachingDataset(
            data_path=data_path,
            transform=self._TRANSFORMS, #Normalize with data augmentation if precised
            img_size= self.img_size,
            caching=self.caching 
        )

        print('Number of images : ', len(train_dataset))
        
        return DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True,
            shuffle=True #shuffle Dataloader at each epoch
        )

    def val_dataloader(self):
        """
        Create the validation partition from the list of image labels in {self._DATA_PATH}/val
        """
        print('val_dataloader')
        data_path = data_path=os.path.join(self._DIR_PATH, "val", self._DATA_NAME)
        val_dataset = CachingDataset(
            data_path=data_path,
            transform= min_max_normalize,#Normalize without data augmentation
            img_size= self.img_size,
            caching=self.caching
        )
        print('Number of images : ', len(val_dataset))

        return DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def test_dataloader(self):
        """
        Create the test partition from the list of image labels in {self._DATA_PATH}/test
        """
        print('test_dataloader')
        data_path = data_path=os.path.join(self._DIR_PATH, "test", self._DATA_NAME)
        test_dataset = CachingDataset(
            data_path=data_path,
            transform=min_max_normalize, #Normalize without data augmentation
            img_size= self.img_size,
            caching=self.caching
        )
        print('Number of images : ', len(test_dataset))
        return DataLoader(
            test_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TransformsComposer.add_specific_args(parser)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--data_name', type=str)
        parser.add_argument('--dir_path', type=str)
        parser.add_argument('--caching', action="store_true")
        parser.add_argument('--img_size', type=int, default=[128, 128])
        return parser