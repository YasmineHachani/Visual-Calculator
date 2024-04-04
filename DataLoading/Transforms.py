from torchvision import transforms
import torch
from argparse import ArgumentParser
from ipdb import set_trace
import numpy as np
import random

    
class TransformsComposer():
    """
    Compose and setup the transforms depending command line arguments.
    Define a series of transforms, each transform takes a dictionnary
    containing a subset of keys from [ 'image'] and
    has to return the same dictionnary with content elements transformed.
    """
    def __init__(self, augmentation) :
        self.transfs = []
        self.augmentations = TrAugment(augmentation)
        self.transfs.append(self.augmentations)
        self.TrCompose = transforms.Compose(self.transfs)

    def __call__(self, ret) :
        return self.TrCompose(ret)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TrAugment.add_specific_args(parser)
        return parser


class TrAugment() :
    """
    Data augmentation techniques for images

    Args :
        Name augmentation (list str) : data augmentation to return
    """
    def __init__(self, augmentation) :
        self.augs = []
        augs_names =  augmentation
        if (augmentation =='none') or (augmentation ==''):
            pass
        else : 
            for name in augs_names :
                self.interpret_name(name)
        print("Augmentation:", augmentation)
        self.declare()

    def interpret_name(self, name) :
        if 'randombrightness' == name :
            self.augs.append(self.randombrightness)
        elif 'hflip' == name :
            self.augs.append(self.hflip)
        elif 'vflip' == name :
            self.augs.append(self.vflip)
        elif 'fill_background' == name :
            self.augs.append(self.fill_background) 
        elif (name == 'none') or (name=='') :
            pass
        else :
            raise Exception(f'image augmentation {name} is unknown')
        self.augs.append(self.min_max_normalize)

    def __call__(self, ret) :
        """
        Call all augmentations defined in the init
        """
        for aug in self.augs[:-1] :
            p = random.random()
            if p>0.5:
                ret=aug(ret)
        ret = self.augs[-1](ret)
        return ret

    def declare(self):
         print(f'image Transformations : {[aug for aug in self.augs]}')

    @staticmethod
    def randombrightness(ret) :
        """
        Apply a random brightness adjustment to all the image
        Args :
          ret : dictionnary containing at least "image"
        Return :
          ret dictionnary containg image with adjusted brightness
              'image' : (C ,W, H)
        """
        fct = torch.rand(1) + 1
        #ret['image'] += 0.5
        ret['image'] = transforms.functional.adjust_brightness(ret['image'], fct)
        #ret['image'] -= 0.5
        return ret

    @staticmethod
    def hflip(ret) :
        """
        Horizontal Flip
        Args :
          ret : dictionnary containing at least "image"
        Return :
          ret dictionnary with image horizontally flipped the same way for all images
              'image' : (C ,W, H)
        """
        hflipper = transforms.RandomHorizontalFlip(p=0.5)
        ret['image'] = hflipper(ret['image'])
        return ret

    @staticmethod
    def vflip(ret) :

        """
        Vertical Flip
        Args :
          ret : dictionnary containing at least "image"
        Return :
          ret dictionnary with image vertically flipped the same way for all images
              'image' : (Channels ,W, H)
        """
        vflipper = transforms.RandomVerticalFlip(p=0.5)
        ret['image'] = vflipper(ret['image'])
        return ret

    @staticmethod
    def fill_background(ret) :
        """
        fill the background of segmented images with random values
        Args :
          ret : dictionnary containing at least "image"
        Return :
          ret dictionnary with randomized background the same way for all images
              'image' : (C ,W, H)
        """
        for i in range(ret['image'].shape[0]):
            if ret['image'][i,0,:,:][ret['image'][i,0,:,:] == - 0.5].any() and ret['image'][i,1,:,:][ret['image'][i,1,:,:] == - 0.5].any() and ret['image'][i,2,:,:][ret['image'][i,2,:,:] == - 0.5].any() :
                fct=torch.rand(ret['image'][i,0,:,:][ret['image'][i,0,:,:] == - 0.5].shape[0]) -0.5
                ret['image'][i,0,:,:][ret['image'][i,0,:,:] == - 0.5] = fct 
                ret['image'][i,1,:,:][ret['image'][i,1,:,:] == - 0.5] = fct 
                ret['image'][i,2,:,:][ret['image'][i,2,:,:] == - 0.5] = fct 
        return ret
    
    @staticmethod
    def min_max_normalize(ret):
        """
        Normalize a tensor using the min-max normalization formula
        """

        # Step 1: Find the minimum and maximum values 
        min_value = torch.amin(ret['image'], dim=(1,2), keepdim=True)
        max_value = torch.amax(ret['image'], dim=(1,2), keepdim=True)

        # Step 2: Standardize the tensor using the min-max normalization formula
        ret['image'] = (ret['image'] - min_value) / (max_value - min_value)

        return ret
    

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--augmentation',type=str,nargs='+', default='none')
        return parser


