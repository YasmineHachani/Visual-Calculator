import torch
import random

#Useful function

def init_array(array):
    for i in range(array.shape[0]):
        array[i] = 0

#Standardization function
    
def min_max_normalize(tensor):
    """
    Normalize a tensor using the min-max normalization formula
    """
    # Step 1: Find the minimum and maximum values 
    min_value = torch.amin(tensor, dim=(1,2), keepdim=True)
    max_value = torch.amax(tensor, dim=(1,2), keepdim=True)

    # Step 2: Standardize the tensor using the min-max normalization formula
    normalized_tensor = (tensor - min_value) / (max_value - min_value)

    return normalized_tensor

# Data augmentation function
    
class AddGaussianNoise(object):
    """
    Class to add Gaussian noise

    Parameters
        ----------
        mean : float - Mean of the gaussian noise
        std : float - Standard deviation of the gaussian noise
    """
    def __init__(self, mean=0., std=1.):
        self.std = random.uniform(1, std)
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)