"""
[Reusable][PyTorch]

Processors and the wrapper hepler function to pre-process dataset in pytorch way.

Functions and classes should be called in load_dataset.py file.

Created by Zhaoyan @ UCL
"""

import numpy as np
from torch.utils.data import Dataset
import torch
from datetime import datetime


class CustomDataset(Dataset):
    """Custom Dataset from a saved dictionary"""
    def __init__(self, list_inputs, list_labels=[], length=-1):
        """

        :param list_inputs: (list) The list of sample lists/arrays.
        :param list_labels: (optional, list) The list of sample lists/arrays.
                            Can be empty list (For dataset that has no ground-truth labels.)
        """
        super(CustomDataset, self).__init__()
        self.list_inputs = list_inputs
        self.list_labels = list_labels
        self.length = len(self.list_inputs[0])
        if length == -1 or length > self.length:
            pass
            print(f'{datetime.now()} I Using full dataset. Length: {self.length}')
        else:
            self.length = length
            self.list_inputs = self.list_inputs[:self.length]
            self.list_labels = self.list_labels[:self.length]
            print(f'{datetime.now()} I Using partial dataset. Length: {self.length}')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_inputs = []
        sample_labels = []
        for i_list_inputs in self.list_inputs:
            sample_inputs.append(i_list_inputs[idx])
        for i_list_labels in self.list_labels:
            sample_labels.append(i_list_labels[idx])
        # DEMO CODE:
        # sample_inputs, sample_labels = self.dataset[idx]
        return sample_inputs, sample_labels


def data_processor_wrapper(DatasetClass, processor):
    """
    A wrapper to process the dataset in a PyTorch way.
    :param DatasetClass: A PyTorch Dataset Class
    :param processor: The processor that is going to be applied on the data samples.
            It should be a function, or an object whose __call__ function is well-defined.
            Arguments: input sample,
                       label sample,
                       the index of this sample in the whole dataset
    :return: An iterable dataset that each sample been processed by the processor.
    """
    class ProcessedDataset(DatasetClass):
        def __getitem__(self, index):
            input, label = super().__getitem__(index)
            processed_input, processed_label = processor(input, label, index)
            return processed_input, processed_label
    return ProcessedDataset


'''
Processor Classes
'''


class Processor_Denoising_AddGau:
    """
    For RIB

    For denoising task.
    Add additive Gaussian noise on the image to make noisy inputs.
    """
    def __init__(self, mu, sigma, deterministic=True, grayscale=False):
        """

        :param mu:
        :param sigma:
        :param deterministic: (bool, optional) Whether to generate deterministic noise for each sample.
        :param grayscale: (bool, optional) Whether to generate noise on different channels uniformly.
        """
        self.mu = mu
        self.sigma = sigma
        self.deterministic = deterministic
        self.grayscale = grayscale

    def __call__(self, input, label, index):
        if self.deterministic:
            np.random.seed(index)
        noise_shape = [1] + list(input.shape)  # Add one dimensional to compate with the DIBLoss model setup.
        if self.grayscale:
            noise_shape[1] = 1
            noise = np.float32(np.random.normal(self.mu, self.sigma, noise_shape))
            noise = np.repeat(noise, input.shape[0], axis=1)  # Repeat the noise along the channel dim to meet the dim of inputs.
        else:
            noise_shape[1] = 3
            noise = np.float32(np.random.normal(self.mu, self.sigma, noise_shape))

        new_label = input
        new_input = input + noise[0]
        return new_input, (new_label, noise)


'''
PyTorch Input transformers
'''


class TransTo3Channels(object):
    """ Transform 1-channel gray scale image to 3-channel grayscale image

    """
    def __init__(self):
        pass

    def __call__(self, image):
        assert image.shape[0] == 1
        return torch.cat([image, image, image], 0)
