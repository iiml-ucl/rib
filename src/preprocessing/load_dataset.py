"""
Created by Zhaoyan @ UCL
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import pathlib
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import PurePath
import pickle
import sys
sys.path.append(os.path.join(os.getcwd(), PurePath('../../..')))
import src.preprocessing.processors as P
import src.preprocessing.util_image_preprocessing as util


def read_rain100_paths():
    path_raw_folder = '/scratch/uceelyu/Datasets/dataset_Rain100/raw'

    path_practical_folder = os.path.join(path_raw_folder, 'Practical')

    path_rain100h_train_rainy_folder = os.path.join(path_raw_folder, PurePath('Rain100H/train/rain/X2'))
    path_rain100h_train_label_folder = os.path.join(path_raw_folder, PurePath('Rain100H/train/norain'))
    path_rain100h_test_rainy_folder = os.path.join(path_raw_folder, PurePath('Rain100H/test/rain/X2'))
    path_rain100h_test_label_folder = os.path.join(path_raw_folder, PurePath('Rain100H/test/norain'))

    path_rain100l_train_rainy_folder = os.path.join(path_raw_folder, PurePath('Rain100L/train/rain'))
    path_rain100l_train_label_folder = os.path.join(path_raw_folder, PurePath('Rain100L/train/norain'))
    path_rain100l_test_rainy_folder = os.path.join(path_raw_folder, PurePath('Rain100L/test/rain/X2'))
    path_rain100l_test_label_folder = os.path.join(path_raw_folder, PurePath('Rain100L/test/norain'))

    paths_practical = sorted([os.path.join(path_practical_folder, i)
                              for i in os.listdir(path_practical_folder)])

    paths_rain100h_train_rainy = sorted([os.path.join(path_rain100h_train_rainy_folder, i)
                                  for i in os.listdir(path_rain100h_train_rainy_folder)])
    paths_rain100h_train_label = sorted([os.path.join(path_rain100h_train_label_folder, i)
                                  for i in os.listdir(path_rain100h_train_label_folder)])
    paths_rain100h_test_rainy = sorted([os.path.join(path_rain100h_test_rainy_folder, i)
                                 for i in os.listdir(path_rain100h_test_rainy_folder)])
    paths_rain100h_test_label = sorted([os.path.join(path_rain100h_test_label_folder, i)
                                 for i in os.listdir(path_rain100h_test_label_folder)])

    paths_rain100l_train_rainy = sorted([os.path.join(path_rain100l_train_rainy_folder, i)
                                  for i in os.listdir(path_rain100l_train_rainy_folder)])
    paths_rain100l_train_label = sorted([os.path.join(path_rain100l_train_label_folder, i)
                                  for i in os.listdir(path_rain100l_train_label_folder)])
    paths_rain100l_test_rainy = sorted([os.path.join(path_rain100l_test_rainy_folder, i)
                                 for i in os.listdir(path_rain100l_test_rainy_folder)])
    paths_rain100l_test_label = sorted([os.path.join(path_rain100l_test_label_folder, i)
                                 for i in os.listdir(path_rain100l_test_label_folder)])

    len_rain100h_test = len(paths_rain100h_test_rainy)
    len_rain100l_test = len(paths_rain100l_test_rainy)

    i_split_rain100h = int(0.5*len_rain100h_test)
    i_split_rain100l = int(0.5*len_rain100l_test)

    return {'rain100:practical': paths_practical,

            'rain100h:train_rainy': paths_rain100h_train_rainy,
            'rain100h:train_label': paths_rain100h_train_label,
            'rain100h:val_rainy': paths_rain100h_test_rainy[:i_split_rain100h],
            'rain100h:val_label': paths_rain100h_test_label[:i_split_rain100h],
            'rain100h:test_rainy': paths_rain100h_test_rainy[i_split_rain100h:],
            'rain100h:test_label': paths_rain100h_test_label[i_split_rain100h:],

            'rain100l:train_rainy': paths_rain100l_train_rainy,
            'rain100l:train_label': paths_rain100l_train_label,
            'rain100l:val_rainy': paths_rain100l_test_rainy[:i_split_rain100l],
            'rain100l:val_label': paths_rain100l_test_label[:i_split_rain100l],
            'rain100l:test_rainy': paths_rain100l_test_rainy[i_split_rain100l:],
            'rain100l:test_label': paths_rain100l_test_label[i_split_rain100l:]
            }


def read_rain12_paths():
    path_raw_folder = '/scratch/uceelyu/Datasets/dataset_Rain12/raw'
    paths_all = os.listdir(path_raw_folder)
    paths_groundtruth = sorted([i for i in paths_all if '_GT' in i])
    paths_rainy = sorted([i for i in paths_all if '_in' in i])
    paths_residual = sorted([i for i in paths_all if '_R' in i])
    paths_groundtruth = [os.path.join(path_raw_folder, i) for i in paths_groundtruth]
    paths_rainy = [os.path.join(path_raw_folder, i) for i in paths_rainy]
    paths_residual = [os.path.join(path_raw_folder, i) for i in paths_residual]
    tot_len = len(paths_groundtruth)
    i_split_1 = int(0.5*tot_len)
    i_split_2 = i_split_1 + int(0.25*tot_len)
    return {
        'rain12:train_label': paths_groundtruth[:i_split_1],
        'rain12:train_rainy': paths_rainy[:i_split_1],
        'rain12:train_noise': paths_residual[:i_split_1],

        'rain12:val_label': paths_groundtruth[i_split_1:i_split_2],
        'rain12:val_rainy': paths_rainy[i_split_1:i_split_2],
        'rain12:val_noise': paths_residual[i_split_1:i_split_2],

        'rain12:test_label': paths_groundtruth[i_split_2:],
        'rain12:test_rainy': paths_rainy[i_split_2:],
        'rain12:test_noise': paths_residual[i_split_2:],
    }


def torch_dataset_download_helper():
    """Call this function if you want to download dataset via PyTorch API"""
    from six.moves import urllib

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]

    urllib.request.install_opener(opener)


def load_mnist_denoising(path_raw_dataset, batch_size=1, mu=0., sigma=0.6, deterministic=True):
    """
    1. Get the MNIST dataset via PyTorch built-in APIs.
    2. Wrap it with customized wrapper with additive Gaussian noise processor
    3. Build PyTorch data loader objects.

    :param path_raw_dataset:
    :param batch_size:
    :param mu:
    :param sigma:
    :param deterministic:
    :return: dict of pytorch DataLoader objects.
        {
            'train':
                (iterable) [noisy_image, (clean_image, noise)]
                    noisy_image shape: [batch, c, w, h]
                    clean_image shape: [batch, c, w, h]
                    noise shape: [batch, 1, c, w, h]
            'val':
                (iterable) [noisy_image, (clean_image, noise)]
                    noisy_image shape: [batch, c, w, h]
                    clean_image shape: [batch, c, w, h]
                    noise shape: [batch, 1, c, w, h]
        }
    """
    MNIST = P.data_processor_wrapper(torchvision.datasets.MNIST,
                                     P.Processor_Denoising_AddGau(mu, sigma, deterministic, grayscale=True))

    transform_input = transforms.Compose([
        transforms.ToTensor(),
        P.TransTo3Channels()
    ])
    try:
        data_train = MNIST(root=path_raw_dataset, train=True, download=False,
                           transform=transform_input)
    except:
        torch_dataset_download_helper()
        data_train = MNIST(root=path_raw_dataset, train=True, download=True,
                           transform=transform_input)
    try:
        data_val = MNIST(root=path_raw_dataset, train=False, download=False,
                         transform=transform_input)
    except:
        torch_dataset_download_helper()
        data_val = MNIST(root=path_raw_dataset, train=False, download=True,
                         transform=transform_input)

    datasets = {'train': data_train, 'val': data_val}
    data_loaders = {i: torch.utils.data.DataLoader(datasets[i], batch_size=batch_size, shuffle=False)
                    for i in ['train', 'val']}
    return data_loaders


def load_derain(paths_data_dict, train_size=-1, batch_size=1):
    """
    1. Load data dictionaries.
    2. Wrap it to a PyTorch Dataset Class.
    3. Build PyTorch data loader objects.

    :param paths_data_dict: (list of String) The path to pre-processed (cropped) and saved patch dictionaries.
                            Should be obtained by util_image_preprocessing.build_patch_dataset() function.
    :param train_size: (optional, int) The number of patches that used for training. if -1, use all patches.
    :param batch_size: (optional, int) the batch size. For visualization, batch size should be 1 to make sure no patch
                        is left.
    :return: (dict of torch Dataloader objects)
        {
            'rain100h:train':   iterable: [sample_rainy] [sample_label, sample_info]
            'rain100h:val':     iterable: [sample_rainy] [sample_label, sample_info]
            'rain100l:train':   iterable: [sample_rainy] [sample_label, sample_info]
            'rain100l:val':     iterable: [sample_rainy] [sample_label, sample_info]
            'rain12:train':     iterable: [sample_rainy] [sample_label, sample_noise, sample_info]
            'rain100:practical':iterable: [sample_rainy] []
        }
    """
    def _find_item_by_string(list_of_item, string):
        for i in list_of_item:
            if string in i:
                return i
        print(f'{datetime.now()} W Dataset not found with searching key: {string}')
        return None
    # Rain100H, train
    path_1 = _find_item_by_string(paths_data_dict, 'rain100h:train_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain100h:train_label')
    if path_1 and path_2:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict1['info']]
        data_rain100h_train = P.CustomDataset(inputs, labels, train_size)
    else:
        data_rain100h_train = None

    # Rain100L, train
    path_1 = _find_item_by_string(paths_data_dict, 'rain100l:train_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain100l:train_label')
    if path_1 and path_2:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict1['info']]
        data_rain100l_train = P.CustomDataset(inputs, labels, train_size)
    else:
        data_rain100l_train = None

    # Rain100H, test
    path_1 = _find_item_by_string(paths_data_dict, 'rain100h:test_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain100h:test_label')
    if path_1 and path_2:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict1['info']]
        data_rain100h_test = P.CustomDataset(inputs, labels)
    else:
        data_rain100h_test = None

    # Rain100L, test
    path_1 = _find_item_by_string(paths_data_dict, 'rain100l:test_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain100l:test_label')
    if path_1 and path_2:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict1['info']]
        data_rain100l_test = P.CustomDataset(inputs, labels)
    else:
        data_rain100l_test = None

    # Rain100H, val
    path_1 = _find_item_by_string(paths_data_dict, 'rain100h:val_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain100h:val_label')
    if path_1 and path_2:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict1['info']]
        data_rain100h_val = P.CustomDataset(inputs, labels)
    else:
        data_rain100h_val = None

    # Rain100L, val
    path_1 = _find_item_by_string(paths_data_dict, 'rain100l:val_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain100l:val_label')
    if path_1 and path_2:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict1['info']]
        data_rain100l_val = P.CustomDataset(inputs, labels)
    else:
        data_rain100l_val = None

    # Rain12, train
    path_1 = _find_item_by_string(paths_data_dict, 'rain12:train_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain12:train_label')
    path_3 = _find_item_by_string(paths_data_dict, 'rain12:train_noise')
    if path_1 and path_2 and path_3:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        dict3 = np.load(path_3, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict3['data'], dict1['info']]
        data_rain12_train = P.CustomDataset(inputs, labels, train_size)
    else:
        data_rain12_train = None

    # Rain12, val
    path_1 = _find_item_by_string(paths_data_dict, 'rain12:val_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain12:val_label')
    path_3 = _find_item_by_string(paths_data_dict, 'rain12:val_noise')
    if path_1 and path_2 and path_3:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        dict3 = np.load(path_3, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict3['data'], dict1['info']]
        data_rain12_val = P.CustomDataset(inputs, labels, train_size)
    else:
        data_rain12_val = None

    # Rain12, test
    path_1 = _find_item_by_string(paths_data_dict, 'rain12:test_rainy')
    path_2 = _find_item_by_string(paths_data_dict, 'rain12:test_label')
    path_3 = _find_item_by_string(paths_data_dict, 'rain12:test_noise')
    if path_1 and path_2 and path_3:
        dict1 = np.load(path_1, allow_pickle=True)
        dict2 = np.load(path_2, allow_pickle=True)
        dict3 = np.load(path_3, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict2['data'], dict3['data'], dict1['info']]
        data_rain12_test = P.CustomDataset(inputs, labels, train_size)
    else:
        data_rain12_test = None

    # Rain100, practical
    path_1 = _find_item_by_string(paths_data_dict, 'rain100:practical')
    if path_1:
        dict1 = np.load(path_1, allow_pickle=True)
        inputs = [dict1['data']]
        labels = [dict1['info']]
        data_rain100_practical = P.CustomDataset(inputs, labels)
    else:
        data_rain100_practical = None

    datasets = {
        'rain100h:train': data_rain100h_train,
        'rain100h:val': data_rain100h_val,
        'rain100h:test': data_rain100h_test,

        'rain100l:train': data_rain100l_train,
        'rain100l:val': data_rain100l_val,
        'rain100l:test': data_rain100l_test,

        'rain12:train': data_rain12_train,
        'rain12:val': data_rain12_val,
        'rain12:test': data_rain12_test,

        'rain100:practical': data_rain100_practical
    }

    dataloaders = {}
    for key in datasets.keys():
        if ':test' in key or ':practical' in key:
            batch_size_real = 1
        else:
            batch_size_real = batch_size
        # print(key)
        # print(batch_size_real)
        if datasets[key] is not None:
            dataloaders[key] = torch.utils.data.DataLoader(datasets[key], batch_size=batch_size_real, shuffle=False)

    print(f'{datetime.now()} I Dataloaders built. Keys: {dataloaders.keys()}')

    return dataloaders


def recipes(task_name, path_raw_dataset='', batch_size=1, **kwargs):
    """

    :param task_name: (String) the name of the task. Select from:
                        'mnist_denoising'
                        'derain'
    :param path_raw_dataset: (optional, String)
    :param batch_size: (optional, int) batch size.
    :param kwargs: The kwargs for selected task.
    :return:
        'mnist_denoising'
            :key: 'train', 'val'
            iterable: input, (label, noise)
        'derain'
            :key: '

    """
    if task_name == 'mnist_denoising':
        dataloaders = load_mnist_denoising(path_raw_dataset, batch_size, **kwargs)

    elif task_name == 'derain100h':
        """
        return:
        {
            'rain100h:train':   iterable: [sample_rainy] [sample_label, sample_info]
            'rain100h:val':     iterable: [sample_rainy] [sample_label, sample_info]
            'rain100h:test':    iterable: [sample_rainy] [sample_label, sample_info]
            'rain100:practical':iterable: [sample_rainy] [sample_info]
        }
        """
        paths_dict_rain12 = read_rain12_paths()
        paths_dict_rain100 = read_rain100_paths()
        paths_data_dict = []
        for i_dict in [paths_dict_rain12, paths_dict_rain100]:
            for key, item in i_dict.items():
                if 'rain100h' in key or 'practical' in key:
                    path_data_dict = util.build_patch_dataset(item, 128, 64,
                                                              '/scratch/uceelyu/pycharm_sync_RIB/dataset/Rain',
                                                              key,
                                                              order='channel_first')
                    paths_data_dict.append(path_data_dict)
        dataloaders = load_derain(paths_data_dict, kwargs['train_size'], batch_size)

    elif task_name == 'derain100l':
        """
        return:
        {
            'rain100l:train':   iterable: [sample_rainy] [sample_label, sample_info]
            'rain100l:val':     iterable: [sample_rainy] [sample_label, sample_info]
            'rain100l:test':    iterable: [sample_rainy] [sample_label, sample_info]
            'rain100:practical':iterable: [sample_rainy] [sample_info]
        }
        """
        paths_dict_rain12 = read_rain12_paths()
        paths_dict_rain100 = read_rain100_paths()
        paths_data_dict = []
        for i_dict in [paths_dict_rain12, paths_dict_rain100]:
            for key, item in i_dict.items():
                if 'rain100l' in key or 'practical' in key:
                    path_data_dict = util.build_patch_dataset(item, 128, 64,
                                                              '/scratch/uceelyu/pycharm_sync_RIB/dataset/Rain',
                                                              key,
                                                              order='channel_first')
                    paths_data_dict.append(path_data_dict)
        dataloaders = load_derain(paths_data_dict, kwargs['train_size'], batch_size)

    elif task_name == 'derain12':
        """
        return:
        {
            'rain12:train':     iterable: [sample_rainy] [sample_label, sample_noise, sample_info]
            'rain12:val':       iterable: [sample_rainy] [sample_label, sample_noise, sample_info]
            'rain12:test':      iterable: [sample_rainy] [sample_label, sample_noise, sample_info]
            'rain100:practical':iterable: [sample_rainy] [sample_info]
        }
        """
        paths_dict_rain12 = read_rain12_paths()
        paths_dict_rain100 = read_rain100_paths()
        paths_data_dict = []
        for i_dict in [paths_dict_rain12, paths_dict_rain100]:
            for key, item in i_dict.items():
                if 'rain12' in key or 'practical' in key:
                    path_data_dict = util.build_patch_dataset(item, 128, 64,
                                                              '/scratch/uceelyu/pycharm_sync_RIB/dataset/Rain',
                                                              key,
                                                              order='channel_first')
                    paths_data_dict.append(path_data_dict)
        dataloaders = load_derain(paths_data_dict, kwargs['train_size'], batch_size)

    return dataloaders


if __name__ == '__main__':

    # data_loaders = recipes('mnist_denoising', '/scratch/uceelyu/pycharm_sync_RIB/dataset/MNIST/raw')
    # print(data_loaders)

    data_loaders = recipes('derain100h', train_size=-1)
    # data_loaders = recipes('derain100l')
    # data_loaders = recipes('derain12')
    print(data_loaders)
    for i_sample in data_loaders['rain100h:train']:
        print(i_sample)
        break
    for i_sample in data_loaders['rain100h:val']:
        print(i_sample)
        break
    for i_sample in data_loaders['rain100h:test']:
        print(i_sample)
        break
    print('\n\n\n')
    for i_sample in data_loaders['rain100:practical']:
        print(i_sample)
        break
