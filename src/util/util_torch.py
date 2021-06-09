"""
Helper functions for PyTorch.

Functions:
    get_trainable_parameters
    save_model_weights

Class:
    Measurement
    PSNR
    SSIM

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from math import exp
from tqdm import tqdm
import os


def get_gpu_status():
    """
    ref: https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf
         https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/7
    :return:
    """
    # os.system('nvidia-smi -q -d  power >tmp')
    # data = open('tmp', 'r').read()
    # print(data)

    os.system('nvidia-smi -q -d  power |grep -A14 GPU|grep Max\ Power\ Limit >tmp')
    power_max = [float(x.split()[4]) for x in open('tmp', 'r').readlines()]
    os.system('nvidia-smi -q -d  power |grep -A14 GPU|grep Avg >tmp')
    power_avg = [float(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('nvidia-smi -q -d memory |grep -A4 GPU|grep Total >tmp')
    mem_tot = [float(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('nvidia-smi -q -d memory |grep -A4 GPU|grep Free >tmp')
    mem_free = [float(x.split()[2]) for x in open('tmp', 'r').readlines()]

    return (mem_tot, mem_free), (power_max, power_avg)


def auto_select_GPU(mode='memory_priority', threshold=0., unit='pct', dwell=5):
    mode_options = ['memory_priority',
                    'power_priority',
                    'memory_threshold',
                    'power_threshold']
    assert mode in mode_options, print(f'{datetime.now()} E auto_select_GPU(): Unknown model_options. Select from: '
                                       f'{mode_options}. Get {mode} instead.')

    tq = tqdm(total=dwell, desc=f'GPU Selecting... {mode}:{threshold}:{unit}', unit='dwell', dynamic_ncols=True)

    for i_dwell in range(dwell):
        (mem_tot_new, mem_free_new), (power_max_new, power_avg_new) = get_gpu_status()
        if i_dwell == 0:
            mem_tot, mem_free, power_max, power_avg = mem_tot_new, mem_free_new, power_max_new, power_avg_new
        else:
            mem_free = [min([mem_free[i], mem_free_new[i]]) for i in range(len(mem_tot_new))]
            power_avg = [max([power_avg[i], power_avg_new[i]]) for i in range(len(mem_tot_new))]
        # sleep(1)
        tq.update()
    tq.close()
    power_free = [i-j for (i, j) in zip(power_max, power_avg)]
    if unit.lower() == 'pct':
        pass
    mem_free_pct = [i/j for (i, j) in zip(mem_free, mem_tot)]
    power_free_pct = [i/j for (i, j) in zip(power_free, power_max)]

    # print(mem_free_pct)
    # print(power_free_pct)

    if mode.lower() == 'memory_priority':
        i_GPU = np.argmax(mem_free_pct)
        print(f'{datetime.now()} i Selected GPU: #{i_GPU}. (from 0 to {len(mem_free_pct)})')
        device = torch.device(f'cuda:{i_GPU}')
        return device
    else:
        return None


def get_trainable_parameters(model):
    trainable_parameters = {}
    for name, param in model.named_parameters():
        trainable_parameters[name] = param
    return trainable_parameters


def save_model_weights(model, weight_save_path):
    assert weight_save_path.endswith('.npy'), f'{datetime.now()} E The model save path must ends with .npy, get' \
                                              f' {weight_save_path} instead.'
    trainable_parameters = get_trainable_parameters(model)
    np.save(weight_save_path, trainable_parameters)


def load_model_weights(model, weight_load_path, device, para_names=None):
    """

    :param model:
    :param weight_load_path:
    :param device:
    :param para_names:
    :return:
    """
    weight_dict = np.load(weight_load_path, allow_pickle=True).item()
    model_state = model.state_dict()
    if para_names is None:  # Load the whole saved model.
        para_names = weight_dict.keys()
    for para_name in para_names:
        model_state[para_name] = weight_dict[para_name].data.to(device)
    model.load_state_dict(model_state)


def set_trainable(model, para_names, switch):
    """

    :param model:
    :param para_names: (list of String)
    :param switch: (bool) True: trainable, False, not trainable
    :return:
    """
    params = get_trainable_parameters(model)
    for para_name in para_names:
        params[para_name].requires_grad = switch


class PSNR_MSE:
    """
    Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]

    Adapted from:
        https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py

    example of usage:
        psnr_getter = PSNR()
        psnr_value, mse_value = psnr_getter(img_label, img_pred)
    """

    def __init__(self, maxi):
        """

        :param maxi: (float) Here, MAXI is the maximum possible pixel value of the image. When the pixels are
                            represented using 8 bits per sample, this is 255.
        """
        self.name = "PSNR"
        self.maxi = maxi

    def __call__(self, img1, img2):
        """

        :param img1: (torch tensor) 3-D tensor for image or 4-D tensor for batch of image. (b, c, w, h)
        :param img2: (torch tensor) 3-D tensor for image or 4-D tensor for batch of image. (b, c, w, h)
        :return:
        """
        assert img1.shape == img2.shape, print(f'{datetime.now()} E util_torch.py/PSNR_MSE.__call__(): Shape does not match:{img1.shape} and {img2.shape}')
        if not torch.is_tensor(img1):
            img1 = torch.as_tensor(img1)
        if not torch.is_tensor(img2):
            img2 = torch.as_tensor(img2)
        mse_value = torch.mean((img1 - img2) ** 2)
        psnr_value = 20 * torch.log10(self.maxi / torch.sqrt(mse_value))
        return float(psnr_value), float(mse_value)


class SSIM(torch.nn.Module):
    """

    Adapted from:
        https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

    example of usage:
        ssim_getter = SSIM()
        ssim_value = ssim_getter(img_label, img_pred)
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(self.window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        """

        :param img1: (torch tensor) 3-D tensor for image or 4-D tensor for batch of image. (b, c, w, h)
        :param img2: (torch tensor) 3-D tensor for image or 4-D tensor for batch of image. (b, c, w, h)
        :return:
        """
        if not torch.is_tensor(img1):
            img1 = torch.as_tensor(img1)
        if not torch.is_tensor(img2):
            img2 = torch.as_tensor(img2)

        if len(img1.shape) < 4:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) < 4:
            img2 = img2.unsqueeze(0)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim_value = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

        return float(ssim_value)


def select_loss(loss_name='mse'):
    if loss_name.lower() == 'mse' or loss_name.lower() == 'l2':
        # In torch, MSE loss is L2 loss
        return F.mse_loss, nn.MSELoss()
    else:
        exit(f'{datetime.now()} E Unknown loss_name. Select from mse.'
             f' Get {loss_name} instead')
