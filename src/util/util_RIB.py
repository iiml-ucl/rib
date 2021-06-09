"""
Utilities for RIB training. To make the experiment setup looks cleaner.

Created by Zhaoyan @ UCL
"""
import distutils.log

import numpy as np
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../..'))
import src.util.util_log as util_log
import src.util.util_torch as util_torch
from src.preprocessing.load_dataset import recipes
from src.preprocessing.util_image_preprocessing import merge_patches
import src.models as models


def train_step(batch_input, batch_labels, batch_noises,
               model_RIB, model_RIB_loss, beta, optimizer, reg_term):
    y, z, = model_RIB(batch_input)
    rib_loss, mini = model_RIB_loss(y, z, batch_labels, batch_noises, beta)
    rib_loss = rib_loss + reg_term
    rib_loss.backward()
    optimizer.step()
    return y, mini


def test_step(batch_input, model_RIB):
    y, _ = model_RIB(batch_input)
    return y


def evaluate(dataloader, name, model, measurements, paths, device, psnr_mse_getter, ssim_getter, patch_size, crop_stride):
    path_save_folder = os.path.join(paths.visualization, name)
    os.makedirs(path_save_folder, exist_ok=True)
    list_input = []
    list_label = []
    list_pred = []
    list_info = []
    tq = tqdm(total=len(dataloader), unit='patch', dynamic_ncols=True, desc=f'eva:{name}')
    for step, samples in enumerate(dataloader):
        batch_input = [i.to(device) for i in samples[0]]
        if len(samples[1]) > 1:
            batch_label = samples[1][0]
            # batch_label = torch.clamp(batch_label * 255, 0., 255).to(torch.uint8)
        else:
            batch_label = None
        batch_info = samples[1][-1]
        batch_pred, _ = model(batch_input)
        # batch_input = torch.clamp(batch_input[0] * 255, 0., 255).to(torch.uint8)
        # batch_pred = torch.clamp(batch_pred * 255, 0., 255).to(torch.uint8)

        list_input.append(batch_input[0][0].cpu().detach().numpy())
        list_pred.append(batch_pred[0].cpu().detach().numpy())
        if batch_label is not None:
            list_label.append(batch_label[0].cpu().detach().numpy())
        list_info.append(batch_info[0].cpu().detach().numpy())
        tq.update(1)
    tq.close()
    # Merge patches into images.
    images_input = merge_patches(list_input, list_info, patch_size, crop_stride, method='toast')
    images_pred = merge_patches(list_pred, list_info, patch_size, crop_stride, method='toast')
    if list_label:
        images_label = merge_patches(list_label, list_info, patch_size, crop_stride, method='toast')
    else:
        images_label = [None] * len(images_input)
    tq = tqdm(total=len(images_input), unit='image', dynamic_ncols=True, desc=f'save:{name}')
    for index, [image_input, image_pred, image_label] in enumerate(zip(images_input,
                                                                       images_pred,
                                                                       images_label)):
        if image_label is not None:
            psnr_value, mse_value = psnr_mse_getter(image_pred, image_label)
            ssim_value = ssim_getter(image_pred, image_label)
            mse_value = float(mse_value)
            psnr_value = float(psnr_value)
            ssim_value = float(ssim_value)
            measurements['mse'].update(mse_value)
            measurements['psnr'].update(psnr_value)
            measurements['ssim'].update(ssim_value)
            img_name_label = f'label_IMG{index:06d}.png'
            img_name_pred = f'pred_IMG{index:06d}' \
                            f'_m{mse_value}' \
                            f'_p{psnr_value}' \
                            f'_s{ssim_value}.png'
            # image_label = np.moveaxis(image_label, 0, -1)
            image_save = image_label * 255.
            image_save = image_save.astype(np.uint8)
            image_save = Image.fromarray(image_save)
            image_save.save(os.path.join(path_save_folder, img_name_label))
        else:
            img_name_pred = f'pred_IMG{index:06d}.png'
        img_name_input = f'input_IMG{index:06d}.png'
        # image_input = np.moveaxis(image_input, 0, -1)
        image_save = image_input * 255.
        image_save = image_save.astype(np.uint8)
        image_save = Image.fromarray(image_save)
        image_save.save(os.path.join(path_save_folder, img_name_input))
        # image_pred = np.moveaxis(image_pred, 0, -1)
        image_save = image_pred * 255.
        image_save = image_save.astype(np.uint8)
        image_save = Image.fromarray(image_save)
        image_save.save(os.path.join(path_save_folder, img_name_pred))
        tq.update(1)
    tq.close()


class TrainSetup(ABC):
    def __init__(self, result_home_path, **kwargs_names):
        self.result_home_path = result_home_path
        self.timestamp = kwargs_names['timestamp']
        self.task_name = kwargs_names['task_name']
        self.encoder_name = kwargs_names['encoder_name']
        self.decoder_name = kwargs_names['decoder_name']


class TrainSetupDerain(TrainSetup):
    def __init__(self, result_home_path,
                 logger,
                 tot_epochs=10,
                 batch_size=1,
                 train_size=-1,
                 learning_rate=1e-3,
                 beta=0.,
                 minimality='None',
                 loss_name='mse',
                 reg_type='None',
                 reg_para=None,
                 encoder_setup=None,
                 decoder_setup=None,
                 log_epochs=1,
                 **kwargs):
        super().__init__(result_home_path, **kwargs)
        logger.update('timestamp', self.timestamp)
        logger.update('task_name', self.task_name)
        logger.update('encoder_name', self.encoder_name)
        logger.update('decoder_name', self.decoder_name)
        self.tot_epochs = tot_epochs
        self.batch_size = batch_size
        self.train_size = train_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.minimality = minimality
        self.loss_name = loss_name
        self.reg_type = reg_type
        self.reg_para = reg_para
        self.log_epochs = log_epochs
        logger.update('tot_epochs', self.tot_epochs)
        logger.update('batch_size', self.batch_size)
        logger.update('train_size', self.train_size)
        logger.update('learning_rate', self.learning_rate)
        logger.update('beta', self.beta)
        logger.update('minimality', self.minimality)
        logger.update('loss_name', self.loss_name)
        logger.update('reg_type', self.reg_type)
        logger.update('reg_para', self.reg_para)
        logger.update('log_epochs', self.log_epochs)
        self.encoder_class = self.get_model_class(self.encoder_name)
        self.decoder_class = self.get_model_class(self.decoder_name)
        self.encoder_model_setup, encoder_setup_info = self.get_model_setup(self.encoder_name, encoder_setup)
        self.decoder_model_setup, decoder_setup_info = self.get_model_setup(self.decoder_name, decoder_setup)
        if encoder_setup_info:
            logger.update(dict=encoder_setup_info)
        if decoder_setup_info:
            logger.update(dict=decoder_setup_info)

    @staticmethod
    def get_model_class(model_name):
        # TODO: add more class options
        if model_name == 'FamilyCNN':
            return models.FamilyCNN
        elif model_name == 'DerainDestoringNet_en':
            return models.DerainDestoringNet_en
        elif model_name == 'DerainDestoringNet_de':
            return models.DerainDestoringNet_de
        else:
            exit(f'{datetime.now()} E util_RIB.TrainSetupDerain(): Unknown model_name. '
                 f'Select from CNN, DerainDestoringNet_en, DerainDestoringNet_de. '
                 f'Get {model_name} instead.')

    @staticmethod
    def get_model_setup(model_name, model_setup):
        # TODO: add more class options
        model_setup_object = None
        model_setup_info = None
        if model_name == 'FamilyCNN':
            model_setup_object = models.FamilyCNN_setup(model_setup)
            model_setup_info = model_setup_object.info
            return model_setup_object, model_setup_info
        elif 'Derain' in model_name:
            pass
        else:
            exit(f'{datetime.now()} E util_RIB.TrainSetupDerain(): Unknown model_name. '
                 f'Select from CNN, DerainDestoringNet_en, DerainDestoringNet_de. '
                 f'Get {model_name} instead.')

        return model_setup_object, model_setup_info


def select_dataset_components(task_name, dataloaders, minimality):
    # dataloaders = recipes(task_name, train_size=train_size, batch_size=batch_size)
    dataloader_train = None
    dataloader_val = None
    dataloader_test = None
    dataloader_practical = None
    for key in dataloaders.keys():
        if ':train' in key:
            dataloader_train = dataloaders[key]
        if ':val' in key:
            dataloader_val = dataloaders[key]
        if ':test' in key:
            dataloader_test = dataloaders[key]
        if ':practical' in key:
            dataloader_practical = dataloaders[key]

    if minimality.lower() == 'none':
        input_idx = [[0, 0]]
        label_idx = [[1, 0]]
        noise_idx = []
    elif minimality.lower() == 'noisy_input':
        input_idx = [[0, 0]]
        label_idx = [[1, 0]]
        noise_idx = [[0, 0]]
    elif minimality.lower() == 'noise':
        assert 'rain12' in task_name, \
            f'{datetime.now()} E Select recipe ({task_name}) does not have noisy input available: {task_name}'
        input_idx = [[0, 0]]
        label_idx = [[1, 0]]
        noise_idx = [[1, 1]]
    elif minimality.lower() == 'noisy_input:noise':
        assert 'rain12' in task_name, \
            f'{datetime.now()} E Select recipe ({task_name}) does not have noisy input available: {task_name}'
        input_idx = [[0, 0]]
        label_idx = [[1, 0]]
        noise_idx = [[0, 0], [1, 1]]
    else:
        exit(f'{datetime.now()} E Unknown minimality selection. Select from: None, noisy_input, noise,'
             f' noisy_input:noise. Get{minimality} instead.')

    return dataloader_train, dataloader_val, dataloader_test, dataloader_practical, input_idx, label_idx, noise_idx
