"""
For pytorch.
Helper functions to pre-process image inputs.

[Reusable]

Created by Zhaoyan @ UCL
"""

import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import cv2


def read_image(path_image, order='channel_first'):
    """
    :param path_image: path to image file.
    :param order: (Optional, String)
        'channel_first': read image in [c, h, w] order. (for PyTorch)
        'channel_last': read image in [h, w, c] order. (for TensorFlow)
    :return:
    """
    if path_image.endswith('.png') or path_image.endswith('.bmp') or path_image.endswith('.jpg'):
        image = cv2.imread(path_image)
        b, g, r = cv2.split(image)
        image = np.float32(cv2.merge([r, g, b]))
    else:
        exit(f'{datetime.now()} E Unknown image format for {path_image}.')

    if order.lower() == 'channel_last':
        pass
    elif order.lower() == 'channel_first':
        image = np.moveaxis(image, -1, 0)
    else:
        exit(f'{datetime.now()} E Unknown order argument. Select from [channel_first, channel_last], get {order} '
             f'instead.')
    return image



def build_patch_dataset(path_image_list,
                        patch_size, crop_stride, path_to_save_folder, name, order='channel_first',
                        keep_exist=True):
    """
    Save the file to a dictionary

    This function will be automatically skipped if the given dictionary is founded in the drive.
    info: [img_index, patch_counter_h, patch_counter_w, tot_patch_h, tot_patch_w, tot_patch]
    :param path_image_list:
    :param patch_size:
    :param crop_stride:
    :param path_to_save_folder: (String) Folder path
    :param name:
    :param order: (Optional, String)
        'channel_first': read image in [c, h, w] order. (for PyTorch)
        'channel_last': read image in [h, w, c] order. (for TensorFlow)
    :param keep_exist: (Optional, bool) If the dataset is already been created:
        True: do not build a new one.
        False: build a new one to replace the existing one.
    :return:
    """
    os.makedirs(path_to_save_folder, exist_ok=True)
    file_name = f'{name}_patch{patch_size}_stride{crop_stride}.npy'
    path_to_save = os.path.join(path_to_save_folder, file_name)
    if os.path.isfile(path_to_save) and keep_exist:  # Skip if exist.
        return path_to_save

    if order == 'channel_first':
        dc, dh, dw = 0, 1, 2
    elif order == 'channel_last':
        dh, dw, dc = 0, 1, 2
    else:
        exit(f'{datetime.now()} E Unknown order argument. Select from [channel_first, channel_last], get {order} '
             f'instead.')

    patch_list = []
    info_list = []
    tq = tqdm(total=len(path_image_list))
    for index, i_path in enumerate(path_image_list):
        img = read_image(i_path)
        img = np.asarray(img) / 255.
        h = img.shape[dh]
        w = img.shape[dw]
        c = img.shape[dc]
        tot_patch_h = len(range(0, h-patch_size+1, crop_stride))
        tot_patch_w = len(range(0, w-patch_size+1, crop_stride))
        tot_patch = tot_patch_h * tot_patch_w
        patch_counter_h = 0

        for ih in range(0, h-patch_size+1, crop_stride):
            patch_counter_w = 0
            for iw in range(0, w-patch_size+1, crop_stride):
                if dc == 0:
                    patch = [img[:, ih:ih+patch_size, iw:iw+patch_size]]
                elif dc == 2:
                    patch = [img[ih:ih+patch_size, iw:iw+patch_size, :]]
                info = np.asarray([[index, patch_counter_h, patch_counter_w, tot_patch_h, tot_patch_w, tot_patch]])

                patch_list.append(patch)
                info_list.append(info)
                patch_counter_w += 1
            patch_counter_h += 1
        tq.update(1)

    patch_full = np.concatenate(patch_list)
    info_full = np.concatenate(info_list)
    full_dict = {'data': patch_full, 'info': info_full}

    # Ref: https://stackoverflow.com/a/29704623
    with open(path_to_save, 'wb') as f:
        pickle.dump(full_dict, f, protocol=4)
    # np.save(path_to_save, full_dict, allow_pickle=True)
    return path_to_save


def merge_patches(data_list, info_list, patch_size, crop_stride, method='average',
                  path_to_save=None,
                  order='channel_first'):
    """

    :param data_list: List of patch value. Shape: [num_patch, h, w, c]
    :param info_list: List of info.
                      info: [img_index, patch_counter_h, patch_counter_w, tot_patch_h, tot_patch_w, tot_patch]
    :param patch_size: (int)
    :param crop_stride: (int)
    :param method: (option String). How to deal with overlay area of the images.
        'average' take the average at the
        'toast': cut the edges and merge
    :param path_to_save: (optional, String) If none, do not plot the images.
    :param order: (Optional, String)
        'channel_first': read image in [c, h, w] order. (for PyTorch)
        'channel_last': read image in [h, w, c] order. (for TensorFlow)
    :return: List of full images, in numpy array. normalized to [0., 1.]
    """
    if order == 'channel_first':
        dc, dh, dw = 0, 1, 2
    elif order == 'channel_last':
        dh, dw, dc = 0, 1, 2
    else:
        exit(f'{datetime.now()} E Unknown order argument. Select from [channel_first, channel_last], get {order} '
             f'instead.')

    idx = -1
    images = []
    if method.lower() == 'average':
        for data, info in zip(data_list, info_list):
            if dc == 0:
                data = np.moveaxis(data, 0, -1)  # Move the RGB channel to the last
            if idx != info[0]:  # New image
                idx = info[0]
                image_name = f'IMAGE{idx:06d}.png'
                base = np.zeros([crop_stride*(info[3]-1)+patch_size,
                                 crop_stride*(info[4]-1)+patch_size,
                                 data.shape[-1]])
                average_map = np.zeros([crop_stride*(info[3]-1)+patch_size,
                                        crop_stride*(info[4]-1)+patch_size,
                                        data.shape[-1]])
            # if dc == 0:
            #     base[:,
            #          info[1]*crop_stride: info[1]*crop_stride+patch_size,
            #          info[2]*crop_stride: info[2]*crop_stride+patch_size] += data
            #     average_map[:,
            #                 info[1]*crop_stride: info[1]*crop_stride+patch_size,
            #                 info[2]*crop_stride: info[2]*crop_stride+patch_size] += 1.
            # elif dc == 2:
            base[info[1]*crop_stride:info[1]*crop_stride+patch_size,
                 info[2]*crop_stride:info[2]*crop_stride+patch_size,
                 :] += data
            average_map[info[1]*crop_stride:info[1]*crop_stride+patch_size,
                        info[2]*crop_stride:info[2]*crop_stride+patch_size,
                        :] += 1.

            if info[1]+1 == info[3] and info[2]+1 == info[4]:  # End of image
                new_image = np.divide(base, average_map)
                new_image = np.clip(new_image, 0., 1.)
                images.append(new_image)
                # print(new_image)
                if path_to_save is not None:
                    os.makedirs(path_to_save, exist_ok=True)
                    # print(path_to_save)
                    # print(image_name)
                    # print(new_image)
                    new_image = new_image * 255.
                    new_image = new_image.astype(np.uint8)
                    plt.imsave(os.path.join(path_to_save, image_name), new_image)
    elif method.lower() == 'toast':
        """
        Cut the polluted edges off.
        """
        assert crop_stride % 2 == 0, f'{datetime.now()} E util_image_processing.merge_patchs(): to use toast method, ' \
                                     f'the crop stride must be an even number. Get {crop_stride} instead.'
        edge = crop_stride // 2

        for data, info in zip(data_list, info_list):
            if dc == 0:
                data = np.moveaxis(data, 0, -1)  # Move the RGB channel to the last
            if idx != info[0]:  # New image
                idx = info[0]
                image_name = f'IMAGE{idx:06d}.png'
                base = np.zeros([crop_stride*info[3],
                                 crop_stride*info[4],
                                 data.shape[-1]])
            base[info[1]*crop_stride:(info[1]+1)*crop_stride,
                 info[2]*crop_stride:(info[2]+1)*crop_stride,
                 :] += data[edge:edge+crop_stride,
                            edge:edge+crop_stride,
                            :]

            if info[1]+1 == info[3] and info[2]+1 == info[4]:  # End of image
                new_image = np.clip(base, 0., 1.)
                images.append(new_image)
                # print(new_image)
                if path_to_save is not None:
                    os.makedirs(path_to_save, exist_ok=True)
                    # print(path_to_save)
                    # print(image_name)
                    # print(new_image)
                    new_image = new_image * 255.
                    new_image = new_image.astype(np.uint8)
                    plt.imsave(os.path.join(path_to_save, image_name), new_image)

    else:
        exit(f'{datetime.now()} E Unknown method argument. Select from [average, toast], get {method} instead')
    return images
