import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

path_result_home_folder = '/scratch/uceelyu/pycharm_sync_RIB/results/derain'

timestamp_keep = ['2021-05-29_02:29:46.726358',
                  '2021-05-29_03:20:32.546199',
                  '2021-05-29_04:08:03.415129',
                  '2021-05-29_04:52:00.804705',
                  '2021-05-29_05:49:18.704900',

                  '2021-05-29_02:37:24.999419',
                  '2021-05-29_04:25:03.006719',
                  '2021-05-29_05:30:44.685682',
                  '2021-05-29_06:36:13.296192',

                  '2021-05-29_19:17:32.821296',
                  '2021-05-29_19:54:24.698361',
                  '2021-05-29_20:31:07.948853',
                  '2021-05-29_21:35:40.380751',
                  '2021-05-29_22:40:00.458219',

                  '2021-05-29_19:16:53.925233',
                  '2021-05-29_19:53:19.893751',
                  '2021-05-29_21:34:51.681418',
                  '2021-05-29_22:39:12.713445' ]


def if_list_item_in_string(list_item, string):
    for i_item in list_item:
        if i_item in string:
            return True
    return False


def remove_item(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f'{datetime.now()} I Remove path: {path}')
    elif os.path.isfile(path):
        os.remove(path)
        print(f'{datetime.now()} I Remove file: {path}')


def remove_records(list_keyword_keep, path_result_home):
    folders = ['curve', 'summary', 'visualization', 'weights']
    level1 = [os.path.join(path_result_home, i) for i in folders]
    to_keep = {i: [] for i in folders}
    to_remove = {i: [] for i in folders}

    for i_folder, i_level1 in zip(folders, level1):
        level2 = os.listdir(i_level1)
        level2 = [os.path.join(i_level1, i) for i in level2]
        for i_level2 in level2:
            if if_list_item_in_string(list_keyword_keep, i_level2):
                to_keep[i_folder].append(i_level2)
            else:
                to_remove[i_folder].append(i_level2)

    print('!!!Attention!!!\n')
    for i_folder in folders:
        print(f'In [{i_folder}], {len(to_keep[i_folder])} will be kept. {len(to_remove[i_folder])} will be removed.')
        print(f'kept: ')
        for i_path in to_keep[i_folder]:
            print(i_path)
        print(f'removed: ')
        for i_path in to_remove[i_folder]:
            print(i_path)
        print('')
    select = input('Are you sure? ')
    if select.lower() == 'y' or select.lower() == 'yes':
        for _, value_list in to_remove.items():
            for i_path in value_list:
                remove_item(i_path)
            print(f'{len(value_list)} file/folder removed.')
    print('\n')


if __name__ == '__main__':
    remove_records(timestamp_keep, path_result_home_folder)


