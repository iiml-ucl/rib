"""
Utilities for record logging, path creating and so on.

Created by Zhaoyan @ UCL
"""

import csv
from datetime import datetime
import numpy as np
import os
import pandas as pd
import shutil


def get_formatted_time():
    """
    Example results: 2020-09-06_22:12:42.307021
    No space. Easier to use in Linux directory commands.
    :return:
    """
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
    return time


class Paths:
    """
    The paths to results.
    """
    def __init__(self, result_home_path,
                 record_name,
                 if_log=True,
                 if_weights=True,
                 if_summary=True,
                 if_visualization=True,
                 if_curve=True,
                 if_temp=True,
                 summary_suffix=None):
        """

        :param result_home_path: (String) Path to result home folder. The home folder is usually for a group of setups
                for the same experiment.
        :param record_name: (String) The name of current record, usually is the timestamp.
        :param if_log:
        :param if_weights:
        :param if_summary:
        :param if_visualization:
        :param if_curve:
        :param summary_suffix: (None, String, optional) The suffix to the summary record.
        :return:
        """
        self.result_home_path = result_home_path
        self.record_name = record_name
        self.dict_folder_paths = {
            'folder:log': os.path.join(result_home_path, 'log'),
            'folder:weights': os.path.join(result_home_path, 'weights'),
            'folder:summary': os.path.join(result_home_path, 'summary'),
            'folder:visualization': os.path.join(result_home_path, 'visualization'),
            'folder:curve': os.path.join(result_home_path, 'curve'),
            'folder:temp': os.path.join(result_home_path, 'temp')
        }

        self.dict_file_paths = {
            'file:log': os.path.join(self.dict_folder_paths['folder:log'],
                                     'log.csv'),
            # It is actually a folder
            'file:weights': os.path.join(self.dict_folder_paths['folder:weights'],
                                         f'{record_name}'),
            # It is actually a folder
            'file:summary': os.path.join(self.dict_folder_paths['folder:summary'],
                                         f'{record_name}'),
            # It is actually a folder
            'file:visualization': os.path.join(self.dict_folder_paths['folder:visualization'],
                                               f'{record_name}'),
            'file:curve': os.path.join(self.dict_folder_paths['folder:curve'],
                                       f'{record_name}.npy')
        }

        os.makedirs(result_home_path, exist_ok=True)

        if if_log:
            os.makedirs(self.dict_folder_paths['folder:log'], exist_ok=True)
        if if_weights:
            os.makedirs(self.dict_file_paths['file:weights'], exist_ok=True)
        if if_summary:
            os.makedirs(self.dict_folder_paths['folder:summary'], exist_ok=True)
        if if_visualization:
            os.makedirs(self.dict_file_paths['file:visualization'], exist_ok=True)
        if if_curve:
            os.makedirs(self.dict_folder_paths['folder:curve'], exist_ok=True)
        if if_temp:
            os.makedirs(self.dict_folder_paths['folder:temp'], exist_ok=True)

        self.others = {}

        if summary_suffix:
            self.add_summary_suffix(summary_suffix)

        self.log = self.dict_file_paths['file:log']
        self.weights = self.dict_file_paths['file:weights']
        self.summary = self.dict_file_paths['file:summary']
        self.summary_folder = self.dict_folder_paths['folder:summary']
        self.visualization = self.dict_file_paths['file:visualization']
        self.curve = self.dict_file_paths['file:curve']

        print(f'{datetime.now()} I Path initialization done. '
              f'\tresult_home_path: {result_home_path}'
              f'\trecord_name: {record_name}')

    def add_summary_suffix(self, summary_suffix):
        """

        :param summary_suffix: (String)
        :return:
        """
        temp_old_name = self.dict_file_paths['file:summary'].split('/')[-1]
        self.dict_file_paths['file:summary'] = \
            os.path.join(self.dict_folder_paths['folder:summary'],
                         f'{temp_old_name}:{summary_suffix}')
        self.summary = self.dict_file_paths['file:summary']

    def add_other_paths(self, name, path, makefolder=False):
        self.others[name] = path
        if makefolder:
            os.makedirs(path, exist_ok=True)

    def get_all_paths(self):
        all_paths = self.dict_folder_paths
        all_paths.update(self.dict_file_paths)
        all_paths.update(self.others)
        return all_paths


class Log:
    """
    Log the numerical results in a csv file that can be viewed by MS Excel.
    """
    def __init__(self):
        """

        :param timestamp: (String) The timestamp of the record. The timestamp can identify a test record.
        :param path: (String) The path of the .npy file>
        """
        self.log_dict_to_save = {}

    def update(self, key=None, value=None, dict=None):
        """

        :param key: (String)
        :param value: (tuple, list, or scalar for int, float, String values) Notice that the output of the torch models
                        should be wrapped with float() to convert to python scalar.
        :return:
        """
        if key:
            self.log_dict_to_save[key] = value
        else:
            self.log_dict_to_save.update(dict)

    def save(self, path_log):
        assert path_log.endswith('.csv'),\
            f'{datetime.now()} E path must be pointed at a .csv file. Get {path_log} instead.'
        for key in self.log_dict_to_save.keys():
            if not isinstance(self.log_dict_to_save[key], list):
                self.log_dict_to_save[key] = [self.log_dict_to_save[key]]
            else:
                if len(self.log_dict_to_save[key]) != 1:
                    self.log_dict_to_save[key] = [self.log_dict_to_save[key]]
        if os.path.exists(path_log):
            df = pd.read_csv(path_log)
            df_new = pd.DataFrame(self.log_dict_to_save)
            df = df.append(df_new, sort=False)
        else:
            df = pd.DataFrame(self.log_dict_to_save)
        df.to_csv(path_log, index=False)
        print(f'{datetime.now()} I Log saved:')
        for key, value in self.log_dict_to_save.items():
            print(f'\t{key}: {value}')


class Measurement:
    def __init__(self, name):
        self.name = name

        self.last_value = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.average = 0.0

        self.tape = {}
        self.tape_index = 0

    def clear(self):
        self.last_value = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.average = 0.0

    def update(self, value):
        self.last_value = value
        self.sum += value
        self.count += 1
        self.average = self.sum / self.count

    def write_tape(self, plot_index=0):
        """
        One could use the log_results function in util_log.py to save the tapes in csv files.
        :param plot_index: (int, optional) The index for plotting. e.g. epoch index.
        :return:
        """
        self.tape[self.tape_index] = {'name': self.name,
                                      'plot_index': plot_index,
                                      'sum': self.sum,
                                      'count': self.count,
                                      'average': self.average,
                                      'last_value': self.last_value}
        self.tape_index += 1

    def save_tape(self, path_tape):
        """

        :param path_tape:
        :return:
        """
        assert path_tape.endswith('npy')
        np.save(path_tape, self.tape)
