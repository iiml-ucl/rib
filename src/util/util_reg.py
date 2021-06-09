"""
Some commonly used regularization functions.

Created by Zhaoyan
"""
import torch
from datetime import datetime


def select_reg(reg_type, model, reg_para):
    """

    :param reg_type: (String)
    :param model: (PyTorch Model)
    :param reg_para:
    :return:
    """
    if reg_type.lower() == 'none':
        return 0.
    elif reg_type.lower() == 'wd':
        return weight_decay(model, reg_para)
    else:
        exit(f'{datetime.now()} W util_reg.select_reg(): Unknown reg_type. '
             f'Select from None, WD. '
             f'Get {reg_type} instead.')



def weight_decay(model, beta):
    sum_weight_norm = 0.
    for name, param in model.named_parameters():
        if '.weight' in name:
            sum_weight_norm += torch.norm(param, p='fro')
    reg = sum_weight_norm * beta
    return reg
