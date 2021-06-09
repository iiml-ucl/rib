"""
The model zoo.

Created by Zhaoyan @ UCL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class MLP_setup():
    """
    Set-up class for MLP models.
    """
    def __init__(self, name, dim_in, dim_out, n_hid_layers, dim_hid):
        self.name = name
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_hid_layers = n_hid_layers
        self.dim_hid = dim_hid

    def add_suffix(self, suffix):
        """
        Add a suffix to the name to identify the models.

        :param suffix: (String)
        :return:
        """
        self.name = f'{self.name}_{suffix}'


class FamilyMLP(nn.Module):
    def __init__(self, setup):
        """

        :param setup: (MLP_setup object)
        """
        super(FamilyMLP, self).__init__()
        self.name = setup.name
        if setup.n_hid_layers == 1:
            layers = OrderedDict([(f'{setup.name}:L01_linear', nn.Linear(setup.dim_in, setup.dim_out))])
        else:
            layers = OrderedDict([(f'{setup.name}:L01_linear', nn.Linear(setup.dim_in, setup.dim_hid)),
                                  (f'{setup.name}:L01_act', nn.ReLU())])
            for i_layer in range(2, setup.n_hid_layers):
                layers.update({f'{setup.name}:L{i_layer:02}_linear': nn.Linear(setup.dim_hid, setup.dim_hid)})
                layers.update({f'{setup.name}:L{i_layer:02}_act': nn.ReLU()})
            layers.update({f'{setup.name}:L{setup.n_hid_layers:02}_linear': nn.Linear(setup.dim_hid, setup.dim_out)})
        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)


class FamilyCNN_setup():
    """
    Set-up  class for CNN models.
    """
    def __init__(self, model_setup):
        self.name = model_setup['model_name']
        self.channel_in = model_setup['channel_in']
        self.channel_out = model_setup['channel_out']
        self.n_hid_layers = model_setup['n_hid_layers']
        self.channel_hid = model_setup['channel_hid']
        self.filter_size = model_setup['filter_size']
        self.padding = model_setup['padding']
        self.stride = model_setup['stride']

        self.info = {
            'model_name': self.name,
            f'{self.name}:channel_in': self.channel_in,
            f'{self.name}:channel_out': self.channel_out,
            f'{self.name}:n_hid_layers': self.n_hid_layers,
            f'{self.name}:channel_hid': self.channel_hid,
            f'{self.name}:filter_size': self.filter_size,
            f'{self.name}:padding': self.padding,
            f'{self.name}:stride': self.stride
        }

    def add_suffix(self, suffix):
        """
        Add a suffix to the name to identify the models.

        :param suffix: (String)
        :return:
        """
        self.name = f'{self.name}_{suffix}'


class FamilyCNN(nn.Module):
    def __init__(self, setup):
        """

        :param setup: (CNN_setup object)
        """
        super(FamilyCNN, self).__init__()
        self.name = setup.name
        if setup.n_hid_layers == 1:
            layers = OrderedDict([(f'{setup.name}:L01_conv', nn.Conv2d(setup.channel_in,
                                                                       setup.channel_out,
                                                                       setup.filter_size,
                                                                       setup.stride,
                                                                       setup.padding))])
        else:
            layers = OrderedDict([(f'{setup.name}:L01_conv', nn.Conv2d(setup.channel_in,
                                                                       setup.channel_hid,
                                                                       setup.filter_size,
                                                                       setup.stride,
                                                                       setup.padding)),
                                  (f'{setup.name}:L01_act', nn.ReLU())])
            for i_layer in range(2, setup.n_hid_layers):
                layers.update({f'{setup.name}:L{i_layer:02}_conv': nn.Conv2d(setup.channel_hid,
                                                                             setup.channel_hid,
                                                                             setup.filter_size,
                                                                             setup.stride,
                                                                             setup.padding)})
                layers.update({f'{setup.name}:L{i_layer:02}_act': nn.ReLU()})
            layers.update({f'{setup.name}:L{setup.n_hid_layers:02}_conv': nn.Conv2d(setup.channel_hid,
                                                                                    setup.channel_out,
                                                                                    setup.filter_size,
                                                                                    setup.stride,
                                                                                    setup.padding)})
        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)


def CNN_margin_test(model_setup, ModelClass, input_image_shape):
    """
    TODO
    CNN layers with or without padding will pollute the margins of the image. It depends on the size of convolution
    kernel, padding setup and so on. Meanwhile, in regression tasks, we usually crop full-size images into patches, and
    splicing the patches into full images in the end. Therefore, if the margins of the patches are polluted, the quality
    of the spliced image will be poor.

    To deal with this problem, we could clip the patch edges that been polluted and keep the good pixels in the center.
    In doing so, we need to know how many pixels on the edges are polluted.

    With this function, we feed the network model that we want know how many edge pixels are polluted with an
    synthetic image. We could observe the edges to know how many pixels are polluted and decide how to crop and splice
    patches in practice.

    :param model_setup:
    :param ModelClass:
    :param input_image_shape:
    :return:
    """
    model = ModelClass(model_setup)


class DerainDestoringNet_en(nn.Module):
    """
    The encoder part of DestoringNet.
    ref: @inproceedings{eigen2013restoring,
          title={Restoring an image taken through a window covered with dirt or rain},
          author={Eigen, David and Krishnan, Dilip and Fergus, Rob},
          booktitle={Proceedings of the IEEE international conference on computer vision},
          pages={633--640},
          year={2013}
        }

    """
    def __init__(self):
        super().__init__()
        self.layer_1_padding = nn.ReflectionPad2d((7, 8, 7, 8))
        self.layer_1_conv = nn.Conv2d(3, 512, 16, 1, bias=True)
        self.layer_2_conv = nn.Conv2d(512, 512, 1, bias=True)
        # self.layer_3_padding = nn.ReflectionPad2d((3, 4, 3, 4))
        # self.layer_3_conv = nn.Conv2d(512, 3, 8, bias=True)

    def forward(self, x):
        self.t1 = self.layer_1_padding(x)
        self.t1 = self.layer_1_conv(self.t1)
        self.t1 = F.tanh(self.t1)
        self.t2 = self.layer_2_conv(self.t1)
        return self.t2


class DerainDestoringNet_de(nn.Module):
    """
    The decoder part of DestoringNet.
    ref: @inproceedings{eigen2013restoring,
          title={Restoring an image taken through a window covered with dirt or rain},
          author={Eigen, David and Krishnan, Dilip and Fergus, Rob},
          booktitle={Proceedings of the IEEE international conference on computer vision},
          pages={633--640},
          year={2013}
        }

    """
    def __init__(self):
        super().__init__()
        # self.layer_1_padding = nn.ReflectionPad2d((7, 8, 7, 8))
        # self.layer_1_conv = nn.Conv2d(self.in_channels, 512, 16, 1, bias=True)
        # self.layer_2_conv = nn.Conv2d(512, 512, 1, bias=True)
        self.layer_3_padding = nn.ReflectionPad2d((3, 4, 3, 4))
        self.layer_3_conv = nn.Conv2d(512, 3, 8, bias=True)

    def forward(self, t2):
        self.t3 = self.layer_3_padding(t2)
        self.t3 = self.layer_3_conv(self.t3)
        return self.t3


class DerainNet(nn.Module):
    """
    TODO: Not finished. The problem is this structure cannot be split into a 'encoder' and a 'decoder'.
    ref: @article{fu2017clearing,
          title={Clearing the skies: A deep network architecture for single-image rain removal},
          author={Fu, Xueyang and Huang, Jiabin and Ding, Xinghao and Liao, Yinghao and Paisley, John},
          journal={IEEE Transactions on Image Processing},
          volume={26},
          number={6},
          pages={2944--2956},
          year={2017},
          publisher={IEEE}
        }
    """

    def __init__(self, setup):
        super().__init__()
        self.name = setup.name
        self.in_channels = setup.in_channels
        # TODO: low-pass filter
        layer_1_padding = nn.ReflectionPad2d((7, 8, 7, 8))
        layer_1_conv = nn.Conv2d(self.in_channels, 512, 16, 1, bias=True)
        layer_2_conv = nn.Conv2d(512, 512, 1, bias=True)
        layer_3_padding = nn.ReflectionPad2d((3, 4, 3, 4))
        layer_3_conv = nn.Conv2d(512, 3, 8, bias=True)

    def forward(self, x):
        pass


class DerainPReNet(nn.Module):
    """
    A network for deraining.


    """
    pass


class DerainPRN(nn.Module):
    """
    ref:
    code: https://github.com/csdwren/PReNet/blob/master/networks.py
    """
    def __init__(self, recurrent_iter=6):
        super(DerainPRN, self).__init__()
        self.iteration = recurrent_iter

        self.conv0 = nn.Sequential(nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU())
        self.res_conv1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv3 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv4 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv5 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv4 = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, input):
        x = input
        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list






