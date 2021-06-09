"""
The rib wrapper.

Ref: https://github.com/YannDubs/Mini_Decodable_Information_Bottleneck
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import PurePath
from datetime import datetime
from copy import copy
import os
import sys
sys.path.append(os.path.join(os.getcwd(), PurePath('../..')))


class RIB(nn.Module):
    def __init__(self, model_class_encoder, model_class_decoder,
                 model_setup_encoder, model_setup_decoder):
        super(RIB, self).__init__()

        self.model_setup_encoder = model_setup_encoder
        self.model_setup_decoder = model_setup_decoder
        if self.model_setup_encoder:
            self.model_setup_encoder.add_suffix('main')
        if self.model_setup_decoder:
            self.model_setup_decoder.add_suffix('main')

        # Define models
        if self.model_setup_encoder:
            self.encoder = model_class_encoder(self.model_setup_encoder)
        else:
            self.encoder = model_class_encoder()
        if self.model_setup_decoder:
            self.decoder = model_class_decoder(self.model_setup_decoder)
        else:
            self.decoder = model_class_decoder()
        self.DIB_phase = 'train_both'
        self.if_encoder_frozen = False
        self.if_decoder_reset = False

    def forward(self, X):
        if self.DIB_phase == 'train_both':  # Phase 1: train both encoder and decoder (Bob and Alice) to meet sufficiency condition.
            z_sample = self.encoder(X[0])
            y_pred = self.decoder(z_sample)
            return y_pred, z_sample
        '''
        elif self.DIB_phase == 'minimality_only':  # Phase 2: train the decoder to reach minimality.
            if not self.if_encoder_frozen:
                self.freeze_encoder()
            if not self.if_decoder_reset:
                self.reset_decoder()
            z_sample = self.encoder(X)
            y_pred = self.decoder(z_sample)
            return y_pred, z_sample
        '''
    '''
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.if_encoder_frozen = True

    def reset_decoder(self):
        self.decoder = MLP(self.MLP_setup_decoder)
        self.if_decoder_reset = True
    '''


class RIBLoss(nn.Module):
    def __init__(self, model_class_decoder, model_setup_decoder, n_decoders, loss_fn):
        super().__init__()
        self.n_decoders = n_decoders
        self.model_class_decoder = model_class_decoder
        self.model_setup_decoder = model_setup_decoder
        self.loss_fn = loss_fn

        decoder_setup_list = [copy(self.model_setup_decoder) for _ in range(self.n_decoders)]
        for idx, i_setup in enumerate(decoder_setup_list):
            if i_setup is not None:
                i_setup.add_suffix(f'hydra_{idx:02d}')
        if self.model_setup_decoder:
            self.decoders = nn.ModuleList(
                [model_class_decoder(i_setup) for i_setup in decoder_setup_list]
            )
        else:
            self.decoders = nn.ModuleList(
                [model_class_decoder() for _ in decoder_setup_list]
            )


        class ScaleGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, beta):
                ctx.save_for_backward(input)
                ctx.beta = beta
                return input

            @staticmethod
            def backward(ctx, grad_output):
                if ctx.needs_input_grad[0]:
                    grad_output = grad_output * (- ctx.beta)
                return grad_output, None

        self.scale_grad = ScaleGrad.apply

    def get_sufficiency(self, y_pred, labels):
        return self.loss_fn(y_pred, labels)

    def get_minimality(self, z_sample, batch_Ni):
        # H_V_nCz = [
        #     self.get_H_V_niCz(decoder, self.Ni_getter(i, x_idcs), z_sample)
        #     for i, decoder in enumerate(self.decoders)
        # ]
        H_V_nCz = [
            self.get_H_V_niCz(decoder, batch_Ni[i], z_sample)
            for i, decoder in enumerate(self.decoders)
        ]
        H_V_nCz = sum(H_V_nCz)
        # H_V_nCz = sum(H_V_nCz) / self.n_decoders
        return H_V_nCz

    def get_H_V_niCz(self, decoder, batch_Ni, z_sample):
        marg_pred_ni = decoder(z_sample)
        H_V_niCz = self.loss_fn(marg_pred_ni, batch_Ni)
        return H_V_niCz

    def forward(self, y_pred, z_sample, labels, batch_Ni, beta):
        """

        :param y_pred:
        :param z_sample:
        :param labels:
        :param batch_Ni: (Tensor) a batch of wrong labels for hydra heads with shape [batch, num_heads, ...]
        :param beta:
        :return:
        """
        # y_pred, z_sample = out
        # labels, x_idcs = targets

        # V-sufficiency
        H_V_yCz = self.get_sufficiency(y_pred, labels[0])
        if self.n_decoders == 0:
            dib = H_V_yCz
            return dib, 0
        # V-minimality
        z_sample = self.scale_grad(z_sample, -beta)
        H_V_nCz = self.get_minimality(z_sample, batch_Ni)

        # DIB
        dib = H_V_yCz + H_V_nCz
        return dib, H_V_nCz
        # return H_V_yCz, H_V_nCz


def train_step(batch_input, batch_labels, batch_indics,
               model_DIB, model_DIB_loss, beta,
               params_encoder, params_decoder_main, params_decoder_hydra,
               optimizer_encoder, optimizer_hydra_encoder, optimizer_hydra_decoder):
    y_pred, z_sample = model_DIB(batch_input)
    H_V_yCz, H_V_nCz = model_DIB_loss(y_pred, z_sample, batch_labels, batch_indics)
    grad_encoder_main = torch.autograd.grad(H_V_yCz, params_encoder, retain_graph=True, allow_unused=True)
    grad_encoder_hydra = torch.autograd.grad(H_V_nCz, params_encoder, retain_graph=True, allow_unused=True)
    grad_decoder_main = torch.autograd.grad(H_V_yCz, params_decoder_main, retain_graph=True, allow_unused=True)
    grad_decoder_hydra = torch.autograd.grad(H_V_nCz, params_decoder_hydra, retain_graph=False, allow_unused=True)

    optimizer_encoder.zero_grad()
    # Inverse the gradients on encoder network generated by hydra classifiers.
    for param, grad_main, grad_hydra in zip(params_encoder, grad_encoder_main, grad_encoder_hydra):
        if grad_hydra is not None:
            param.grad = (grad_main - beta * grad_hydra)
    for param, grad in zip(params_decoder_main, grad_decoder_main):
        param.grad = grad
    for param, grad in zip(params_decoder_hydra, grad_decoder_hydra):
        param.grad = grad

    optimizer_encoder.step()
    optimizer_hydra_encoder.step()
    optimizer_hydra_decoder.step()

    return y_pred

    # loss = F.cross_entropy(y_pred, batch_labels)
    # acc = (torch.argmax(y_pred).long() == batch_labels).float().mean()

    # return loss, acc


# def train_step_new(batch_input, batch_labels, batch_indics,
#                    model_DIB, model_DIB_loss, beta, optimizer):
#     y_pred, z_sample = model_DIB(batch_input)
#     dib = model_DIB_loss(y_pred, z_sample, batch_labels, batch_indics, beta)
#     dib.backward()
#     optimizer.step()
#
#     return y_pred
#
#
# def val_step(batch_input, batch_labels, model_DIB):
#     y_pred, _ = model_DIB(batch_input)
#     return y_pred
#     # loss = F.cross_entropy(y_pred, batch_labels)
#     # acc = (torch.argmax(y_pred).long() == batch_labels).float().mean()
#     # return loss, acc
#
#
# def find_model_params_by_name(model, name_str):
#     params = []
#     for name, param in model.named_parameters():
#         if name_str in name:
#             params.append(param)
#     return params
#
#
# def experiment_1():
#     device = torch.device('cuda:0')
#
#     # beta = 0.001
#     beta = 0.01
#     lr = 1e-3
#     n_decoders = 10
#     num_classes = 10
#
#     tot_epoch = 20
#     batch_size = 100
#     num_batch = 10
#
#     PATH_PROJ = '/scratch/uceelyu/pycharm_sync_sensitivityIB'
#
#     path_summary_folder = os.path.join(PATH_PROJ, f'VIB/DIB/results/summary')
#     print(f'{datetime.now()} I Open TensorBoard with:\n'
#           f'tensorboard --logdir={path_summary_folder} --host=0.0.0.0 --port=6007\n\n\n')
#
#     writer = SummaryWriter(log_dir=os.path.join(path_summary_folder, f'{datetime.now()}'))
#     m_loss = {'train': Measurement('loss_train'),
#               'val': Measurement('loss_val')}
#     m_acc = {'train': Measurement('acc_train'),
#              'val': Measurement('acc_val')}
#
#     data_loaders = cifar_mnist.load_torch_dataset(PATH_PROJ=PurePath(PATH_PROJ),
#                                                   dataset_name='cifar10',
#                                                   batch_size=batch_size,
#                                                   download=False,
#                                                   onehot=False,
#                                                   with_idx=True,)
#
#     encoder_setup = MLP_setup('encoder', dim_in=32*32*3, dim_out=128, n_hid_layers=4, dim_hid=128)
#     decoder_setup = MLP_setup('decoder', dim_in=128, dim_out=10, n_hid_layers=2, dim_hid=64)
#
#     DIB_model = DIB(MLP_setup_encoder=encoder_setup, MLP_setup_decoder=decoder_setup)
#     DIB_loss = DIBLoss(num_classes, n_decoders, decoder_setup, num_train=batch_size*len(data_loaders['train']))
#
#     DIB_model.to(device)
#     DIB_loss.to(device)
#
#     DIB_model.zero_grad()
#     DIB_loss.zero_grad()
#
#     params_encoder = find_model_params_by_name(DIB_model, 'encoder_main')
#     params_decoder_main = find_model_params_by_name(DIB_model, 'decoder_main')
#     params_decoder_hydra = find_model_params_by_name(DIB_loss, 'hydra')
#
#     optimizer_encoder = torch.optim.Adam(params_encoder, lr=lr)
#     optimizer_decoder_main = torch.optim.Adam(params_decoder_main, lr=lr)
#     optimizer_decoder_hydra = torch.optim.Adam(params_decoder_hydra, lr=lr)
#
#     loss = torch.nn.CrossEntropyLoss(reduction='mean')
#
#     tq = tqdm(total=tot_epoch, unit='ep', dynamic_ncols=True)
#     for ep in range(1, tot_epoch+1, 1):
#         m_loss['train'].clear()
#         m_loss['val'].clear()
#         m_acc['train'].clear()
#         m_acc['val'].clear()
#         for phase in ['train', 'val']:
#             tq.set_description(f'{phase} ep {ep}')
#             data_loader = data_loaders[phase]
#             for inputs, labels in data_loader:
#                 if phase == 'train' and step >= num_batch:
#                     break
#                 batch_inputs = inputs.reshape([batch_size, -1]).to(device)
#                 batch_labels = labels[0].reshape([-1]).to(device)
#                 batch_indices = labels[-1].reshape([-1]).to(device)
#                 with torch.set_grad_enabled(phase == 'train'):
#                     if phase == 'train':
#                         y_pred = train_step(batch_inputs, batch_labels, batch_indices, DIB_model, DIB_loss,
#                                             beta, params_encoder, params_decoder_main, params_decoder_hydra,
#                                             optimizer_encoder, optimizer_decoder_main, optimizer_decoder_hydra)
#                         # loss, acc = train_step(inputs, labels, indices, DIB_model, DIB_loss,
#                         #                        params_encoder,
#                         #                        optimizer_encoder, optimizer_decoder_main, optimizer_decoder_hydra)
#                 if phase == 'val':
#                     y_pred = val_step(batch_inputs, batch_labels, DIB_model)
#                     # loss, acc = val_step(inputs, labels, DIB_model)
#                 loss_value = loss(y_pred, batch_labels)
#                 acc = (torch.argmax(y_pred, dim=1).long() == batch_labels).float().mean()
#                 m_loss[phase].update(float(loss_value))
#                 m_acc[phase].update(float(acc))
#                 tq.set_postfix(loss=float(loss_value), acc=float(acc))
#             writer.add_scalar(f'loss/{phase}', m_loss[phase].average, ep)
#             writer.add_scalar(f'acc/{phase}', m_acc[phase].average, ep)
#         tq.update(1)
#
#
# def experiment_2():
#     device = torch.device('cuda:1')
#
#     beta = 100
#     # beta = 0.
#     lr = 1e-3
#     n_decoders = 4
#     # n_decoders = 0
#
#     num_classes = 10
#     tot_epoch = 300
#     batch_size = 100
#     num_batch = 10000
#
#     PATH_PROJ = '/scratch/uceelyu/pycharm_sync_sensitivityIB'
#
#     path_summary_folder = os.path.join(PATH_PROJ, f'VIB/DIB/results/summary')
#     print(f'{datetime.now()} I Open TensorBoard with:\n'
#           f'tensorboard --logdir={path_summary_folder} --host=0.0.0.0 --port=6007\n\n\n')
#
#     writer = SummaryWriter(log_dir=os.path.join(path_summary_folder, f'{datetime.now()}'
#                                                                      f'_hydra{n_decoders}'
#                                                                      f'_beta{beta}'))
#     m_loss = {'train': Measurement('loss_train'),
#               'val': Measurement('loss_val')}
#     m_acc = {'train': Measurement('acc_train'),
#              'val': Measurement('acc_val')}
#
#     data_loaders = cifar_mnist.load_torch_dataset(PATH_PROJ=PurePath(PATH_PROJ),
#                                                   dataset_name='cifar10',
#                                                   batch_size=batch_size,
#                                                   download=False,
#                                                   onehot=False,
#                                                   with_idx=True,)
#
#     # data_loaders = cifar_mnist.load_overlay(PATH_PROJ=PurePath(PATH_PROJ),
#     #                                         batch_size=batch_size,
#     #                                         onehot=False,
#     #                                         with_idx=True,)
#
#     encoder_setup = MLP_setup('encoder', dim_in=32*32*3, dim_out=1024, n_hid_layers=3, dim_hid=2048, softmax=False)
#     decoder_setup = MLP_setup('decoder', dim_in=1024, dim_out=10, n_hid_layers=1, dim_hid=128, softmax=False)
#
#     DIB_model = DIB(MLP_setup_encoder=encoder_setup, MLP_setup_decoder=decoder_setup)
#     DIB_loss = DIBLoss_new(num_classes, n_decoders, decoder_setup, num_train=batch_size*len(data_loaders['train']))
#
#     DIB_model.to(device)
#     DIB_loss.to(device)
#
#     DIB_model.zero_grad()
#     DIB_loss.zero_grad()
#
#     params_main = DIB_model.parameters()
#     params_hydra = DIB_loss.parameters()
#     params = list(params_main) + list(params_hydra)
#
#     optimizer = torch.optim.Adam(params, lr=lr)
#
#     loss = torch.nn.CrossEntropyLoss(reduction='mean')
#
#     tq = tqdm(total=tot_epoch, unit='ep', dynamic_ncols=True)
#     for ep in range(1, tot_epoch+1, 1):
#         m_loss['train'].clear()
#         m_loss['val'].clear()
#         m_acc['train'].clear()
#         m_acc['val'].clear()
#         for phase in ['train', 'val']:
#             tq.set_description(f'{phase} ep {ep}')
#             data_loader = data_loaders[phase]
#             for step, [inputs, labels] in enumerate(data_loader):
#                 if phase == 'train' and step >= num_batch:
#                     break
#                 batch_inputs = inputs.reshape([batch_size, -1]).to(device)
#                 batch_labels = labels[0].reshape([-1]).to(device)
#                 batch_indices = labels[-1].reshape([-1]).to(device)
#                 optimizer.zero_grad()
#                 with torch.set_grad_enabled(phase == 'train'):
#                     if phase == 'train':
#                         y_pred = train_step_new(batch_inputs, batch_labels, batch_indices,
#                                                 DIB_model, DIB_loss, beta, optimizer)
#                 if phase == 'val':
#                     y_pred = val_step(batch_inputs, batch_labels, DIB_model)
#                     # loss, acc = val_step(inputs, labels, DIB_model)
#                 loss_value = loss(y_pred, batch_labels)
#                 acc = (torch.argmax(y_pred, dim=1).long() == batch_labels).float().mean()
#                 m_loss[phase].update(float(loss_value))
#                 m_acc[phase].update(float(acc))
#                 tq.set_postfix(loss=float(loss_value), acc=float(acc))
#             writer.add_scalar(f'loss/{phase}', m_loss[phase].average, ep)
#             writer.add_scalar(f'acc/{phase}', m_acc[phase].average, ep)
#         tq.update(1)
#
#
# if __name__ == '__main__':
#     experiment_2()
#