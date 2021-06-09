"""
Training for Denoise MNIST

Created by Zhaoyan @ UCL
"""
import os
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))
import src.util.util_torch as util_torch
import src.util.util_reg as util_reg
import src.util.util_RIB as util_RIB
import src.rib as rib
from src.util.util_log import get_formatted_time, Log, Paths
from src.util.util_log import Measurement as M
from src.preprocessing.load_dataset import recipes


def run(temp_id, temp_ids, device, setup, logger, paths):

    record_suffix = f'{setup.task_name}:{setup.minimality}:{setup.beta}'
    paths.add_summary_suffix(record_suffix)
    paths.add_other_paths('folder:init', os.path.join(result_home_path, 'init'), True)
    paths.add_other_paths('file:init_main', os.path.join(paths.others['folder:init'],
                                                         f'main:{setup.encoder_name}.npy'), False)
    paths.add_other_paths('file:init_hydra', os.path.join(paths.others['folder:init'],
                                                          f'hydra:{setup.decoder_name}.npy'), False)

    m_suff = {'train': M('suff:train'),
              'val': M('suff:val')}
    m_mini = {'train': M('mini:train'),
              'val': M('mini:val')}
    m_mse = {'train': M('mse:train'),
             'val': M('mse:val'),
             'test': M('mse:test'),
             'practical': M('mse:practical')}
    m_psnr = {'train': M('psnr:train'),
              'val': M('psnr:val'),
              'test': M('psnr:test'),
              'practical': M('psnr:practical')}
    m_ssim = {'train': M('ssim:train'),
              'val': M('ssim:val'),
              'test': M('ssim:test'),
              'practical': M('ssim:practical')}

    print(f'{datetime.now()} I Open TensorBoard with:\n'
          f'tensorboard --logdir={paths.summary} --host=0.0.0.0 --port=6007\n\n\n')
    writer = SummaryWriter(log_dir=paths.summary)

    # -----------------Build model-----------------
    psnr_mse_getter = util_torch.PSNR_MSE(1.)
    ssim_getter = util_torch.SSIM()


    dataloaders = recipes(setup.task_name, setup.path_raw_dataset, setup.batch_size, setup.dataset_kwargs)
    dataloader_train, dataloader_val, dataloader_test, dataloader_practical,\
        input_idx, label_idx, noise_idx = util_RIB.select_dataset_components(setup.task_name,
                                                                             dataloaders,
                                                                             setup.minimality)
    dataloaders = {'train': dataloader_train,
                   'val': dataloader_val,
                   'test': dataloader_test,
                   'practical': dataloader_practical}

    RIB_model = rib.RIB(setup.encoder_class,
                        setup.decoder_class,
                        setup.encoder_model_setup,
                        setup.decoder_model_setup).to(device)

    loss_fn, loss_getter = util_torch.select_loss(setup.loss_name)
    RIB_loss = rib.RIBLoss(setup.decoder_class,
                           setup.decoder_model_setup,
                           n_decoders=len(noise_idx),
                           loss_fn=loss_fn).to(device)
    """
    for model, path_model_init in zip([RIB_model, RIB_loss],
                                      [paths.others['file:init_main'], paths.others['file:init_hydra']]):
        if not os.path.isfile(path_model_init):
            util_torch.save_model_weights(model, path_model_init)
        else:
            try:
                util_torch.load_model_weights(model, path_model_init, device)
            except():
                util_torch.save_model_weights(model, path_model_init)
    """
    params_main = RIB_model.parameters()
    params_hydra = RIB_loss.parameters()

    params_all = list(params_main) + list(params_hydra)

    optimizer = torch.optim.Adam(params_all, lr=setup.learning_rate)

    # -----------------Train model-----------------
    tq = tqdm(total=setup.tot_epochs, unit='ep', dynamic_ncols=True, desc='')
    for ep in range(1, setup.tot_epochs+1, 1):
        for phase in ['train', 'val']:
            m_suff[phase].clear()
            m_mini[phase].clear()
            m_mse[phase].clear()
            m_psnr[phase].clear()
            m_ssim[phase].clear()
            tq.set_description(f'id: {temp_id}/{temp_ids}: {phase.ljust(10, " ")} ep {ep:03d}')
            dataloader = dataloaders[phase]
            for step, samples in enumerate(dataloader):
                batch_inputs = []
                for i in input_idx:
                    batch_inputs.append(samples[i[0]][i[1]].to(device))
                batch_labels = []
                for i in label_idx:
                    batch_labels.append(samples[i[0]][i[1]].to(device))
                batch_noises = []
                for i in noise_idx:
                    batch_noises.append(samples[i[0]][i[1]].to(device))

                optimizer.zero_grad()
                y = None
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        reg_term = util_reg.select_reg(setup.reg_type, RIB_model, setup.reg_para)
                        y, mini = util_RIB.train_step(batch_inputs, batch_labels, batch_noises,
                                                      RIB_model, RIB_loss, setup.beta, optimizer, reg_term)
                if phase == 'val':
                    y = util_RIB.test_step(batch_inputs, RIB_model)
                loss_value = loss_getter(y, batch_labels[0])
                psnr_value, mse_value = psnr_mse_getter(y, batch_labels[0])
                ssim_value = ssim_getter(y, batch_labels[0])
                suff_value = float(loss_value)
                mini_value = float(mini)
                mse_value = float(mse_value)
                psnr_value = float(psnr_value)
                ssim_value = float(ssim_value)

                m_suff[phase].update(suff_value)
                m_mini[phase].update(mini_value)
                m_mse[phase].update(mse_value)
                m_psnr[phase].update(psnr_value)
                m_ssim[phase].update(ssim_value)
                tq.set_postfix(suff=suff_value, psnr=psnr_value)
            # End of epoch
            if ep == 1 or ep % setup.log_epochs == 0:
                writer.add_scalar(f'suff/{phase}', m_suff[phase].average, ep)
                writer.add_scalar(f'mini/{phase}', m_mini[phase].average, ep)
                writer.add_scalar(f'mse/{phase}', m_mse[phase].average, ep)
                writer.add_scalar(f'psnr/{phase}', m_psnr[phase].average, ep)
                writer.add_scalar(f'ssim/{phase}', m_ssim[phase].average, ep)

            m_suff[phase].write_tape(ep)
            m_mini[phase].write_tape(ep)
            m_mse[phase].write_tape(ep)
            m_psnr[phase].write_tape(ep)
            m_ssim[phase].write_tape(ep)
        tq.update(1)
    tq.close()

    # Save weights
    util_torch.save_model_weights(RIB_model, os.path.join(paths.weights, 'trained_model.npy'))

    tapes = {'suff:train': m_suff['train'].tape, 'suff:val': m_suff['val'].tape,
             'mini:train': m_mini['train'].tape, 'mini:val': m_mini['val'].tape,
             'mse:train':  m_mse['train'].tape,  'mse:val':  m_mse['val'].tape,
             'psnr:train': m_psnr['train'].tape, 'psnr:val': m_psnr['val'].tape,
             'ssim:train': m_ssim['train'].tape, 'ssim:val': m_ssim['val'].tape,
             }

    # Save tapes
    np.save(paths.curve, tapes)

    logger.update(key='suff:train', value=m_suff['train'].average)
    logger.update(key='suff:val',   value=m_suff['val'].average)
    logger.update(key='mini:train', value=m_mini['train'].average)
    logger.update(key='mini:val',   value=m_mini['val'].average)
    logger.update(key='mse:train', value=m_mse['train'].average)
    logger.update(key='mse:val',   value=m_mse['val'].average)
    logger.update(key='mse:test',  value=m_mse['test'].average)
    logger.update(key='psnr:train', value=m_psnr['train'].average)
    logger.update(key='psnr:val',   value=m_psnr['val'].average)
    logger.update(key='ssim:train', value=m_ssim['train'].average)
    logger.update(key='ssim:val',   value=m_ssim['val'].average)

    logger.update(dict=paths.get_all_paths())

    # -----------------Test model-----------------
    phases = ['test', 'practical']
    for phase in phases:
        dataloader = dataloaders[phase]
        measurements = {'mse': m_mse[phase],
                        'psnr': m_psnr[phase],
                        'ssim': m_ssim[phase]}
        util_RIB.evaluate(dataloader, phase, RIB_model, measurements, paths, device, psnr_mse_getter, ssim_getter,
                          patch_size=128, crop_stride=64)
        logger.update(key=f'mse:{phase}',  value=m_mse[phase].average)
        logger.update(key=f'psnr:{phase}',  value=m_psnr[phase].average)
        logger.update(key=f'ssim:{phase}',  value=m_ssim[phase].average)

    # Save log
    logger.save(paths.log)


if __name__ == '__main__':

    temp_ids = 1

    for temp_id in range(temp_ids):
        result_home_path = '/scratch/uceelyu/pycharm_sync_RIB/results/mnist_denoise'
        timestamp = get_formatted_time()

        logger = Log()
        paths = Paths(result_home_path, timestamp)

        # -----------------Train setup hyper-para-----------------
        device = util_torch.auto_select_GPU(dwell=1)
        encoder_setup = {'model_name': 'encoder',
                         'channel_in': 3,
                         'channel_out': 64,
                         'n_hid_layers': 4,
                         'channel_hid': 64,
                         'filter_size': 3,
                         'padding': 1,
                         'stride': 1,
                         }
        decoder_setup = {'model_name': 'decoder',
                         'channel_in': 64,
                         'channel_out': 3,
                         'n_hid_layers': 1,
                         'channel_hid': 0,
                         'filter_size': 3,
                         'padding': 1,
                         'stride': 1,
                         }

        setup = util_RIB.TrainSetupDerain(result_home_path,
                                          logger,
                                          timestamp=timestamp,
                                          tot_epochs=100,
                                          batch_size=100,
                                          train_size=-1,
                                          learning_rate=1e-3,
                                          beta=100.,
                                          minimality='noisy_input',
                                          task_name='mnist_denoise',
                                          encoder_name='FamilyCNN',
                                          decoder_name='FamilyCNN',
                                          loss_name='mse',
                                          reg_type='None',
                                          reg_para=1e-3,
                                          encoder_setup=encoder_setup,
                                          decoder_setup=decoder_setup,
                                          log_epochs=10,
                                          )

        run(temp_id, temp_ids, device, setup, logger, paths)
        # torch.cuda.empty_cache()
    exit()
