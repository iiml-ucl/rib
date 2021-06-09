import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import matplotlib
import os


def smooth(scalars, weight, window_size=21):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    smoothed = medfilt(smoothed, window_size).reshape([-1])
    return smoothed

# ['suff:train', 'suff:val', 'mini:train', 'mini:val', 'mse:train', 'mse:val', 'psnr:train', 'psnr:val', 'ssim:train', 'ssim:val']

path_save_folder = '/scratch/uceelyu/pycharm_sync_RIB/results/derain/postprocessing_plot'

paths_data_None = ['/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_19:17:32.821296.npy',
                   '/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_19:54:24.698361.npy',
                   '/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_20:31:07.948853.npy',
                   '/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_21:35:40.380751.npy',
                   '/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_22:40:00.458219.npy',]

paths_data_RIB = ['/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_19:16:53.925233.npy',
                  '/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_19:53:19.893751.npy',
                  '/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_21:34:51.681418.npy',
                  '/scratch/uceelyu/pycharm_sync_RIB/results/derain/curve/2021-05-29_22:39:12.713445.npy',]

data_list_RIB = [np.load(i, allow_pickle=True).item() for i in paths_data_RIB]
data_list_None = [np.load(i, allow_pickle=True).item() for i in paths_data_None]


loss_avg_val_RIB = []
loss_avg_val_None = []
loss_avg_train_RIB = []
loss_avg_train_None = []

loss_sigma_val_RIB = []
loss_sigma_val_None = []
loss_sigma_train_RIB = []
loss_sigma_train_None = []

mini_avg_val_RIB = []
mini_avg_val_None = []
mini_avg_train_RIB = []
mini_avg_train_None = []
mini_sigma_val_RIB = []
mini_sigma_val_None = []
mini_sigma_train_RIB = []
mini_sigma_train_None = []

ssim_avg_val_RIB = []
ssim_avg_val_None = []
ssim_avg_train_RIB = []
ssim_avg_train_None = []
ssim_sigma_val_RIB = []
ssim_sigma_val_None = []
ssim_sigma_train_RIB = []
ssim_sigma_train_None = []

psnr_avg_val_RIB = []
psnr_avg_val_None = []
psnr_avg_train_RIB = []
psnr_avg_train_None = []
psnr_sigma_val_RIB = []
psnr_sigma_val_None = []
psnr_sigma_train_RIB = []
psnr_sigma_train_None = []

print(data_list_RIB[0].keys())
# print(data_list_RIB[0]['suff:val'].tape)
# exit()

# exit()
# print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in data_list_RIB[0].items()) + "}")

step = 10

for ep in range(6000):
    # if ep % step == 0:
    if ep == 0 or (ep+1) % step == 0:
        # Suff
        avg_suff_train_RIB = np.average([i['suff:train'][ep]['average'] for i in data_list_RIB])
        avg_suff_train_None = np.average([i['suff:train'][ep]['average'] for i in data_list_None])
        sigma_suff_train_RIB = np.std([i['suff:train'][ep]['average'] for i in data_list_RIB])
        sigma_suff_train_None = np.std([i['suff:train'][ep]['average'] for i in data_list_None])
        loss_avg_train_RIB.append(avg_suff_train_RIB)
        loss_avg_train_None.append(avg_suff_train_None)
        loss_sigma_train_RIB.append(sigma_suff_train_RIB)
        loss_sigma_train_None.append(sigma_suff_train_None)

        avg_suff_val_RIB = np.average([i['suff:val'].tape[ep]['average'] for i in data_list_RIB])
        avg_suff_val_None = np.average([i['suff:val'].tape[ep]['average'] for i in data_list_None])
        sigma_suff_val_RIB = np.std([i['suff:val'].tape[ep]['average'] for i in data_list_RIB])
        sigma_suff_val_None = np.std([i['suff:val'].tape[ep]['average'] for i in data_list_None])
        loss_avg_val_RIB.append(avg_suff_val_RIB)
        loss_avg_val_None.append(avg_suff_val_None)
        loss_sigma_val_RIB.append(sigma_suff_val_RIB)
        loss_sigma_val_None.append(sigma_suff_val_None)
        # Mini
        avg_mini_train_RIB = np.average([i['mini:train'][ep]['average'] for i in data_list_RIB])
        avg_mini_train_None = np.average([i['mini:train'][ep]['average'] for i in data_list_None])
        sigma_mini_train_RIB = np.std([i['mini:train'][ep]['average'] for i in data_list_RIB])
        sigma_mini_train_None = np.std([i['mini:train'][ep]['average'] for i in data_list_None])
        mini_avg_train_RIB.append(avg_mini_train_RIB)
        mini_avg_train_None.append(avg_mini_train_None)
        mini_sigma_train_RIB.append(sigma_mini_train_RIB)
        mini_sigma_train_None.append(sigma_mini_train_None)

        avg_mini_val_RIB = np.average([i['mini:val'].tape[ep]['average'] for i in data_list_RIB])
        avg_mini_val_None = np.average([i['mini:val'].tape[ep]['average'] for i in data_list_None])
        sigma_mini_val_RIB = np.std([i['mini:val'].tape[ep]['average'] for i in data_list_RIB])
        sigma_mini_val_None = np.std([i['mini:val'].tape[ep]['average'] for i in data_list_None])
        mini_avg_val_RIB.append(avg_mini_val_RIB)
        mini_avg_val_None.append(avg_mini_val_None)
        mini_sigma_val_RIB.append(sigma_mini_val_RIB)
        mini_sigma_val_None.append(sigma_mini_val_None)
        # PSNR
        avg_psnr_train_RIB = np.average([i['psnr:train'][ep]['average'] for i in data_list_RIB])
        avg_psnr_train_None = np.average([i['psnr:train'][ep]['average'] for i in data_list_None])
        sigma_psnr_train_RIB = np.std([i['psnr:train'][ep]['average'] for i in data_list_RIB])
        sigma_psnr_train_None = np.std([i['psnr:train'][ep]['average'] for i in data_list_None])
        psnr_avg_train_RIB.append(avg_psnr_train_RIB)
        psnr_avg_train_None.append(avg_psnr_train_None)
        psnr_sigma_train_RIB.append(sigma_psnr_train_RIB)
        psnr_sigma_train_None.append(sigma_psnr_train_None)

        avg_psnr_val_RIB = np.average([i['psnr:val'].tape[ep]['average'] for i in data_list_RIB])
        avg_psnr_val_None = np.average([i['psnr:val'].tape[ep]['average'] for i in data_list_None])
        sigma_psnr_val_RIB = np.std([i['psnr:val'].tape[ep]['average'] for i in data_list_RIB])
        sigma_psnr_val_None = np.std([i['psnr:val'].tape[ep]['average'] for i in data_list_None])
        psnr_avg_val_RIB.append(avg_psnr_val_RIB)
        psnr_avg_val_None.append(avg_psnr_val_None)
        psnr_sigma_val_RIB.append(sigma_psnr_val_RIB)
        psnr_sigma_val_None.append(sigma_psnr_val_None)
        # SSIM
        avg_ssim_train_RIB = np.average([i['ssim:train'][ep]['average'] for i in data_list_RIB])
        avg_ssim_train_None = np.average([i['ssim:train'][ep]['average'] for i in data_list_None])
        sigma_ssim_train_RIB = np.std([i['ssim:train'][ep]['average'] for i in data_list_RIB])
        sigma_ssim_train_None = np.std([i['ssim:train'][ep]['average'] for i in data_list_None])
        ssim_avg_train_RIB.append(avg_ssim_train_RIB)
        ssim_avg_train_None.append(avg_ssim_train_None)
        ssim_sigma_train_RIB.append(sigma_ssim_train_RIB)
        ssim_sigma_train_None.append(sigma_ssim_train_None)

        avg_ssim_val_RIB = np.average([i['ssim:val'].tape[ep]['average'] for i in data_list_RIB])
        avg_ssim_val_None = np.average([i['ssim:val'].tape[ep]['average'] for i in data_list_None])
        sigma_ssim_val_RIB = np.std([i['ssim:val'].tape[ep]['average'] for i in data_list_RIB])
        sigma_ssim_val_None = np.std([i['ssim:val'].tape[ep]['average'] for i in data_list_None])
        ssim_avg_val_RIB.append(avg_ssim_val_RIB)
        ssim_avg_val_None.append(avg_ssim_val_None)
        ssim_sigma_val_RIB.append(sigma_ssim_val_RIB)
        ssim_sigma_val_None.append(sigma_ssim_val_None)

print('loss')
print(loss_avg_train_RIB[-1])
print(loss_avg_train_None[-1])
print(loss_sigma_train_RIB[-1])
print(loss_sigma_train_None[-1])

print(loss_avg_val_RIB[-1])
print(loss_avg_val_None[-1])
print(loss_sigma_val_RIB[-1])
print(loss_sigma_val_None[-1])

print('mini')
print(mini_avg_train_RIB[-1])
print(mini_avg_train_None[-1])
print(mini_sigma_train_RIB[-1])
print(mini_sigma_train_None[-1])

print(mini_avg_val_RIB[-1])
print(mini_avg_val_None[-1])
print(mini_sigma_val_RIB[-1])
print(mini_sigma_val_None[-1])

print('psnr')
print(psnr_avg_train_RIB[-1])
print(psnr_avg_train_None[-1])
print(psnr_sigma_train_RIB[-1])
print(psnr_sigma_train_None[-1])

print(psnr_avg_val_RIB[-1])
print(psnr_avg_val_None[-1])
print(psnr_sigma_val_RIB[-1])
print(psnr_sigma_val_None[-1])

print('ssim')
print(ssim_avg_train_RIB[-1])
print(ssim_avg_train_None[-1])
print(ssim_sigma_train_RIB[-1])
print(ssim_sigma_train_None[-1])

print(ssim_avg_val_RIB[-1])
print(ssim_avg_val_None[-1])
print(ssim_sigma_val_RIB[-1])
print(ssim_sigma_val_None[-1])

eps = np.asarray([1] + list(range(10, 6001, step)))

colors = ['#E74C3C', '#3498DB']
font = {'size': 12}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(4, 1, figsize=(6, 8))

RIB = smooth(loss_avg_val_RIB, 0.9)
Van = smooth(loss_avg_val_None, 0.9)
RIB_sigma = smooth(loss_sigma_val_RIB, 0.9, 21)
Van_sigma = smooth(loss_sigma_val_None, 0.9, 21)
axs[0].plot(eps, RIB, linestyle='-', color=colors[0], label='M-S validation')
axs[0].plot(eps, Van, linestyle='-', color=colors[1], label='Vanilla validation')
axs[0].fill_between(eps, RIB+RIB_sigma, RIB-RIB_sigma, color=colors[0], alpha=0.1)
axs[0].fill_between(eps, Van+Van_sigma, Van-Van_sigma, color=colors[1], alpha=0.1)
# axs[0].set_xlabel('epoch', fontsize=12)
axs[0].set_ylabel('sufficiency \nquadratic loss', fontsize=14)
axs[0].set_ylim(0.0005, 0.001)
axs[0].grid(True)

RIB = smooth(mini_avg_val_RIB, 0.6)
Van = smooth(mini_avg_val_None, 0.6)
RIB_sigma = smooth(mini_sigma_val_RIB, 0.9, 21)
Van_sigma = smooth(mini_sigma_val_None, 0.9, 21)
axs[1].plot(eps, RIB, linestyle='-', color=colors[0], label='M-S validation')
axs[1].plot(eps, Van, linestyle='-', color=colors[1], label='Vanilla validation')
axs[1].fill_between(eps, RIB+RIB_sigma, RIB-RIB_sigma, color=colors[0], alpha=0.1)
axs[1].fill_between(eps, Van+Van_sigma, Van-Van_sigma, color=colors[1], alpha=0.1)
# axs[1].set_xlabel('epoch', fontsize=12)
axs[1].set_ylabel('minimality \nquadratic loss', fontsize=14)
axs[1].set_ylim(0, 0.0005)
axs[1].grid(True)

RIB = smooth(psnr_avg_val_RIB, 0.6)
Van = smooth(psnr_avg_val_None, 0.6)
RIB_sigma = smooth(psnr_sigma_val_RIB, 0.9, 21)
Van_sigma = smooth(psnr_sigma_val_None, 0.9, 21)
axs[2].plot(eps, RIB, linestyle='-', color=colors[0], label='M-S validation')
axs[2].plot(eps, Van, linestyle='-', color=colors[1], label='Vanilla validation')
axs[2].fill_between(eps, RIB+RIB_sigma, RIB-RIB_sigma, color=colors[0], alpha=0.1)
axs[2].fill_between(eps, Van+Van_sigma, Van-Van_sigma, color=colors[1], alpha=0.1)
# axs[2].set_xlabel('epoch', fontsize=12)
axs[2].set_ylabel('PSNR', fontsize=12)
axs[2].set_ylim(30, 33)
axs[2].grid(True)

RIB = smooth(ssim_avg_val_RIB, 0.6)
Van = smooth(ssim_avg_val_None, 0.6)
RIB_sigma = smooth(ssim_sigma_val_RIB, 0.9, 21)
Van_sigma = smooth(ssim_sigma_val_None, 0.9, 21)
axs[3].plot(eps, RIB, linestyle='-', color=colors[0], label='M-S validation')
axs[3].plot(eps, Van, linestyle='-', color=colors[1], label='Vanilla validation')
axs[3].fill_between(eps, RIB+RIB_sigma, RIB-RIB_sigma, color=colors[0], alpha=0.1)
axs[3].fill_between(eps, Van+Van_sigma, Van-Van_sigma, color=colors[1], alpha=0.1)
axs[3].set_xlabel('epoch', fontsize=12)
axs[3].set_ylabel('SSIM', fontsize=12)
axs[3].set_ylim(0.9, 0.94)
axs[3].grid(True)
axs[3].legend(loc='lower right')

fig.tight_layout()
fig.savefig(os.path.join(path_save_folder, 'curves_derain.png'))
