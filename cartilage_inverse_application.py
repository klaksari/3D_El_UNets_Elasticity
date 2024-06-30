# Import 

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import time

import torch
import torch.nn as nn
import elunet_3d_transversely_isotropic as eu3d

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
    prog='3D_ElUNet_Cartilage_Application',
    description='Sample code for 3D spatial estimation of tranversely isotropic elasticity parameters')

parser.add_argument('--unet_num_channels',
                    help='Num channels per unet depth (default=[64, 128, 256, 512])', type=int, nargs='+', default=[64, 128])

parser.add_argument(
    '-e', '--epochs', help='Maximum number of epochs', type=int, nargs=1)
parser.add_argument('-np', '--noise_percentage',
                    help='Gaussian noise percentage to add to strain data (default 5.0)', type=float, nargs=1, default=0.0)
parser.add_argument('-lr', '--learning_rate',
                    help='Initial learning rate (default 0.001)', type=float, nargs=1, default=[0.001])
parser.add_argument('--lr_update_threshold', help='Patience argument for scheduler (default 200000)',
                    type=int, nargs=1)
parser.add_argument('--loss_report_freq', help='Frequency of loss report updates',
                    type=int, nargs=1, default=[1])
parser.add_argument('--save_freq', help='Frequency of saving the model',
                    type=int, nargs=1, default=[5000])
parser.add_argument(
    '-pl', '--plot', help='plot_results, default is false', action="store_true")
parser.add_argument('--training_time', help='Training time in minutes that the model is expected to run',
                    type=float, nargs=1)
parser.add_argument('-ip', '--input_path', help='Input domain data path (default ./input_domain_data)',
                    type=str, nargs=1, default=['input_domain_data'])
parser.add_argument('-of', '--output_path',
                    help='Output path (default res**)', type=str, nargs=1, default='output')

args = parser.parse_args()


# Prepare training data
main_input_path = args.input_path[0]

# Read x loading to create a non-dimensionalization factor based on the mean stress on the boundaries
input_path_x = os.path.join(main_input_path,'XLoading')
sigma0_x = eu3d.convert_mat_to_tensor(os.path.join(input_path_x,'Sxx'),'Sxx')
sigma0_y = eu3d.convert_mat_to_tensor(os.path.join(input_path_x,'Syy'),'Syy')
sigma0_z = eu3d.convert_mat_to_tensor(os.path.join(input_path_x,'Szz'),'Szz')
sigma0 = (torch.mean(torch.abs(sigma0_x[:, -1, :]))+torch.mean(
    torch.abs(sigma0_y[-1, :, :]))+torch.mean(torch.abs(sigma0_z[:, :, -1])))/3
sigma0 = sigma0.float().to(device)
print('Maximum stress on the top boundary and non-dimensionalization factor is ' +
      str(sigma0.item()) + ' Pa')

# Read the input data from loading in the x direction
input_path_x = os.path.join(main_input_path,'XLoading')

stress_stack_1 = eu3d.stack_inputs_from_vars_3d(['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Sxz'], input_path_x, rescale=False, reshape_into_batch=False)
strain_stack_xloading = eu3d.stack_inputs_from_vars_noisy_3d(['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz'], input_path_x, percentage=args.noise_percentage[0], rescale=False, reshape_into_batch=True)

strain_stack_xloading[:, 3, :] = 2*strain_stack_xloading[:, 3, :]
strain_stack_xloading[:, 4, :] = 2*strain_stack_xloading[:, 4, :]
strain_stack_xloading[:, 5, :] = 2*strain_stack_xloading[:, 5, :]

strain_stack_xloading = strain_stack_xloading.float().to(device)

strain_stack_normalized_xloading = eu3d.stack_inputs_from_vars_noisy_3d(['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz'], input_path_x, percentage=args.noise_percentage[0], rescale=True, reshape_into_batch=False)
strain_stack_normalized_xloading = strain_stack_normalized_xloading.unsqueeze(0).float().to(device)
# create boundary stress tensors
stress_boundaries_xloading = {'sxx_bound_gt': torch.stack((stress_stack_1[0, :, 0, :], stress_stack_1[0, :, -1, :]), dim=0),
                       'szz_bound_gt': torch.stack((stress_stack_1[2, :, :, 0], stress_stack_1[2, :, :, -1]), dim=0),
                       'syy_bound_gt': torch.stack((stress_stack_1[1, 0, :, :], stress_stack_1[1, -1, :, :]), dim=0)}


# Read the input data from loading in the y direction
input_path_y = os.path.join(main_input_path,'YLoading')

stress_stack_2 = eu3d.stack_inputs_from_vars_3d(['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Sxz'], input_path_y, rescale=False, reshape_into_batch=False)
strain_stack_yloading = eu3d.stack_inputs_from_vars_noisy_3d(['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz'], input_path_y, percentage=args.noise_percentage[0], rescale=False, reshape_into_batch=True)

strain_stack_yloading[:, 3, :] = 2*strain_stack_yloading[:, 3, :]
strain_stack_yloading[:, 4, :] = 2*strain_stack_yloading[:, 4, :]
strain_stack_yloading[:, 5, :] = 2*strain_stack_yloading[:, 5, :]

strain_stack_yloading = strain_stack_yloading.float().to(device)

strain_stack_normalized_yloading = eu3d.stack_inputs_from_vars_noisy_3d(['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz'], input_path_y, percentage=args.noise_percentage[0], rescale=True, reshape_into_batch=False)
strain_stack_normalized_yloading = strain_stack_normalized_yloading.unsqueeze(0).float().to(device)

# create boundary stress tensors
stress_boundaries_yloading = {'sxx_bound_gt': torch.stack((stress_stack_2[0, :, 0, :], stress_stack_2[0, :, -1, :]), dim=0),
                       'szz_bound_gt': torch.stack((stress_stack_2[2, :, :, 0], stress_stack_2[2, :, :, -1]), dim=0),
                       'syy_bound_gt': torch.stack((stress_stack_2[1, 0, :, :], stress_stack_2[1, -1, :, :]), dim=0)}

# Read the input data from loading in the y direction
input_path_z = os.path.join(main_input_path,'ZLoading')

stress_stack_3 = eu3d.stack_inputs_from_vars_3d(['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Sxz'], input_path_z, rescale=False, reshape_into_batch=False)
strain_stack_zloading = eu3d.stack_inputs_from_vars_noisy_3d(['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz'], input_path_z, percentage=5, rescale=False, reshape_into_batch=True)

strain_stack_zloading[:, 3, :] = 2*strain_stack_zloading[:, 3, :]
strain_stack_zloading[:, 4, :] = 2*strain_stack_zloading[:, 4, :]
strain_stack_zloading[:, 5, :] = 2*strain_stack_zloading[:, 5, :]

strain_stack_zloading = strain_stack_zloading.float().to(device)

strain_stack_normalized_zloading = eu3d.stack_inputs_from_vars_noisy_3d(['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz'], input_path_z, percentage=5, rescale=True, reshape_into_batch=False)
strain_stack_normalized_zloading = strain_stack_normalized_zloading.unsqueeze(0).float().to(device)

# create boundary stress tensors
stress_boundaries_zloading = {'sxx_bound_gt': torch.stack((stress_stack_3[0, :, 0, :], stress_stack_3[0, :, -1, :]), dim=0),
                       'szz_bound_gt': torch.stack((stress_stack_3[2, :, :, 0], stress_stack_3[2, :, :, -1]), dim=0),
                       'syy_bound_gt': torch.stack((stress_stack_3[1, 0, :, :], stress_stack_3[1, -1, :, :]), dim=0)}


# Create the average strain tensor from the two loading conditions
mean_strain_stack_normalized = (strain_stack_normalized_xloading +
                           strain_stack_normalized_yloading +
                           strain_stack_normalized_zloading)/3

# Store input info in strain_dict and stress_boundary_dict
strain_dict = {'strain_stack_xloading': strain_stack_xloading,
               'strain_stack_yloading': strain_stack_yloading,
               'strain_stack_zloading': strain_stack_zloading,
               'strain_stack_normalized_xloading': strain_stack_normalized_xloading,
               'strain_stack_normalized_yloading': strain_stack_normalized_yloading,
               'strain_stack_normalized_zloading': strain_stack_normalized_zloading,
               'mean_strain_stack_normalized': mean_strain_stack_normalized}

stress_boundary_dict = {'stress_boundaries_xloading': stress_boundaries_xloading,
                'stress_boundaries_yloading': stress_boundaries_yloading,
                'stress_boundaries_zloading': stress_boundaries_zloading}

# Read the ground truth data for angles and parameters
input_path_param = os.path.join(main_input_path, 'AnglesAndParams')

angles = {'psi_gt': eu3d.convert_mat_to_tensor(os.path.join(input_path_param,'psi_volume'), 'psi_volume'),
          'theta_gt': eu3d.convert_mat_to_tensor(os.path.join(input_path_param,'theta_volume'), 'theta_volume'),}

parameters = {'ymxx_gt': eu3d.convert_mat_to_tensor(os.path.join(input_path_param,'YMxx'), 'YMxx'),
              'ymyy_gt': eu3d.convert_mat_to_tensor(os.path.join(input_path_param,'YMyy'), 'YMyy'),
              'gxy_gt': eu3d.convert_mat_to_tensor(os.path.join(input_path_param,'Gxy'), 'Gxy'),
              'gyz_gt': eu3d.convert_mat_to_tensor(os.path.join(input_path_param,'Gyz'), 'Gyz'),
              'prxy_gt': eu3d.convert_mat_to_tensor(os.path.join(input_path_param,'PRxy'), 'PRxy'),
              'pryz_gt': eu3d.convert_mat_to_tensor(os.path.join(input_path_param,'PRyz'), 'PRyz')}


# Train and save the results
batch_size = strain_stack_xloading.shape[0]

model = eu3d.DoubleUNET(
    in_channels_first=6,
    in_channels_second=6,
    out_channels_first=5,
    out_channels_second=6,
    features=args.unet_num_channels)

model = model.to(device)
model = model.float()
model = torch.compile(model)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=args.learning_rate[0])

if args.training_time and args.lr_update_threshold:
    patience = args.lr_update_threshold[0]
elif args.training_time and not args.lr_update_threshold:
    raise ValueError(
        'Please set a lr_update_threshold for the scheduler')
elif args.epochs and args.lr_update_threshold:
    patience = args.lr_update_threshold[0]
elif args.epochs and not args.lr_update_threshold:
    patience = int(args.epochs[0]/10)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=patience, threshold=0, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

if args.training_time and args.epochs:
    raise ValueError(
        'Please set either epochs or training_time argument, not both.')

elif args.training_time:
    model, loss_histories = eu3d.train_epoch_number(
        model, criterion, optimizer, scheduler,
        strain_dict,
        stress_boundary_dict,
        sigma0,
        angles, parameters,
        batch_size,
        training_duration_max=args.training_time[0],
        report_freq=args.loss_report_freq[0],
        save_freq=args.save_freq[0])
elif args.epochs:
    model, loss_histories = eu3d.train_epoch_number(
        model, criterion, optimizer, scheduler,
        strain_dict,
        stress_boundary_dict,
        sigma0,
        angles, parameters,
        batch_size, max_epochs=args.epochs[0],
        report_freq=args.loss_report_freq[0], save_freq=args.save_freq[0])

# Write output in files
main_output_path = args.output_path[0]
if not os.path.isdir(main_output_path):
    os.makedirs(main_output_path)

checkpoint = {
    "state_dict": model.state_dict(),
}
eu3d.save_checkpoint(checkpoint, filename=os.path.join(main_output_path, "my_checkpoint_cartilage.pth.tar"))


np.save('running_loss_history', np.array(
    loss_histories['running_loss_history']))


np.save(os.path.join(main_output_path, 'ymxx_mae_history'), np.array(loss_histories['ymxx_mae_history']))
np.save(os.path.join(main_output_path, 'ymyy_mae_history'), np.array(loss_histories['ymyy_mae_history']))
np.save(os.path.join(main_output_path, 'gxy_mae_history'), np.array(loss_histories['gxy_mae_history']))
np.save(os.path.join(main_output_path, 'gyz_mae_history'), np.array(loss_histories['gyz_mae_history']))
np.save(os.path.join(main_output_path, 'prxy_mae_history'), np.array(loss_histories['prxy_mae_history']))
np.save(os.path.join(main_output_path, 'pryz_mae_history'), np.array(loss_histories['pryz_mae_history']))

y = model.forward(mean_strain_stack_normalized,strain_stack_normalized_xloading)

zeta11_out = y[0][0, 0, :, :]
zeta22_out = y[0][0, 1, :, :]
zeta12_out = y[0][0, 2, :, :]
zeta23_out = y[0][0, 3, :, :]
Gxy_pred = sigma0*y[0][0, 4, :, :]
Gyz_pred = sigma0*(zeta22_out-zeta23_out)/2

YMxx_pred = sigma0*(- 2*zeta12_out**2 + zeta11_out*zeta22_out +
                    zeta11_out*zeta23_out)/(zeta22_out + zeta23_out)
YMyy_pred = sigma0*((zeta22_out - zeta23_out)*(- 2*zeta12_out**2 + zeta11_out *
                                               zeta22_out + zeta11_out*zeta23_out))/(- zeta12_out**2 + zeta11_out*zeta22_out)
PRxy_pred = zeta12_out/(zeta22_out + zeta23_out)
PRyz_pred = (- zeta12_out ** 2 + zeta11_out*zeta23_out) / \
    (- zeta12_out ** 2 + zeta11_out*zeta22_out)


dict_matlab = {"YMxx_pred": YMxx_pred.detach().cpu().numpy(),
               "YMyy_pred": YMyy_pred.detach().cpu().numpy(),
               "PRxy_pred": PRxy_pred.detach().cpu().numpy(),
               "PRyz_pred": PRyz_pred.detach().cpu().numpy(),
               "Gxy_pred": Gxy_pred.detach().cpu().numpy(),
               "Gyz_pred": Gyz_pred.detach().cpu().numpy()}

scipy.io.savemat(os.path.join(main_output_path, 'results.mat'), dict_matlab)

print('All results saved.')

if args.plot:
    plt.plot(np.array(loss_histories['ymxx_mae_history']), label='$E_{xx}$')
    plt.yscale('log')
    plt.legend()
    plt.plot(np.array(loss_histories['ymyy_mae_history']), label='$E_{yy}$')
    plt.yscale('log')
    plt.legend()

    plt.plot(np.array(loss_histories['gxy_mae_history']), label='$G_{xy}$')
    plt.yscale('log')
    plt.legend()
    plt.plot(np.array(loss_histories['gyz_mae_history']), label='$G_{yz}$')
    plt.yscale('log')
    plt.legend()

    plt.plot(np.array(loss_histories['prxy_mae_history']), label=r'$\nu_{xy}$')
    plt.yscale('log')
    plt.legend()
    plt.plot(np.array(loss_histories['pryz_mae_history']), label=r'$\nu_{yz}$')
    plt.yscale('log')
    plt.legend()
    plt.title('%MAE history')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.savefig(os.path.join(main_output_path, 'mean_absolute_error.png'))
    plt.show()