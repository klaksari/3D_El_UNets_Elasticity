import argparse
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import os
import elunet_3d_isotropic as eu3d
import scipy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
    prog='3D_ElUNet_Brain_Application',
    description='Sample code for 3D spatial estimation of elasticity parameters')


parser.add_argument('-ps', '--parameter_and_stress',
                    action="store_true")
parser.add_argument('--unet_num_channels',
                    help='Num channels per unet depth (default=[64, 128, 256, 512])', type=int, nargs='+', default=[64, 128])
parser.add_argument(
    '-e', '--epochs', help='Maximum number of epochs', type=int, nargs=1)
parser.add_argument('-wl', '--weighted_loss',
                    help='Assign spatial weights to boundary and residual loss terms for adverserial (min-max) optimization', action="store_true")
parser.add_argument('-np', '--noise_percentage',
                    help='Gaussian noise percentage to add to strain data (default 5.0)', type=float, nargs=1, default=0.0)
parser.add_argument('-lr', '--learning_rate',
                    help='Initial learning rate (default 0.001)', type=float, nargs=1, default=[0.001])
parser.add_argument('--lr_update_threshold', help='Patience argument for scheduler (default 200000)',
                    type=int, nargs=1)
parser.add_argument('-ld', '--loading_dir',
                    help='loading direction for loading either x, y, or z', type=str, nargs=1, default='z')
parser.add_argument('--conv_upsampling',
                    help='Turn on convolution transpose upsampling method, otherwise bilinear upsampling is performed in the UNet up path', action="store_true")
parser.add_argument('--loss_report_freq', help='Frequency of loss report updates',
                    type=int, nargs=1, default=[1])
parser.add_argument(
    '-pl', '--plot', help='plot_results, default is false', action="store_true")
parser.add_argument('--training_time', help='Training time in minutes that the model is expected to run',
                    type=float, nargs=1)

parser.add_argument('-ip', '--input_path', help='Input domain data path (default ./input_domain_data)',
                    type=str, nargs=1, default=['input_domain_data'])
parser.add_argument('-of', '--output_path',
                    help='Output path (default res**)', type=str, nargs=1, default='output')

args = parser.parse_args()


# Read domain strain maps

strain_stack = eu3d.stack_inputs_from_vars_noisy_3d(
    ['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz'], args.input_path[0], rescale=False).float().unsqueeze(0)
strain_stack_normalized = eu3d.stack_inputs_from_vars_noisy_3d(
    ['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz'], args.input_path[0], percentage=args.noise_percentage[0], rescale=True).float().unsqueeze(0)
stress_stack = eu3d.stack_inputs_from_vars_3d(
    ['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Sxz'], args.input_path[0], rescale=False).float().unsqueeze(0)
# Read the stress distribution the max of which will be assigned as the non-dimensionalization factor
if args.loading_dir[0] == 'x':
    sigma_0 = torch.max(torch.abs(stress_stack[0, 0, :, -1, :]))
elif args.loading_dir[0] == 'y':
    sigma_0 = torch.max(torch.abs(stress_stack[0, 1, -1, :, :]))
elif args.loading_dir[0] == 'z':
    sigma_0 = torch.max(torch.abs(stress_stack[0, 2, :, :, -1]))
else:
    raise ValueError(
        'loading_dir argument should be either x, y, or z')


print('Maximum stress on the top boundary and non-dimensionalization factor is ' +
      str(sigma_0.item()) + ' Pa')


youngs_gt = eu3d.convert_mat_to_tensor(
    os.path.join(args.input_path[0], 'Youngs.mat'), 'Youngs')
poissons_gt = eu3d.convert_mat_to_tensor(os.path.join(
    args.input_path[0], 'Poissons.mat'), 'Poissons')


sxx_bound_gt = torch.stack(
    (stress_stack[0, 0, :, 0, :], stress_stack[0, 0, :, -1, :]), dim=0)/sigma_0
syy_bound_gt = torch.stack(
    (stress_stack[0, 1, 0, :, :], stress_stack[0, 1, -1, :, :]), dim=0)/sigma_0
szz_bound_gt = torch.stack(
    (stress_stack[0, 2, :, :, 0], stress_stack[0, 2, :, :, -1]), dim=0)/sigma_0


if args.parameter_and_stress:
    model = eu3d.UNET(in_channels=6, out_channels=8, features=args.unet_num_channels,
                      convtrans_upsampling=args.conv_upsampling)
else:
    model = eu3d.UNET(in_channels=6, out_channels=2, features=args.unet_num_channels,
                      convtrans_upsampling=args.conv_upsampling)


model = model.to(device)
model = model.float()
learning_rate = args.learning_rate[0]
criterion = nn.MSELoss()


if args.weighted_loss:
    loss_weight_constit = torch.ones_like(
        strain_stack[0, 1, :], requires_grad=True)
    loss_weight_boundx = torch.ones_like(sxx_bound_gt, requires_grad=True)
    loss_weight_boundy = torch.ones_like(syy_bound_gt, requires_grad=True)
    loss_weight_boundz = torch.ones_like(szz_bound_gt, requires_grad=True)

    loss_weight_res = torch.ones_like(
        strain_stack[0, 1, :], requires_grad=False)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': loss_weight_constit, 'maximize': True},
        {'params': loss_weight_boundx, 'maximize': True},
        {'params': loss_weight_boundy, 'maximize': True},
        {'params': loss_weight_boundz, 'maximize': True}
    ], lr=learning_rate)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    if args.weighted_loss:
        if not args.parameter_and_stress:
            raise ValueError(
                'Weighted loss distribution is only available for training where the parameter_and_stress is True')
        model, loss_histories = eu3d.train_running_time_weighted_loss(model, strain_stack, strain_stack_normalized,
                                                                      sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                                                      loss_weight_constit, loss_weight_boundx, loss_weight_boundy, loss_weight_boundz, loss_weight_res,
                                                                      youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                                                      symmetry_x=True,  training_duration_max=args.training_time[0],
                                                                      loss_report_freq=args.loss_report_freq[0], early_stopping_threshold=1e-8)
    else:
        model, loss_histories = eu3d.train_running_time(model, strain_stack, strain_stack_normalized,
                                                        sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                                        youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                                        symmetry_x=True, parameter_and_stress_out=args.parameter_and_stress,
                                                        training_duration_max=args.training_time[0],
                                                        loss_report_freq=args.loss_report_freq[0], early_stopping_threshold=1e-8)
elif args.epochs:
    if args.weighted_loss:
        if not args.parameter_and_stress:
            raise ValueError(
                'Weighted loss distribution is only available for training where the parameter_and_stress is True')
        model, loss_histories = eu3d.train_epoch_number_weighted_loss(model, strain_stack, strain_stack_normalized,
                                                                      sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                                                      loss_weight_constit, loss_weight_boundx, loss_weight_boundy, loss_weight_boundz, loss_weight_res,
                                                                      youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                                                      symmetry_x=True, max_epochs=args.epochs[0],
                                                                      loss_report_freq=args.loss_report_freq[0], early_stopping_threshold=1e-8)
    else:
        model, loss_histories = eu3d.train_epoch_number(model, strain_stack, strain_stack_normalized,
                                                        sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                                        youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                                        symmetry_x=True, parameter_and_stress_out=args.parameter_and_stress,
                                                        max_epochs=args.epochs[0],
                                                        loss_report_freq=args.loss_report_freq[0], early_stopping_threshold=1e-8)


if not os.path.isdir(args.output_path[0]):
    os.mkdir(args.output_path[0])

eu3d.save_checkpoint(
    model, optimizer, filename=os.path.join(args.output_path[0], 'my_checkpoint_UNet_modular.pth.tar'))

if args.plot:
    plt.plot(loss_histories['running_loss_history'], label='training loss')
    plt.yscale('log')

    plt.plot(loss_histories['youngs_mae_history'], label='E MAE')
    plt.plot(loss_histories['poissons_mae_history'], label='nu MAE')
    plt.legend()

    plt.savefig(os.path.join(args.output_path[0], 'loss_history.png'))
    plt.show()

    y = model.forward(strain_stack_normalized)
    if args.parameter_and_stress:
        lame1_out = y[0, 0, :]
        lame2_out = y[0, 1, :]
        sxx_out = y[0, 2, :]
        syy_out = y[0, 3, :]
        szz_out = y[0, 4, :]
        sxy_out = y[0, 5, :]
        syz_out = y[0, 6, :]
        sxz_out = y[0, 7, :]
        c11_out = 2*lame2_out + lame1_out
        c12_out = lame1_out
        c33_out = 2*lame2_out
    else:
        lame1_out = y[0, 0, :]
        lame2_out = y[0, 1, :]
        c11_out = 2*lame2_out + lame1_out
        c12_out = lame1_out
        c33_out = 2*lame2_out
        sxx_out = c11_out*strain_stack[0, 0, :] + c12_out * \
            (strain_stack[0, 1, :]+strain_stack[0, 2, :])
        syy_out = c11_out*strain_stack[0, 1, :] + c12_out * \
            (strain_stack[0, 0, :]+strain_stack[0, 2, :])
        szz_out = c11_out*strain_stack[0, 2, :] + c12_out * \
            (strain_stack[0, 0, :]+strain_stack[0, 1, :])
        sxy_out = c33_out*strain_stack[0, 3, :]
        syz_out = c33_out*strain_stack[0, 4, :]
        sxz_out = c33_out*strain_stack[0, 5, :]

    lame1_out = sigma_0*lame1_out
    lame2_out = sigma_0*lame2_out
    sxx_out = sigma_0*sxx_out
    syy_out = sigma_0*syy_out
    sxy_out = sigma_0*sxy_out

    youngs_pred = lame2_out * \
        (3*lame1_out + 2*lame2_out)/(lame1_out+lame2_out)
    poissons_pred = lame1_out/(2*(lame1_out+lame2_out))

    gt_img_list = [youngs_gt[40, :, :].squeeze(
    )/1000, youngs_gt[80, :, :].squeeze()/1000, youngs_gt[120, :, :].squeeze()/1000]
    estimated_img_list = [youngs_pred[40, :, :].squeeze(
    )/1000, youngs_pred[80, :, :].squeeze()/1000, youngs_pred[120, :, :].squeeze()/1000]
    eu3d.gt_estimation_plotter(os.path.join(
        args.output_path[0], 'youngs_estim.png'), gt_img_list, estimated_img_list, 0.8, 2.2, cmap='jet')

    gt_img_list = [poissons_gt[40, :, :].squeeze(
    ), poissons_gt[80, :, :].squeeze(), poissons_gt[120, :, :].squeeze()]
    estimated_img_list = [poissons_pred[40, :, :].squeeze(
    ), poissons_pred[80, :, :].squeeze(), poissons_pred[120, :, :].squeeze()]
    eu3d.gt_estimation_plotter(os.path.join(
        args.output_path[0], 'poissons_estim.png'), gt_img_list, estimated_img_list, 0.3, 0.5, cmap='jet')

    dict_matlab = {"youngs_pred": youngs_pred.detach().cpu().numpy(),
                   "poissons_pred": poissons_pred.detach().cpu().numpy()}

    if args.weighted_loss:
        dict_matlab["loss_weight_constit"] = loss_weight_constit.detach(
        ).cpu().numpy()
        dict_matlab["loss_weight_boundx"] = loss_weight_boundx.detach(
        ).cpu().numpy()
        dict_matlab["loss_weight_boundy"] = loss_weight_boundy.detach(
        ).cpu().numpy()
        dict_matlab["loss_weight_boundz"] = loss_weight_boundz.detach(
        ).cpu().numpy()

    scipy.io.savemat(
        os.path.join(args.output_path[0], 'results.mat'), dict_matlab)
    np.save(os.path.join(args.output_path[0], 'loss_history'),
            np.array(loss_histories['running_loss_history']))
    np.save(os.path.join(args.output_path[0], 'youngs_mae_hist'),
            np.array(loss_histories['youngs_mae_history']))
    np.save(os.path.join(args.output_path[0], 'poissons_mae_hist'),
            np.array(loss_histories['poissons_mae_history']))