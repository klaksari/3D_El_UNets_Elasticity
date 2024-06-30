# Import packages

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import time

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Preprocessing functions

def rescale_normalize(x):
    '''
    Rescale the image between min and max
    
    Parameters:
    - x (tensor): torch tensor. 
    
    Returns:
    torch tensor: Normalized tensor
    '''

    return (x - torch.min(x))/(torch.max(x)-torch.min(x))

def convert_mat_to_tensor(filename, var_name):
    '''
    Read a .mat MATLAB file containing one variable that is named the same as the filename and stores it as a torch tensor.

    Parameters:
    - filename (str): Filename without the ".mat" extension
    - var_name (str): Variable name in the .mat file

    Returns:
    torch tensor: Tensor with float precision
    '''

    ImageData = scipy.io.loadmat(filename)
    variable_of_interest = torch.tensor(
        np.array(ImageData[var_name])).float().to(device)
    return variable_of_interest

def stack_inputs_from_vars_2d(input_filenames, input_path, rescale=False):
    '''
    Stack 2D tensors in the 0 dimension.
    
    Parameters:
    - input_filenames (list): List of filenames to 2D files to be read.
    - input_path (str): Path to the directory that contains files to be read.
    - rescale (bool): Set to true if each channel needs to be normalized (Defaulted to False).
    
    Returns:
    torch tensor: An l-by-n-by-m tensor where l is the number of data channels, and n and m are dimensions of each channel.
    '''
    
    # Reads files and stacks them in the first dimension
    input_torch = []
    for i in range(len(input_filenames)):
        input_filename = os.path.join(input_path, input_filenames[i])
        input_torch.append(torch.tensor(
            np.loadtxt(input_filename, delimiter=',')))

    variable_stack = torch.stack(
        tuple(input_torch[i] for i in range(len(input_torch))), axis=0)
    if rescale == True:
        for i in range(len(input_torch)):
            variable_stack[i, :, :] = rescale_normalize(
                variable_stack[i, :, :])
    return variable_stack

def stack_inputs_from_vars_3d(input_filenames, input_path, rescale=False):
    '''
    Stack 3D tensors in the 0 dimension.
    
    Parameters:
    - input_filenames (list): List of filenames of 3D files to be read.
    - input_path (str): Path to the directory that contains the files to be read.
    - rescale (bool): Set to True if each channel needs to be normalized (Defaulted to False).
    
    Returns:
    torch tensor: An l-by-n-by-m-by-p tensor where l is the number of data channels, and n, m, and p are the dimensions of each channel.
    '''

    input_torch = []
    for i in range(len(input_filenames)):
        file_location = os.path.join(input_path, input_filenames[i]+'.mat')
        input_torch.append(convert_mat_to_tensor(
            file_location, input_filenames[i]))

    variable_stack = torch.stack(
        tuple(input_torch[i] for i in range(len(input_torch))), axis=0)
    if rescale == True:
        for i in range(len(input_torch)):
            variable_stack[i, :, :, :] = rescale_normalize(
                variable_stack[i, :, :, :])
    return variable_stack

def stack_inputs_from_vars_noisy_3d(input_filenames, input_path, percentage=5, random_seed_number=1, rescale=False):
    '''
    Stack 3D tensors in the 0 dimension and adds desired amount of Gaussian noise.
    
    Parameters:
    - input_filenames (list): List of filenames of 3D files to be read.
    - input_path (str): Path to the directory that contains the files to be read.
    - percentage (float): Percentage (between 0 and 100) of Gaussian noise to add to each data channel.
    - random_seed_number (int): Seed number for random noise generation.
    - rescale (bool): Set to true if each channel needs to be normalized (Defaulted to False).
    
    Returns:
    torch tensor: An l-by-n-by-m-by-p tensor where l is the number of data channels, and n, m, and p are the dimensions of each channel
    '''
    
    np.random.seed(random_seed_number)
    percentage = percentage/100
    # Reads files and stacks them in the first dimension
    input_torch = []
    for i in range(len(input_filenames)):
        filename = os.path.join(input_path, input_filenames[i])
        tensor = convert_mat_to_tensor(filename, input_filenames[i])
        if rescale:
            tensor = rescale_normalize(tensor)
        noise = torch.randn_like(tensor) * torch.std(tensor) * percentage
        noisy_tensor = tensor + noise
        input_torch.append(noisy_tensor)

    variable_stack = torch.stack(
        tuple(input_torch[i] for i in range(len(input_torch))), axis=0)
    if rescale:
        for i in range(len(input_torch)):
            variable_stack[i, :, :, :] = rescale_normalize(
                variable_stack[i, :, :, :])

    return variable_stack


class DoubleConv(nn.Module):
    # class that contains the double convolution at each step
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding=1, bias=False),
            # Because we are using batch normalization, bias might not be necessary
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=False),
            # Because we are using batch normalization, bias might not be necessary
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
  # class that contains the upsampling followed by convolution at each step
    def __init__(self, feature, scale_factor):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(feature*2, feature, 2, 1, padding='same', bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    '''Customized UNET class
        credits to Aladdin Persson's tutorial on UNet from scratch in PyTorch:
        https://youtu.be/IHq1t7NxS8k?si=K3JwDYH0LHY0S7z4
    '''
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128], convtrans_upsampling=False):
        super(UNET, self).__init__()
        # The convolutions lists should be stored in the ModuleList
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Pooling layer downsamples the image by half
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            # As we start the up path, we need to upsample each image (ConvTranspose2d with same kernel_size and stride as in the pooling layer)
            # and also divide the number of channels by half so that we can concat the output with the skip_connections later

            if convtrans_upsampling:
                self.ups.append(nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2))
            else:
                self.ups.append(UpSample(feature=feature, scale_factor=2))

            # the result of the concatenation of previous output with the skip connections (hence the input size of feature*2) goes throught the DoubleConv filters
            self.ups.append(DoubleConv(feature*2, feature))

        # At the bottleneck level the convolution filters are double the last item in features
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # the last layer simply brings the multi-channel space back to the image space with a conv kerenel of 1
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # a forward pass of the UNet
        skip_connections = []
        for down in self.downs:
            x = down(x)
            # store the result of each doubleconv from the down path in skip_connections
            skip_connections.append(x)
            x = self.pool(x)
        # When we get down, run the bottleneck block
        x = self.bottleneck(x)

        # For ease of use, let's reverse the order in skip_connections for the path up
        skip_connections = skip_connections[::-1]

        # we march in range(len(self.ups)) by steps of 2 since we have instances of up_conv and double_conv stored there
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]

            # in case the upsampling from ConvTranspose2d results in a shape not consistent with the skip_connection (for input images that are not divisible by 16)
            if x.shape != skip_connection.shape:
                print(x.shape)
                print(skip_connection.shape)
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2:])
            # concatenate the upsampled output with the skip_connection from the down_path
            # order of channels are batch, channel, height, width --> so the channel is dim = 1
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # then run the double_conv layer
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)


# Helper functions for training and visualization

def save_checkpoint(model, optimizer, filename="my_checkpoint_UNet.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    print("=> Saving checkpoint")
    torch.save(checkpoint, filename)


def compute_loss(y, parameter_and_stress_out, criterion,
                 strain_stack,
                 sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                 symmetry_x=True):
    '''
    Compute loss associated with network output.
    
    Parameters:
    - y (tensor): Network output tensor.
    - parameter_and_stress_out (bool): Set to True if the network outputs material parameters
    and stress and False if it outputs only material parameters.
    - criterion (torch.nn.MSELoss): The loss function used to compute the loss.
    - strain_stack (tensor): Ground truth 3D strain tensor of shape (1, 6, n, m, p), where n, m, and p are the dimensions of each strain component.
    - sxx_bound_gt (tensor): Ground truth xx stress values at the lateral boundaries of shape (2, n, p).
    - syy_bound_gt (tensor): Ground truth yy stress values at the axial boundaries of shape (2, m, p).
    - szz_bound_gt (tensor): Ground truth zz stress values at the axial boundaries of shape (2, n, m).
    - symmetry_x (bool): Set to True if the material is symmetric along the x-axis (Default: True).
    
    Returns:
    tensor: Loss value.
    '''
    exx_gt = strain_stack[0, 0, :]
    eyy_gt = strain_stack[0, 1, :]
    ezz_gt = strain_stack[0, 2, :]
    exy_gt = strain_stack[0, 3, :]
    eyz_gt = strain_stack[0, 4, :]
    exz_gt = strain_stack[0, 5, :]
    zeros_tensor = torch.zeros_like(exx_gt).to(device)

    if parameter_and_stress_out:
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
        loss_sxx_out = criterion(c11_out*exx_gt +
                                 c12_out*(eyy_gt+ezz_gt), sxx_out)
        loss_syy_out = criterion(c11_out*eyy_gt +
                                 c12_out*(exx_gt+ezz_gt), syy_out)
        loss_szz_out = criterion(c11_out*ezz_gt +
                                 c12_out*(exx_gt+eyy_gt), szz_out)
        loss_sxy_out = criterion(c33_out*exy_gt, sxy_out)
        loss_syz_out = criterion(c33_out*eyz_gt, syz_out)
        loss_sxz_out = criterion(c33_out*exz_gt, sxz_out)

        loss_const = loss_sxx_out + loss_syy_out + loss_szz_out + \
            loss_sxy_out + loss_syz_out + loss_sxz_out
    else:
        lame1_out = y[0, 0, :]
        lame2_out = y[0, 1, :]
        c11_out = 2*lame2_out + lame1_out
        c12_out = lame1_out
        c33_out = 2*lame2_out
        sxx_out = c11_out*exx_gt + c12_out*(eyy_gt+ezz_gt)
        syy_out = c11_out*eyy_gt + c12_out*(exx_gt+ezz_gt)
        szz_out = c11_out*ezz_gt + c12_out*(exx_gt+eyy_gt)
        sxy_out = c33_out*exy_gt
        syz_out = c33_out*eyz_gt
        sxz_out = c33_out*exz_gt

    sxx_bound_out = torch.stack((sxx_out[:, 0, :], sxx_out[:, -1, :]), dim=0)
    szz_bound_out = torch.stack((szz_out[:, :, 0], szz_out[:, :, -1]), dim=0)
    syy_bound_out = torch.stack((syy_out[0, :, :], syy_out[-1, :, :]), dim=0)

    loss_bound_sxx = criterion(sxx_bound_out, sxx_bound_gt)
    loss_bound_szz = criterion(szz_bound_out, szz_bound_gt)
    loss_bound_syy = criterion(syy_bound_out, syy_bound_gt)

    spacing = 1 / \
        (((zeros_tensor.shape[0]-1) +
         (zeros_tensor.shape[1]-1)+(zeros_tensor.shape[2]-1))/3)

    _, sxx_x_out, _ = torch.gradient(sxx_out, spacing=spacing)
    syy_y_out, _, _ = torch.gradient(syy_out, spacing=spacing)
    _, _, szz_z_out = torch.gradient(szz_out, spacing=spacing)
    sxy_y_out, sxy_x_out, _ = torch.gradient(sxy_out, spacing=spacing)
    syz_y_out, _, syz_z_out = torch.gradient(syz_out, spacing=spacing)
    _, sxz_x_out, sxz_z_out = torch.gradient(sxz_out, spacing=spacing)

    if symmetry_x:
        sxx_x_out[:, int(sxx_x_out.shape[1]/2)+1:, :] = - \
            sxx_x_out[:, int(sxx_x_out.shape[1]/2)+1:, :]
        sxy_x_out[:, int(sxy_x_out.shape[1]/2)+1:, :] = - \
            sxy_x_out[:, int(sxy_x_out.shape[1]/2)+1:, :]
        sxz_x_out[:, int(sxz_x_out.shape[1]/2)+1:, :] = - \
            sxz_x_out[:, int(sxz_x_out.shape[1]/2)+1:, :]

    loss_equib_x = criterion(sxx_x_out + sxy_y_out + sxz_z_out, zeros_tensor)
    loss_equib_y = criterion(sxy_x_out+syy_y_out+syz_z_out, zeros_tensor)
    loss_equib_z = criterion(sxz_x_out+syz_y_out+szz_z_out, zeros_tensor)

    if parameter_and_stress_out:
        loss = loss_const + loss_bound_sxx + loss_bound_syy + loss_bound_szz +\
            loss_equib_x + loss_equib_y + loss_equib_z
    else:
        loss = loss_bound_sxx + loss_bound_syy + loss_bound_szz +\
            loss_equib_x + loss_equib_y + loss_equib_z
    return loss


def compute_loss_weighted_paramstress(y, criterion,
                                      strain_stack,
                                      sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                      loss_weight_constit, loss_weight_boundx, loss_weight_boundy, loss_weight_boundz, loss_weight_res,
                                      symmetry_x=True):
    '''
    Compute loss associated with network output in the self-adaptive loss weighting scenario.
    
    Parameters:
    - y (tensor): Network output tensor.
    - parameter_and_stress_out (bool): Set to True if the network outputs material parameters
    and stress and False if it outputs only material parameters.
    - criterion (torch.nn.MSELoss): The loss function used to compute the loss.
    - strain_stack (tensor): Ground truth 3D strain tensor of shape (1, 6, n, m, p), where n, m, and p are the dimensions of each strain component.
    - sxx_bound_gt (tensor): Ground truth xx stress values at the lateral boundaries of shape (2, n, p).
    - syy_bound_gt (tensor): Ground truth yy stress values at the axial boundaries of shape (2, m, p).
    - szz_bound_gt (tensor): Ground truth zz stress values at the axial boundaries of shape (2, n, m).
    - loss_weight_constit (tensor): Weight for the constitutive equations term.
    - loss_weight_boundx (tensor): Weight for the xx stress values at the boundaries perpendicular to the x direction.
    - loss_weight_boundy (tensor): Weight for the yy stress values at the boundaries perpendicular to the y direction.
    - loss_weight_boundz (tensor): Weight for the zz stress values at the boundaries perpendicular to the z direction.
    - loss_weight_res (tensor): Weight for the static equilibrium term.
    - symmetry_x (bool): Set to True if the material is symmetric along the x-axis (Default: True).
    
    Returns:
    tensor: Loss value.
    '''

    exx_gt = strain_stack[0, 0, :]
    eyy_gt = strain_stack[0, 1, :]
    ezz_gt = strain_stack[0, 2, :]
    exy_gt = strain_stack[0, 3, :]
    eyz_gt = strain_stack[0, 4, :]
    exz_gt = strain_stack[0, 5, :]
    zeros_tensor = torch.zeros_like(exx_gt).to(device)

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
    loss_sxx_out = criterion(loss_weight_constit*(c11_out*exx_gt +
                                                  c12_out*(eyy_gt+ezz_gt)), loss_weight_constit*sxx_out)
    loss_syy_out = criterion(loss_weight_constit*(c11_out*eyy_gt +
                                                  c12_out*(exx_gt+ezz_gt)), loss_weight_constit*syy_out)
    loss_szz_out = criterion(loss_weight_constit*(c11_out*ezz_gt +
                                                  c12_out*(exx_gt+eyy_gt)), loss_weight_constit*szz_out)
    loss_sxy_out = criterion(
        loss_weight_constit*(c33_out*exy_gt), loss_weight_constit*sxy_out)
    loss_syz_out = criterion(
        loss_weight_constit*(c33_out*eyz_gt), loss_weight_constit*syz_out)
    loss_sxz_out = criterion(
        loss_weight_constit*(c33_out*exz_gt), loss_weight_constit*sxz_out)

    loss_const = loss_sxx_out + loss_syy_out + loss_szz_out + \
        loss_sxy_out + loss_syz_out + loss_sxz_out

    sxx_bound_out = loss_weight_boundx * \
        torch.stack((sxx_out[:, 0, :], sxx_out[:, -1, :]), dim=0)
    syy_bound_out = loss_weight_boundy * \
        torch.stack((syy_out[0, :, :], syy_out[-1, :, :]), dim=0)
    szz_bound_out = loss_weight_boundz * \
        torch.stack((szz_out[:, :, 0], szz_out[:, :, -1]), dim=0)

    sxx_bound_gt = loss_weight_boundx*sxx_bound_gt
    syy_bound_gt = loss_weight_boundy*syy_bound_gt
    szz_bound_gt = loss_weight_boundz*szz_bound_gt

    loss_bound_sxx = criterion(sxx_bound_out, sxx_bound_gt)
    loss_bound_szz = criterion(szz_bound_out, szz_bound_gt)
    loss_bound_syy = criterion(syy_bound_out, syy_bound_gt)

    spacing = 1 / \
        (((zeros_tensor.shape[0]-1) +
         (zeros_tensor.shape[1]-1)+(zeros_tensor.shape[2]-1))/3)

    _, sxx_x_out, _ = torch.gradient(sxx_out, spacing=spacing)
    syy_y_out, _, _ = torch.gradient(syy_out, spacing=spacing)
    _, _, szz_z_out = torch.gradient(szz_out, spacing=spacing)
    sxy_y_out, sxy_x_out, _ = torch.gradient(sxy_out, spacing=spacing)
    syz_y_out, _, syz_z_out = torch.gradient(syz_out, spacing=spacing)
    _, sxz_x_out, sxz_z_out = torch.gradient(sxz_out, spacing=spacing)

    sxx_x_out = sxx_x_out*loss_weight_res
    syy_y_out = syy_y_out*loss_weight_res
    szz_z_out = szz_z_out*loss_weight_res
    sxy_y_out = sxy_y_out*loss_weight_res
    sxy_x_out = sxy_x_out*loss_weight_res
    syz_y_out = syz_y_out*loss_weight_res
    syz_z_out = syz_z_out*loss_weight_res
    sxz_x_out = sxz_x_out*loss_weight_res
    sxz_z_out = sxz_z_out*loss_weight_res

    if symmetry_x:
        sxx_x_out[:, int(sxx_x_out.shape[1]/2)+1:, :] = - \
            sxx_x_out[:, int(sxx_x_out.shape[1]/2)+1:, :]
        sxy_x_out[:, int(sxy_x_out.shape[1]/2)+1:, :] = - \
            sxy_x_out[:, int(sxy_x_out.shape[1]/2)+1:, :]
        sxz_x_out[:, int(sxz_x_out.shape[1]/2)+1:, :] = - \
            sxz_x_out[:, int(sxz_x_out.shape[1]/2)+1:, :]

    loss_equib_x = criterion(sxx_x_out + sxy_y_out + sxz_z_out, zeros_tensor)
    loss_equib_y = criterion(sxy_x_out+syy_y_out+syz_z_out, zeros_tensor)
    loss_equib_z = criterion(sxz_x_out+syz_y_out+szz_z_out, zeros_tensor)

    loss = loss_const + loss_bound_sxx + loss_bound_syy + loss_bound_szz +\
        loss_equib_x + loss_equib_y + loss_equib_z

    return loss


def train_running_time(model, strain_stack, strain_stack_normalized,
                       sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                       youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                       symmetry_x=True, parameter_and_stress_out=True,
                       training_duration_max=60,
                       loss_report_freq=500, early_stopping_threshold=1e-8):
    '''
    Run training in a given amount of time.
    
    Parameters:
    - model (UNET object): An instance of the UNET class.
    - strain_stack (tensor): Strain tensor where
    channels are stacked in the order of xx, yy, zz, xy, yz, and xz.
    - strain_stack_normalized (tensor): Normalized strain tensor where 
    channels are stacked in the order of xx, yy, zz, xy, yz, and xz.
    - sxx_bound_gt (tensor): xx stress values at the lateral boundaries.
    - syy_bound_gt (tensor): yy stress values at the axial boundaries.
    - szz_bound_gt (tensor): zz stress values at the axial boundaries.
    - youngs_gt (tensor): Young's modulus ground truth distribution provided for real-time accuracy report.
    - poissons_gt (tensor): Poisson's ratio ground truth distribution provided for real-time accuracy report.
    - sigma_0 (float): reference characteristic stress value for dimensionless implementation.
    - criterion (torch.nn.MSELoss): In default mode, this is a MSELoss object.
    - optimizer: Pytorch optimizer.
    - scheduler: Pytorch scheduler for learning rate decay control.
    - symmetry_x (bool): Set to True if the material is symmetric along the x-axis, False otherwise.
    - parameter_and_stress_out (bool): Set to True if the network outputs material parameters
    and stress, and False if it outputs only material parameters.
    - training_duration_max (float): Training time in minutes.
    - loss_report_freq (int): Frequency of loss reporting.
    - early_stopping_threshold (float): Threshold for early stopping of training.
    
    Returns:
    - model (UNET object): Instance of the UNET class after training.
    - loss_histories (dict): Dictionary containing various loss and accuracy values across training.
    '''
    
    
    running_loss_history = []
    youngs_mae_history = []
    poissons_mae_history = []
    mem_allec_history = []

    training_duration = 0
    e = 0
    start = time.time()

    while training_duration < training_duration_max:

        y = model.forward(strain_stack_normalized)

        loss = compute_loss(y, parameter_and_stress_out, criterion,
                            strain_stack,
                            sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                            symmetry_x=symmetry_x)

        with torch.no_grad():
            lame1_pred_dim = sigma_0*y[0, 0, :]
            lame2_pred_dim = sigma_0*y[0, 1, :]
            Youngs_pred = lame2_pred_dim * \
                (3*lame1_pred_dim + 2*lame2_pred_dim) / \
                (lame1_pred_dim+lame2_pred_dim)
            Poissons_pred = lame1_pred_dim/(2*(lame1_pred_dim+lame2_pred_dim))

            MAE_Youngs = 100 * \
                torch.abs(Youngs_pred-youngs_gt)/torch.abs(youngs_gt)
            MAE_Youngs = torch.mean(MAE_Youngs)

            MAE_Poissons = 100 * \
                torch.abs(Poissons_pred-poissons_gt)/torch.abs(poissons_gt)
            MAE_Poissons = torch.mean(MAE_Poissons)

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        if e % loss_report_freq == 0:
            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")
            print('Youngs_MAE: ', "{:.2f}".format(
                MAE_Youngs.item()), 'Poissons_MAE: ', "{:.2f}".format(MAE_Poissons.item()))
        e += 1

        # the running loss for every opoch is the average of loss associated with the loss from all the iterations from batches in that epoch
        running_loss_history.append(loss.item())
        youngs_mae_history.append(MAE_Youngs.item())
        poissons_mae_history.append(MAE_Poissons.item())
        mem_allec_history.append(mem_allec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        end = time.time()
        training_duration = (end - start)/60

        if loss < early_stopping_threshold:
            print('Early stopping. Threshold reached.')
            loss_histories = {}
            loss_histories['running_loss_history'] = running_loss_history
            loss_histories['youngs_mae_history'] = youngs_mae_history
            loss_histories['poissons_mae_history'] = poissons_mae_history
            print('Training took {:.2f} minutes'.format(training_duration))
            return model, loss_histories
    loss_histories = {}
    loss_histories['running_loss_history'] = running_loss_history
    loss_histories['youngs_mae_history'] = youngs_mae_history
    loss_histories['poissons_mae_history'] = poissons_mae_history
    print('Training took {:.2f} minutes'.format(training_duration))
    return model, loss_histories


def train_running_time_weighted_loss(model, strain_stack, strain_stack_normalized,
                                     sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                     loss_weight_constit, loss_weight_boundx, loss_weight_boundy, loss_weight_boundz, loss_weight_res,
                                     youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                     symmetry_x=True, training_duration_max=60,
                                     loss_report_freq=500, early_stopping_threshold=1e-8):
    '''
    Run training for a given number of epochs using the self-adaptive spatial weighting approach.
    
    Parameters:
    - model (UNET object): An instance of the UNET class.
    - strain_stack (tensor): Strain tensor where
    channels are stacked in the order of xx, yy, zz, xy, yz, and xz.
    - strain_stack_normalized (tensor): Normalized strain tensor where 
    channels are stacked in the order of xx, yy, zz, xy, yz, and xz.
    - sxx_bound_gt (tensor): xx stress values at the lateral boundaries.
    - syy_bound_gt (tensor): yy stress values at the axial boundaries.
    - szz_bound_gt (tensor): zz stress values at the axial boundaries.
    - loss_weight_constit (tensor): Weight for the constitutive equations term.
    - loss_weight_boundx (tensor): Weight for the xx stress values at the boundaries perpendicular to the x direction.
    - loss_weight_boundy (tensor): Weight for the yy stress values at the boundaries perpendicular to the y direction.
    - loss_weight_boundz (tensor): Weight for the zz stress values at the boundaries perpendicular to the z direction.
    - loss_weight_res (tensor): Weight for the static equilibrium term.    
    - youngs_gt (tensor): Young's modulus ground truth distribution provided for real-time accuracy report.
    - poissons_gt (tensor): Poisson's ratio ground truth distribution provided for real-time accuracy report.
    - sigma_0 (float): reference characteristic stress value for dimensionless implementation.
    - criterion (torch.nn.MSELoss): In default mode, this is a MSELoss object.
    - optimizer: Pytorch optimizer.
    - scheduler: Pytorch scheduler for learning rate decay control.
    - symmetry_x (bool): Set to True if the material is symmetric along the x-axis, False otherwise.
    - parameter_and_stress_out (bool): Set to True if the network outputs material parameters
    and stress, and False if it outputs only material parameters.
    - training_duration_max (float): Training time in minutes.
    - loss_report_freq (int): Frequency of loss reporting.
    - early_stopping_threshold (float): Threshold for early stopping of training.
    
    Returns:
    - model (UNET object): Instance of the UNET class after training.
    - loss_histories (dict): Dictionary containing various loss and accuracy values across training.
    '''    

    running_loss_history = []
    running_loss_history_weighted = []
    youngs_mae_history = []
    poissons_mae_history = []
    mem_allec_history = []

    training_duration = 0
    e = 0
    start = time.time()

    while training_duration < training_duration_max:

        y = model.forward(strain_stack_normalized)

        loss = compute_loss_weighted_paramstress(y, criterion,
                                                 strain_stack,
                                                 sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                                 loss_weight_constit, loss_weight_boundx, loss_weight_boundy, loss_weight_boundz, loss_weight_res,
                                                 symmetry_x=symmetry_x)

        with torch.no_grad():
            loss_unweighted = compute_loss(y, True, criterion,
                                        strain_stack,
                                        sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                        symmetry_x=symmetry_x)
            lame1_pred_dim = sigma_0*y[0, 0, :]
            lame2_pred_dim = sigma_0*y[0, 1, :]
            Youngs_pred = lame2_pred_dim * \
                (3*lame1_pred_dim + 2*lame2_pred_dim) / \
                (lame1_pred_dim+lame2_pred_dim)
            Poissons_pred = lame1_pred_dim/(2*(lame1_pred_dim+lame2_pred_dim))

            MAE_Youngs = 100 * \
                torch.abs(Youngs_pred-youngs_gt)/torch.abs(youngs_gt)
            MAE_Youngs = torch.mean(MAE_Youngs)

            MAE_Poissons = 100 * \
                torch.abs(Poissons_pred-poissons_gt)/torch.abs(poissons_gt)
            MAE_Poissons = torch.mean(MAE_Poissons)

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        if e % loss_report_freq == 0:
            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")
            print('Youngs_MAE: ', "{:.2f}".format(
                MAE_Youngs.item()), 'Poissons_MAE: ', "{:.2f}".format(MAE_Poissons.item()))
        e += 1

        # the running loss for every opoch is the average of loss associated with the loss from all the iterations from batches in that epoch
        running_loss_history.append(loss_unweighted.item())
        running_loss_history_weighted.append(loss.item())
        youngs_mae_history.append(MAE_Youngs.item())
        poissons_mae_history.append(MAE_Poissons.item())
        mem_allec_history.append(mem_allec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        end = time.time()
        training_duration = (end - start)/60

        if loss < early_stopping_threshold:
            print('Early stopping. Threshold reached.')
            loss_histories = {}
            loss_histories['running_loss_history'] = running_loss_history
            loss_histories['running_loss_history_weighted'] = running_loss_history_weighted
            loss_histories['youngs_mae_history'] = youngs_mae_history
            loss_histories['poissons_mae_history'] = poissons_mae_history
            print('Training took {:.2f} minutes'.format(training_duration))
            return model, loss_histories
    loss_histories = {}
    loss_histories['running_loss_history'] = running_loss_history
    loss_histories['running_loss_history_weighted'] = running_loss_history_weighted
    loss_histories['youngs_mae_history'] = youngs_mae_history
    loss_histories['poissons_mae_history'] = poissons_mae_history
    print('Training took {:.2f} minutes'.format(training_duration))
    return model, loss_histories


def train_epoch_number(model, strain_stack, strain_stack_normalized,
                       sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                       youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                       symmetry_x=True, parameter_and_stress_out=True, max_epochs=1000,
                       loss_report_freq=500, early_stopping_threshold=1e-8):
    """
    Train the model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The neural network model.
        strain_stack (torch.Tensor): The input strain stack.
        strain_stack_normalized (torch.Tensor): The normalized input strain stack.
        sxx_bound_gt (torch.Tensor): The ground truth sxx boundary.
        syy_bound_gt (torch.Tensor): The ground truth syy boundary.
        szz_bound_gt (torch.Tensor): The ground truth szz boundary.
        youngs_gt (torch.Tensor): The ground truth Young's modulus.
        poissons_gt (torch.Tensor): The ground truth Poisson's ratio.
        sigma_0 (float): Scaling factor for the predicted parameters.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        symmetry_x (bool): Whether to enforce symmetry along the x-axis. Defaults to True.
        parameter_and_stress_out (bool): Whether to output both parameters and stresses. Defaults to True.
        max_epochs (int): The maximum number of epochs to train. Defaults to 1000.
        loss_report_freq (int): The frequency at which to report the loss. Defaults to 500.
        early_stopping_threshold (float): The threshold for early stopping. Defaults to 1e-8.

    Returns:
        tuple: A tuple containing the trained model and the loss histories.

    """
    running_loss_history = []
    youngs_mae_history = []
    poissons_mae_history = []
    mem_allec_history = []

    e = 0
    training_duration = 0
    start = time.time()

    for e in range(max_epochs):

        y = model.forward(strain_stack_normalized)

        loss = compute_loss(y, parameter_and_stress_out, criterion,
                            strain_stack,
                            sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                            symmetry_x=symmetry_x)

        with torch.no_grad():
            lame1_pred_dim = sigma_0*y[0, 0, :, :]
            lame2_pred_dim = sigma_0*y[0, 1, :, :]
            Youngs_pred = lame2_pred_dim *\
                (3*lame1_pred_dim + 2*lame2_pred_dim) /\
                (lame1_pred_dim+lame2_pred_dim)
            Poissons_pred = lame1_pred_dim/(2*(lame1_pred_dim+lame2_pred_dim))

            MAE_Youngs = 100 * \
                torch.abs(Youngs_pred-youngs_gt)/torch.abs(youngs_gt)
            MAE_Youngs = torch.mean(MAE_Youngs)

            MAE_Poissons = 100 * \
                torch.abs(Poissons_pred-poissons_gt)/torch.abs(poissons_gt)
            MAE_Poissons = torch.mean(MAE_Poissons)

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        if e % loss_report_freq == 0:
            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")
            print('Youngs_MAE: ', "{:.2f}".format(
                MAE_Youngs.item()), 'Poissons_MAE: ', "{:.2f}".format(MAE_Poissons.item()))
        # the running loss for every opoch is the average of loss associated with the loss from all the iterations from batches in that epoch
        running_loss_history.append(loss.item())
        youngs_mae_history.append(MAE_Youngs.item())
        poissons_mae_history.append(MAE_Poissons.item())
        mem_allec_history.append(mem_allec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        end = time.time()
        training_duration = (end-start)/60
        if loss < early_stopping_threshold:
            print('Early stopping. Threshold reached.')
            print('Training took {:.2f} minutes'.format(training_duration))
            loss_histories = {}
            loss_histories['running_loss_history'] = running_loss_history
            loss_histories['youngs_mae_history'] = youngs_mae_history
            loss_histories['poissons_mae_history'] = poissons_mae_history
            return model, loss_histories
    loss_histories = {}
    loss_histories['running_loss_history'] = running_loss_history
    loss_histories['youngs_mae_history'] = youngs_mae_history
    loss_histories['poissons_mae_history'] = poissons_mae_history
    print('Training took {:.2f} minutes'.format(training_duration))
    return model, loss_histories


def train_epoch_number_weighted_loss(model, strain_stack, strain_stack_normalized,
                                     sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                     loss_weight_constit, loss_weight_boundx, loss_weight_boundy, loss_weight_boundz, loss_weight_res,
                                     youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                     symmetry_x=True, max_epochs=1000,
                                     loss_report_freq=500, early_stopping_threshold=1e-8):
    """
    Trains the model for a specified number of epochs using weighted loss.

    Args:
        model (nn.Module): The neural network model.
        strain_stack (torch.Tensor): The input strain stack.
        strain_stack_normalized (torch.Tensor): The normalized input strain stack.
        sxx_bound_gt (torch.Tensor): The ground truth sxx boundary.
        syy_bound_gt (torch.Tensor): The ground truth syy boundary.
        szz_bound_gt (torch.Tensor): The ground truth szz boundary.
        loss_weight_constit (float): The weight for the constitutive loss.
        loss_weight_boundx (float): The weight for the x-boundary loss.
        loss_weight_boundy (float): The weight for the y-boundary loss.
        loss_weight_boundz (float): The weight for the z-boundary loss.
        loss_weight_res (float): The weight for the residual loss.
        youngs_gt (torch.Tensor): The ground truth Young's modulus.
        poissons_gt (torch.Tensor): The ground truth Poisson's ratio.
        sigma_0 (float): The scaling factor for the predicted stress.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        symmetry_x (bool): Whether to assume symmetry along the x-axis. Defaults to True.
        max_epochs (int): The maximum number of epochs to train. Defaults to 1000.
        loss_report_freq (int): The frequency at which to report the loss. Defaults to 500.
        early_stopping_threshold (float): The threshold for early stopping. Defaults to 1e-8.

    Returns:
        nn.Module: The trained model.
        dict: A dictionary containing the loss histories.
    """
    running_loss_history = []
    running_loss_history_weighted = []
    youngs_mae_history = []
    poissons_mae_history = []
    mem_allec_history = []

    training_duration = 0
    start = time.time()

    for e in range(max_epochs):

        y = model.forward(strain_stack_normalized)

        loss = compute_loss_weighted_paramstress(y, criterion,
                                                 strain_stack,
                                                 sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                                 loss_weight_constit, loss_weight_boundx, loss_weight_boundy, loss_weight_boundz, loss_weight_res,
                                                 symmetry_x=symmetry_x)

        with torch.no_grad():
            loss_unweighted = compute_loss(y, True, criterion,
                                        strain_stack,
                                        sxx_bound_gt, syy_bound_gt, szz_bound_gt,
                                        symmetry_x=symmetry_x)
            lame1_pred_dim = sigma_0*y[0, 0, :, :]
            lame2_pred_dim = sigma_0*y[0, 1, :, :]
            Youngs_pred = lame2_pred_dim *\
                (3*lame1_pred_dim + 2*lame2_pred_dim) /\
                (lame1_pred_dim+lame2_pred_dim)
            Poissons_pred = lame1_pred_dim/(2*(lame1_pred_dim+lame2_pred_dim))

            MAE_Youngs = 100 * \
                torch.abs(Youngs_pred-youngs_gt)/torch.abs(youngs_gt)
            MAE_Youngs = torch.mean(MAE_Youngs)

            MAE_Poissons = 100 * \
                torch.abs(Poissons_pred-poissons_gt)/torch.abs(poissons_gt)
            MAE_Poissons = torch.mean(MAE_Poissons)

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        if e % loss_report_freq == 0:
            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")
            print('Youngs_MAE: ', "{:.2f}".format(
                MAE_Youngs.item()), 'Poissons_MAE: ', "{:.2f}".format(MAE_Poissons.item()))
        # the running loss for every opoch is the average of loss associated with the loss from all the iterations from batches in that epoch
        running_loss_history.append(loss_unweighted.item())
        running_loss_history_weighted.append(loss.item())
        youngs_mae_history.append(MAE_Youngs.item())
        poissons_mae_history.append(MAE_Poissons.item())
        mem_allec_history.append(mem_allec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        end = time.time()
        training_duration = (end-start)/60
        if loss < early_stopping_threshold:
            print('Early stopping. Threshold reached.')
            print('Training took {:.2f} minutes'.format(training_duration))
            loss_histories = {}
            loss_histories['running_loss_history'] = running_loss_history
            loss_histories['running_loss_history_weighted'] = running_loss_history_weighted
            loss_histories['youngs_mae_history'] = youngs_mae_history
            loss_histories['poissons_mae_history'] = poissons_mae_history
            return model, loss_histories
    loss_histories = {}
    loss_histories['running_loss_history'] = running_loss_history
    loss_histories['running_loss_history_weighted'] = running_loss_history_weighted
    loss_histories['youngs_mae_history'] = youngs_mae_history
    loss_histories['poissons_mae_history'] = poissons_mae_history
    print('Training took {:.2f} minutes'.format(training_duration))
    return model, loss_histories


def gt_estimation_plotter(filename, gt_img_list, estimated_img_list, vmin, vmax, cmap='jet'):
    '''
    Plot the estimated and ground truth 3D images side-by-side along with the relative error.
    
    Parameters:
    - filename (str): Name of the file to save the plot.
    - gt_img_list (list): List of ground truth 3D images.
    - estimated_img_list (list): List of estimated 3D images from the model.
    - vmin (float): Minimum value for the colormap.
    - vmax (float): Maximum value for the colormap.
    - cmap (str): Colormap (matplotlib standards).
    
    Returns:
    None
    '''

    f, axes = plt.subplots(len(gt_img_list), 3, sharey=True, figsize=(10, 10))

    for i in range(len(gt_img_list)):
        gt_img = gt_img_list[i]
        estimated_img = estimated_img_list[i]
        if isinstance(gt_img, torch.Tensor):
            gt_img = gt_img.detach().cpu().numpy()

        if isinstance(estimated_img, torch.Tensor):
            estimated_img = estimated_img.detach().cpu().numpy()

        ax1_sub = axes[i, 0].pcolor(gt_img, cmap=cmap, vmin=vmin,
                                    vmax=vmax)
        axes[i, 0].set_aspect('equal', 'box')
        axes[i, 0].set_title('Ground truth')
        ax2_sub = axes[i, 1].pcolor(estimated_img, cmap=cmap, vmin=vmin,
                                    vmax=vmax)
        axes[i, 1].set_aspect('equal', 'box')
        axes[i, 1].set_title('Estimated')

        ax3_sub = axes[i, 2].pcolor(100*np.abs(estimated_img-gt_img)/gt_img, cmap=cmap, vmin=0,
                                    vmax=20)
        axes[i, 2].set_aspect('equal', 'box')
        axes[i, 2].set_title('Error')

        f.colorbar(ax1_sub, ax=axes[i, 0], fraction=0.046, pad=0.04)
        f.colorbar(ax2_sub, ax=axes[i, 1], fraction=0.046, pad=0.04)
        f.colorbar(ax3_sub, ax=axes[i, 2], fraction=0.046, pad=0.04)

        axes[i, 0].xaxis.set_visible(False)
        axes[i, 1].xaxis.set_visible(False)
        axes[i, 2].xaxis.set_visible(False)
        axes[i, 0].yaxis.set_visible(False)
        axes[i, 1].yaxis.set_visible(False)
        axes[i, 2].yaxis.set_visible(False)

    plt.savefig(filename)
    plt.show()
