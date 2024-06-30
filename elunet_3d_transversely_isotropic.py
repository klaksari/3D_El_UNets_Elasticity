# Import packages

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import time

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
# import some useful modules from torchvision
from torchvision import datasets, transforms
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Read files and preprocess into torch tensors


def rescale_normalize(x):
    '''
    rescale x by normalizing it with respect to its min and max values

    Parameters:
    - x (torch tensor): input tensor

    Returns:
    - normalized tensor
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

def stack_inputs_from_vars_3d(input_filenames, input_path, rescale=False, reshape_into_batch=False):
    '''
    Stack 3D tensors in the 0 dimension.
    
    Parameters:
    - input_filenames (list): List of filenames of 3D files to be read.
    - input_path (str): Path to the directory that contains the files to be read.
    - rescale (bool): Set to True if each channel needs to be normalized (Defaulted to False).
    - reshape_into_batch (bool): Set to True if the output tensor needs to be reshaped into a batch (Defaulted to False).
    
    Returns:
    if reshape_into_batch is False
        torch tensor: An p-by-l-by-m-by-n tensor where p is the number of data channels, and l, m, and n are the dimensions of each channel.
    if reshape_into_batch is True
        torch tensor: An l*m*n-by-p-by-1 tensor where p is the number of data channels, and l, m, and n are the dimensions of each channel.
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
    if reshape_into_batch:
        batch_size = torch.numel(variable_stack[0, :])
        stack_batch = variable_stack[0, :].reshape(batch_size, 1, 1)
        for i in range(1, variable_stack.shape[0]):
            stack_batch = torch.cat(
                (stack_batch, variable_stack[i, :].reshape(batch_size, 1, 1)), axis=1)
        return stack_batch
    return variable_stack

def stack_inputs_from_vars_noisy_3d(input_filenames, input_path, percentage=5, random_seed_number=1, rescale=False, reshape_into_batch=False):
    '''
    Stack 3D tensors in the 0 dimension and adds desired amount of Gaussian noise.
    
    Parameters:
    - input_filenames (list): List of filenames of 3D files to be read.
    - input_path (str): Path to the directory that contains the files to be read.
    - percentage (float): Percentage (between 0 and 100) of Gaussian noise to add to each data channel.
    - random_seed_number (int): Seed number for random noise generation.
    - rescale (bool): Set to true if each channel needs to be normalized (Defaulted to False).
    - reshape_into_batch (bool): Set to true if the output tensor needs to be reshaped into a batch (Defaulted to False).
    
    Returns:
    if reshape_into_batch is False
        torch tensor: An p-by-l-by-m-by-n tensor where p is the number of data channels, and l, m, and n are the dimensions of each channel.
    if reshape_into_batch is True
        torch tensor: An l*m*n-by-p-by-1 tensor where p is the number of data channels, and l, m, and n are the dimensions of each channel.
    '''
    
    np.random.seed(random_seed_number)
    percentage = percentage/100
    # Reads files and stacks them in the first dimension
    input_torch = []

    for i in range(len(input_filenames)):
        filename = os.path.join(input_path, input_filenames[i])
        tensor = convert_mat_to_tensor(filename, input_filenames[i])
        # if rescale:
        #     tensor = rescale_normalize(tensor)
        noise = torch.randn_like(tensor) * torch.std(tensor) * percentage
        noisy_tensor = tensor + noise
        input_torch.append(noisy_tensor)

    variable_stack = torch.stack(
        tuple(input_torch[i] for i in range(len(input_torch))), axis=0)
    
    if rescale:
        for i in range(len(input_torch)):
            variable_stack[i, :, :, :] = rescale_normalize(
                variable_stack[i, :, :, :])
            
    if reshape_into_batch:
        batch_size = torch.numel(variable_stack[0, :])
        stack_batch = variable_stack[0, :].reshape(batch_size, 1, 1)
        for i in range(1, variable_stack.shape[0]):
            stack_batch = torch.cat(
                (stack_batch, variable_stack[i, :].reshape(batch_size, 1, 1)), axis=1)
        return stack_batch
    return variable_stack

def compute_coeff_matrix(ymxx, ymyy, prxy, pryz, gxy):
    '''
    Computes the 6*6 coefficient matrix given input ymxx, ymyy, prxy, pryz, and gxy values.
    
    Parameters:
    - ymxx (float): Young's modulus in the x direction.
    - ymyy (float): Young's modulus in the y direction.
    - prxy (float): Poisson's ratio in the xy plane.
    - pryz (float): Poisson's ratio in the yz plane.
    - gxy (float): Shear modulus in the xy plane.

    Returns:
    torch tensor: A 6-by-6 coefficient matrix.
    '''

    coeff_11 = (ymxx**2*(pryz - 1))/(2*ymyy*prxy**2 - ymxx + pryz*ymxx)
    coeff_12 = -(prxy*ymxx*ymyy)/(2*ymyy*prxy**2 - ymxx + pryz*ymxx)
    coeff_22 = -(ymyy*(- ymyy*prxy**2 + ymxx)) / \
        ((pryz + 1)*(2*ymyy*prxy**2 - ymxx + pryz*ymxx))
    coeff_23 = -(ymyy*(ymyy*prxy**2 + pryz*ymxx)) / \
        ((pryz + 1)*(2*ymyy*prxy**2 - ymxx + pryz*ymxx))
    coeff__44 = gxy
    coeff__55 = ymyy/(2*(pryz + 1))
    coeff__66 = gxy

    coeff_matrix = torch.tensor([[coeff_11, coeff_12, coeff_12, 0, 0, 0],
                                 [coeff_12, coeff_22, coeff_23, 0, 0, 0],
                                 [coeff_12, coeff_23, coeff_22, 0, 0, 0],
                                 [0, 0, 0, coeff__44, 0, 0],
                                 [0, 0, 0, 0, coeff__55, 0],
                                 [0, 0, 0, 0, 0, coeff__66]])
    return coeff_matrix

# Define necessary classes
class GlobalCoefficient:
    "Computation for the global coefficient matrix based on the Roe rotation convention and the transversely isotropic material model"
    def __init__(self, batch_size):
        # Initialize by inputting the batchsize, which is basically equal torch.numel(volume) for the full batch
        self.batch_size = batch_size
        # Create unit vectors in x, y, and z direction
        self.xdir = torch.tensor([1, 0, 0]).repeat(
            batch_size, 1).float().to(device)
        self.ydir = torch.tensor([0, 1, 0]).repeat(
            batch_size, 1).float().to(device)
        self.zdir = torch.tensor([0, 0, 1]).repeat(
            batch_size, 1).float().to(device)

    def compute_coefficient_matrix(self, zeta11_out, zeta22_out, zeta12_out, zeta23_out, muxy_out, reshape_instead_of_view=False):
        '''
        Compute the global coefficient matrix at each location in the domain based on the Roe rotation convention

        Parameters:
        - zeta11_out (torch tensor): Volumetric zeta11 material parameter reshaped into the batch dimension.
        - zeta22_out (torch tensor): Volumetric zeta22 material parameter reshaped into the batch dimension.
        - zeta12_out (torch tensor): Volumetric zeta12 material parameter reshaped into the batch dimension.
        - zeta23_out (torch tensor): Volumetric zeta23 material parameter reshaped into the batch dimension.
        - muxy_out (torch tensor): Volumetric muxy material parameter reshaped into the batch dimension.
        - reshape_instead_of_view (bool): Set to True if the input tensors need to be reshaped instead of viewed (Defaulted to False).

        Returns:
        torch tensor: batchsize-by-6-by-6 tensor.
        '''

        # Construct coefficient matrix from material parameters
        batch_size = self.batch_size

        # Construct the elements of the rotation matrix--sin(angle), cos(angle), zero, and one)--and reshaping to a shape of [batch_size, 1, 1]

        if reshape_instead_of_view:
            zeta11_out = zeta11_out.reshape(batch_size, 1).squeeze()
            zeta22_out = zeta22_out.reshape(batch_size, 1).squeeze()
            zeta12_out = zeta12_out.reshape(batch_size, 1).squeeze()
            zeta23_out = zeta23_out.reshape(batch_size, 1).squeeze()
            muxy_out = muxy_out.reshape(batch_size, 1).squeeze()
            muyz_out = (zeta22_out-zeta23_out)/2
        else:
            zeta11_out = zeta11_out.view(batch_size, 1).squeeze()
            zeta22_out = zeta22_out.view(batch_size, 1).squeeze()
            zeta12_out = zeta12_out.view(batch_size, 1).squeeze()
            zeta23_out = zeta23_out.view(batch_size, 1).squeeze()
            muxy_out = muxy_out.view(batch_size, 1).squeeze()
            muyz_out = (zeta22_out-zeta23_out)/2

        c_matrix_prime = torch.zeros(self.batch_size, 6, 6).float().to(device)
        c_matrix_prime[:, 0, 0] = zeta11_out
        c_matrix_prime[:, 0, 1] = zeta12_out
        c_matrix_prime[:, 0, 2] = zeta12_out
        c_matrix_prime[:, 1, 0] = zeta12_out
        c_matrix_prime[:, 1, 1] = zeta22_out
        c_matrix_prime[:, 1, 2] = zeta23_out
        c_matrix_prime[:, 2, 0] = zeta12_out
        c_matrix_prime[:, 2, 1] = zeta23_out
        c_matrix_prime[:, 2, 2] = zeta22_out
        c_matrix_prime[:, 3, 3] = muxy_out
        c_matrix_prime[:, 4, 4] = muyz_out
        c_matrix_prime[:, 5, 5] = muxy_out

        return c_matrix_prime

    def construct_yprime_matrix(self, theta, reshape_instead_of_view=False):
        '''
        Compute yprime_rotation matrix

        Parameters:
        - theta (torch tensor): l-by-m-by-n volumetric matrix containing the theta angle at each voxel.
        - reshape_instead_of_view (bool): Set to True if the input tensors need to be reshaped instead of viewed (Defaulted to False).

        Returns:
        torch tensor: A batchsize-by-3-by-3 rotation matrix for each voxel in the domain
        '''

        # Compute yprime_rotation matrix
        # Input arg:
        # theta (l-by-m-by-n volumetric matrix containing the theta angle at each voxel)
        batch_size = self.batch_size

        # Construct the elements of the rotation matrix--sin(angle), cos(angle), zero, and one)--and reshaping to a shape of [batch_size, 1, 1]
        if reshape_instead_of_view:
            cosine_batch = torch.cos(theta.reshape(batch_size, 1, 1))
            sine_batch = torch.sin(theta.reshape(batch_size, 1, 1))
        else:
            cosine_batch = torch.cos(theta.view(batch_size, 1, 1))
            sine_batch = torch.sin(theta.view(batch_size, 1, 1))
        zeros_batch = torch.zeros_like(cosine_batch)
        ones_batch = torch.ones_like(cosine_batch)
        # Construct the rows of the rotation matrix by:
        # first concatenating respective elements along the width dimension (dim=2),
        # followed by concatenating the rows along the height dimension (dim =1)
        # Note that dim = 0 is the batch dimension
        row_1 = torch.concat((cosine_batch, zeros_batch, sine_batch), dim=2)
        row_2 = torch.concat((zeros_batch, ones_batch, zeros_batch), dim=2)
        row_3 = torch.concat((-sine_batch, zeros_batch, cosine_batch), dim=2)
        rot_matrix_yprime = torch.concat((row_1, row_2, row_3), dim=1)

        return rot_matrix_yprime

    def construct_z_matrix(self, psi, reshape_instead_of_view=False):
        '''
        Compute z_rotation matrix

        Parameters:
        - psi (torch tensor): l-by-m-by-n volumetric matrix containing the psi angle at each voxel.
        - reshape_instead_of_view (bool): Set to True if the input tensors need to be reshaped instead of viewed (Defaulted to False).

        Returns:
        torch tensor: A batchsize-by-3-by-3 rotation matrix for each voxel in the domain
        '''
        batch_size = self.batch_size

        # Construct the elements of the rotation matrix--sin(angle), cos(angle), zero, and one)--and reshaping to a shape of [batch_size, 1, 1]

        if reshape_instead_of_view:
            cosine_batch = torch.cos(psi.reshape(batch_size, 1, 1))
            sine_batch = torch.sin(psi.reshape(batch_size, 1, 1))
        else:
            cosine_batch = torch.cos(psi.view(batch_size, 1, 1))
            sine_batch = torch.sin(psi.view(batch_size, 1, 1))

        zeros_batch = torch.zeros_like(cosine_batch)
        ones_batch = torch.ones_like(cosine_batch)
        # Construct the rows of the rotation matrix by:
        # first concatenating respective elements along the width dimension (dim=2),
        # followed by concatenating the rows along the height dimension (dim =1)
        # Note that dim = 0 is the batch dimension
        row_1 = torch.concat((cosine_batch, -sine_batch, zeros_batch), dim=2)
        row_2 = torch.concat((sine_batch, cosine_batch, zeros_batch), dim=2)
        row_3 = torch.concat((zeros_batch, zeros_batch, ones_batch), dim=2)
        rot_matrix_z = torch.concat((row_1, row_2, row_3), dim=1)
        return rot_matrix_z

    def construct_zsecond_matrix(self, phi, reshape_instead_of_view=False):
        '''
        Compute zsecond_rotation matrix

        Parameters:
        - phi (torch tensor): l-by-m-by-n volumetric matrix containing the phi angle at each voxel.
        - reshape_instead_of_view (bool): Set to True if the input tensors need to be reshaped instead of viewed (Defaulted to False).
        
        Returns:
        torch tensor: A batchsize-by-3-by-3 rotation matrix for each voxel in the domain
        '''

        # Compute zsecond_rotation matrix
        # Input arg:
        # phi (l-by-m-by-n volumetric matrix containing the phi angle at each voxel)
        batch_size = self.batch_size

        # Construct the elements of the rotation matrix--sin(angle), cos(angle), zero, and one)--and reshaping to a shape of [batch_size, 1, 1]
        if reshape_instead_of_view:
            cosine_batch = torch.cos(phi.reshape(batch_size, 1, 1))
            sine_batch = torch.sin(phi.reshape(batch_size, 1, 1))
        else:
            cosine_batch = torch.cos(phi.view(batch_size, 1, 1))
            sine_batch = torch.sin(phi.view(batch_size, 1, 1))

        zeros_batch = torch.zeros_like(cosine_batch)
        ones_batch = torch.ones_like(cosine_batch)
        # Construct the rows of the rotation matrix by:
        # first concatenating respective elements along the width dimension (dim=2),
        # followed by concatenating the rows along the height dimension (dim =1)
        # Note that dim = 0 is the batch dimension
        row_1 = torch.concat((cosine_batch, -sine_batch, zeros_batch), dim=2)
        row_2 = torch.concat((sine_batch, cosine_batch, zeros_batch), dim=2)
        row_3 = torch.concat((zeros_batch, zeros_batch, ones_batch), dim=2)
        rot_matrix_zsecond = torch.concat((row_1, row_2, row_3), dim=1)
        return rot_matrix_zsecond

    def compute_angle_cosine(self, a, b):
        '''
        Compute the cosine of angle between two vectors:

        Parameters:
        - a (torch tensor): Vector a containing three elements per voxel reshaped into size [batch_size,3].
        - b (torch tensor): Vector ba containing three elements per voxel reshaped into size [batch_size,3].

        Returns:
        torch tensor: Cosine of the angle between the two vectors.
        '''
        # Compute the cosine of angle between two vectors:
        # Input args
        # Vectors a and b, each with size [batch_size,3]

        return torch.linalg.vecdot(a, b)/(torch.linalg.vector_norm(a, dim=1)*torch.linalg.vector_norm(b, dim=1))

    def compute_rotated_unit_vectors(self, rot_matrix_z, rot_matrix_yprime, rot_matrix_zsecond):
        '''
        Compute the orientation of the coordinate system unit vectors across the entire domain

        Parameters:
        - rot_matrix_z (torch tensor): Rotation matrix around the z axis.
        - rot_matrix_yprime (torch tensor): Rotation matrix around the y' axis.
        - rot_matrix_zsecond (torch tensor): Rotation matrix around the z" axis.

        Returns:
        torch tensor: xdir_rotated, ydir_rotated, zdir_rotated
        '''

        rot_matrix = torch.matmul(torch.matmul(
            rot_matrix_z, rot_matrix_yprime), rot_matrix_zsecond)

        # To ensure a vectorized implementation of the matrix-vector multiplication, we have to add an extra dimension to the vectors (becoming [batchsize, 3, 1]) before using matmul
        xdir_rotated = torch.matmul(
            rot_matrix, self.xdir.unsqueeze(dim=-1)).squeeze()
        ydir_rotated = torch.matmul(
            rot_matrix, self.ydir.unsqueeze(dim=-1)).squeeze()
        zdir_rotated = torch.matmul(
            rot_matrix, self.zdir.unsqueeze(dim=-1)).squeeze()
        
        return xdir_rotated, ydir_rotated, zdir_rotated

    def compute_t_components(self, xdir, ydir, zdir, xdir_rotated, ydir_rotated, zdir_rotated):

        '''
        Compute the elements of the T rotation matrix (which is an alternative way to rotate, based on rotation of global coordinate system unit vectors)

        Parameters:
        - xdir (torch tensor): Global coordinate system x direction per voxel reshaped into size [batch_size,3]
        - ydir (torch tensor): Global coordinate system y direction per voxel reshaped into size [batch_size,3]
        - zdir (torch tensor): Global coordinate system z direction per voxel reshaped into size [batch_size,3]
        - xdir_rotated (torch tensor): Local coordinate system x direction per voxel reshaped into size [batch_size,3]
        - ydir_rotated (torch tensor): Local coordinate system y direction per voxel reshaped into size [batch_size,3]
        - zdir_rotated (torch tensor): Local coordinate system z direction per voxel reshaped into size [batch_size,3]

        Returns:
        torch tensor: t_11, t_12, t_13, t_21, t_22, t_23, t_31, t_32, t_33
        '''

        t_11 = self.compute_angle_cosine(xdir_rotated, xdir)
        t_12 = self.compute_angle_cosine(xdir_rotated, ydir)
        t_13 = self.compute_angle_cosine(xdir_rotated, zdir)
        t_21 = self.compute_angle_cosine(ydir_rotated, xdir)
        t_22 = self.compute_angle_cosine(ydir_rotated, ydir)
        t_23 = self.compute_angle_cosine(ydir_rotated, zdir)
        t_31 = self.compute_angle_cosine(zdir_rotated, xdir)
        t_32 = self.compute_angle_cosine(zdir_rotated, ydir)
        t_33 = self.compute_angle_cosine(zdir_rotated, zdir)
        return t_11, t_12, t_13, t_21, t_22, t_23, t_31, t_32, t_33

    def compute_B(self, t_11, t_12, t_13,
                  t_21, t_22, t_23,
                  t_31, t_32, t_33):
        '''
        Compute the 6-by-6 Bond transformation matrix using the elements of T matrix
        '''


        B = torch.zeros(self.batch_size, 6, 6).float().to(device)

        B[:, 0, 0] = t_11*t_11
        B[:, 0, 1] = t_12*t_12
        B[:, 0, 2] = t_13*t_13
        B[:, 0, 3] = t_11*t_12
        B[:, 0, 4] = t_12*t_13
        B[:, 0, 5] = t_13*t_11

        B[:, 1, 0] = t_21*t_21
        B[:, 1, 1] = t_22*t_22
        B[:, 1, 2] = t_23*t_23
        B[:, 1, 3] = t_21*t_22
        B[:, 1, 4] = t_22*t_23
        B[:, 1, 5] = t_23*t_21

        B[:, 2, 0] = t_31*t_31
        B[:, 2, 1] = t_32*t_32
        B[:, 2, 2] = t_33*t_33
        B[:, 2, 3] = t_31*t_32
        B[:, 2, 4] = t_32*t_33
        B[:, 2, 5] = t_33*t_31

        B[:, 3, 0] = 2*t_11*t_21
        B[:, 3, 1] = 2*t_12*t_22
        B[:, 3, 2] = 2*t_13*t_23
        B[:, 3, 3] = t_11*t_22+t_12*t_21
        B[:, 3, 4] = t_12*t_23+t_13*t_22
        B[:, 3, 5] = t_13*t_21 + t_11*t_23

        B[:, 4, 0] = 2*t_21*t_31
        B[:, 4, 1] = 2*t_22*t_32
        B[:, 4, 2] = 2*t_23*t_33
        B[:, 4, 3] = t_21*t_32+t_22*t_31
        B[:, 4, 4] = t_22*t_33+t_23*t_32
        B[:, 4, 5] = t_23*t_31 + t_21*t_33

        B[:, 5, 0] = 2*t_31*t_11
        B[:, 5, 1] = 2*t_32*t_12
        B[:, 5, 2] = 2*t_33*t_13
        B[:, 5, 3] = t_31*t_12+t_32*t_11
        B[:, 5, 4] = t_32*t_13+t_33*t_12
        B[:, 5, 5] = t_33*t_11 + t_31*t_13

        return B

    def forward_xdir_rotated(self, theta, phi, psi, reshape_instead_of_view=False):
        '''
        Compute the final direction of the x-axis after the rotation

        Parameters:
        - theta (torch tensor): Rotation around y' axis.
        - phi (torch tensor): Rotation around z" axis.
        - psi (torch tensor): Rotation around z axis.
        - reshape_instead_of_view (bool): Set to True if the input tensors need to be reshaped instead of viewed (Defaulted to False).

        Returns:
        torch tensor: x-axis vector direction after the rotation for all voxels in the domain
        '''

        rot_matrix_z = self.construct_z_matrix(
            psi, reshape_instead_of_view=reshape_instead_of_view)
        rot_matrix_yprime = self.construct_yprime_matrix(
            theta, reshape_instead_of_view=reshape_instead_of_view)
        rot_matrix_zsecond = self.construct_zsecond_matrix(
            phi, reshape_instead_of_view=reshape_instead_of_view)
        xdir_rotated, _, _ = self.compute_rotated_unit_vectors(
            rot_matrix_z, rot_matrix_yprime, rot_matrix_zsecond)
        return xdir_rotated

    def forward(self, theta, phi, psi, zeta11_out, zeta22_out, zeta12_out, zeta23_out, muxy_out, reshape_instead_of_view=False):
        '''

        Compute the global coefficient matrix over the entire in the domain based on the Roe rotation convention

        Parameters:
        - theta (torch tensor): Rotation around y' axis.
        - phi (torch tensor): Rotation around z" axis.
        - psi (torch tensor): Rotation around z axis.
        - zeta11_out (torch tensor): Volumetric zeta11 material parameter reshaped into the batch dimension.
        - zeta22_out (torch tensor): Volumetric zeta22 material parameter reshaped into the batch dimension.
        - zeta12_out (torch tensor): Volumetric zeta12 material parameter reshaped into the batch dimension.
        - zeta23_out (torch tensor): Volumetric zeta23 material parameter reshaped into the batch dimension.
        - muxy_out (torch tensor): Volumetric muxy material parameter reshaped into the batch dimension.
        - reshape_instead_of_view (bool): Set to True if the input tensors need to be reshaped instead of viewed (Defaulted to False).

        Returns:
        torch tensor: Global coefficient matrix over the entire in the domain


        '''

        c_matrix_prime = self.compute_coefficient_matrix(
            zeta11_out, zeta22_out, zeta12_out, zeta23_out, muxy_out, reshape_instead_of_view=reshape_instead_of_view)
        rot_matrix_z = self.construct_z_matrix(
            psi, reshape_instead_of_view=reshape_instead_of_view)
        rot_matrix_yprime = self.construct_yprime_matrix(
            theta, reshape_instead_of_view=reshape_instead_of_view)
        rot_matrix_zsecond = self.construct_zsecond_matrix(
            phi, reshape_instead_of_view=reshape_instead_of_view)
        xdir_rotated, ydir_rotated, zdir_rotated = self.compute_rotated_unit_vectors(
            rot_matrix_z, rot_matrix_yprime, rot_matrix_zsecond)
        t_11, t_12, t_13, t_21, t_22, t_23, t_31, t_32, t_33 = self.compute_t_components(self.xdir, self.ydir, self.zdir,
                                                                                         xdir_rotated, ydir_rotated, zdir_rotated)
        B = self.compute_B(t_11, t_12, t_13,
                           t_21, t_22, t_23,
                           t_31, t_32, t_33)
        c = torch.matmul(torch.matmul(
            torch.transpose(B, 1, 2), c_matrix_prime), B)
        return c, xdir_rotated

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

class ReshapeConv(nn.Module):
    # Reshape to image space
    def __init__(self, in_channels, out_channels):
        super(ReshapeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
  # class that contains the upsampling convolution at each step
    def __init__(self, feature, scale_factor):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(feature*2, feature, 2, 1, padding='same', bias=False)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    # UNET class containing the downward, bottleneck, upward, and final layer of the structure
    # The features are the number of convolution filters in the order of the down path
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128]):
        super(UNET, self).__init__()
        # The convolutions lists should be stored in the ModuleList
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Pooling layer downsamples the image by half
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            # As we start the up path, we need to upsample each image (ConvTranspose2d with same kernel_size and stride as in the pooling layer)
            # and also divide the number of channels by half so that we can concat the output with the skip_connections later
            self.ups.append(UpSample(feature=feature, scale_factor=2))
            # the result of the concatenation of previous output with the skip connections (hence the input size of feature*2) goes throught the DoubleConv filters
            self.ups.append(DoubleConv(feature*2, feature))

        # At the bottleneck level the convolution filters are double the last item in features
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # the last layer simply brings the multi-channel space back to the image space with a conv kernel of 1
        self.final_conv = ReshapeConv(features[0], out_channels)

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

class DoubleUNET(nn.Module):
    # DoubleUNET class containing two UNETs one for estimation of material parameters and the other for stress
    def __init__(self, in_channels_first=6, in_channels_second=6, out_channels_first=5, out_channels_second=6, features=[64, 128]):
        super(DoubleUNET, self).__init__()
        self.unet1 = UNET(in_channels=in_channels_first,
                          out_channels=out_channels_first, features=features)
        self.unet2 = UNET(in_channels=in_channels_second,
                          out_channels=out_channels_second, features=features)

    def forward(self, x, y):
        param_output = self.unet1(x)
        stress_output = self.unet2(y)
        return [param_output, stress_output]

# Define helper functions for training and visualization
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model,state, filename):
    print("=> Loading checkpoint")
    torch.load(state, filename)
    model.load_state_dict(checkpoint["state_dict"])

def compute_loss(y,  criterion, coefficient_instance,
                 strain_stack, stress_boundary, sigma0,
                 angles):
    '''
    Compute the loss for the model for given input and output

    Parameters:
    - y (torch tensor): Output of the model
    - criterion (torch.nn): Loss function
    - coefficient_instance (GlobalCoefficient): Instance of the GlobalCoefficient class
    - strain_stack (torch tensor): 1-by-6-by-3-by-3 strain matrix
    - stress_boundary (torch tensor): Stress boundary conditions for given load
    - sigma0 (float): Normalizing factor for stress

    Returns:
    torch tensor: Loss value

    '''

    zeta11_out = y[0][0, 0, :]
    zeta22_out = y[0][0, 1, :]
    zeta12_out = y[0][0, 2, :]
    zeta23_out = y[0][0, 3, :]
    muxy_out = y[0][0, 4, :]

    phi_gt = torch.zeros_like(angles['psi_gt'])

    c, _ = coefficient_instance.forward(angles['theta_gt'], phi_gt, angles['psi_gt'],
                                        zeta11_out, zeta22_out, zeta12_out, zeta23_out, muxy_out,
                                        reshape_instead_of_view=True)

    s_matrix = torch.matmul(c, strain_stack)
    
    # Create a tensor of zeros with the same size as the 3 dimensions of the geometry
    zeros_tensor = torch.zeros_like(angles['theta_gt']).to(device)

    sxx_out = y[1][0, 0, :, :]
    syy_out = y[1][0, 1, :, :]
    szz_out = y[1][0, 2, :, :]
    sxy_out = y[1][0, 3, :, :]
    syz_out = y[1][0, 4, :, :]
    sxz_out = y[1][0, 5, :, :]

    sxx_out_from_param = s_matrix[:, 0].view(zeros_tensor.shape)
    syy_out_from_param = s_matrix[:, 1].view(zeros_tensor.shape)
    szz_out_from_param = s_matrix[:, 2].view(zeros_tensor.shape)
    sxy_out_from_param = s_matrix[:, 3].view(zeros_tensor.shape)
    syz_out_from_param = s_matrix[:, 4].view(zeros_tensor.shape)
    sxz_out_from_param = s_matrix[:, 5].view(zeros_tensor.shape)

    sxx_bound_out = torch.stack((sxx_out[:, 0, :], sxx_out[:, -1, :]), dim=0)
    szz_bound_out = torch.stack((szz_out[:, :, 0], szz_out[:, :, -1]), dim=0)
    syy_bound_out = torch.stack((syy_out[0, :, :], syy_out[-1, :, :]), dim=0)

    sxx_bound_gt = stress_boundary['sxx_bound_gt']/sigma0
    szz_bound_gt = stress_boundary['szz_bound_gt']/sigma0
    syy_bound_gt = stress_boundary['syy_bound_gt']/sigma0

    loss_bound_sxx = criterion(sxx_bound_out, sxx_bound_gt)
    loss_bound_szz = criterion(szz_bound_out, szz_bound_gt)
    loss_bound_syy = criterion(syy_bound_out, syy_bound_gt)

    spacing = 1 / \
    (((zeros_tensor.shape[0]-1) +
        (zeros_tensor.shape[1]-1)+(zeros_tensor.shape[2]-1))/3)

    _, sxx_x_out, _ = torch.gradient(sxx_out, spacing = spacing)
    syy_y_out, _, _ = torch.gradient(syy_out, spacing = spacing)
    _, _, szz_z_out = torch.gradient(szz_out, spacing = spacing)
    sxy_y_out, sxy_x_out, _ = torch.gradient(sxy_out, spacing = spacing)
    syz_y_out, _, syz_z_out = torch.gradient(syz_out, spacing = spacing)
    _, sxz_x_out, sxz_z_out = torch.gradient(sxz_out, spacing = spacing)

    loss_equib_x = criterion(sxx_x_out + sxy_y_out + sxz_z_out, zeros_tensor)
    loss_equib_y = criterion(sxy_x_out+syy_y_out+syz_z_out, zeros_tensor)
    loss_equib_z = criterion(sxz_x_out+syz_y_out+szz_z_out, zeros_tensor)

    loss_stress = criterion(sxx_out_from_param, sxx_out) + \
        criterion(syy_out_from_param, syy_out) + \
        criterion(szz_out_from_param, szz_out) + \
        criterion(sxy_out_from_param, sxy_out) + \
        criterion(syz_out_from_param, syz_out) + \
        criterion(sxz_out_from_param, sxz_out)
    

    loss = loss_bound_sxx + loss_bound_syy + loss_bound_szz + \
        loss_equib_x + loss_equib_y + loss_equib_z + loss_stress

    return loss

def compute_error_metrics(y, parameters, sigma0):
    '''
    Compute the error metrics for the model for given input and output

    Parameters:
    - y (torch tensor): Output of the model
    - parameters (dict): Dictionary containing the ground truth values for the material parameters
    - sigma0 (float): Normalizing factor for stress

    Returns:
    dict: Dictionary containing the error metrics
    '''
    zeta11_out = y[0][0, 0, :, :]
    zeta22_out = y[0][0, 1, :, :]
    zeta12_out = y[0][0, 2, :, :]
    zeta23_out = y[0][0, 3, :, :]
    muxy_out = y[0][0, 4, :, :]
    muyz_out = (zeta22_out - zeta23_out) / 2

    YMxx_pred = (-2 * zeta12_out ** 2 + zeta11_out * zeta22_out +
                    zeta11_out * zeta23_out) / (zeta22_out + zeta23_out)
    YMyy_pred = ((zeta22_out - zeta23_out) * (-2 * zeta12_out ** 2 + zeta11_out *
                                                zeta22_out + zeta11_out * zeta23_out)) / (-zeta12_out ** 2 + zeta11_out * zeta22_out)
    PRxy_pred = zeta12_out / (zeta22_out + zeta23_out)
    PRyz_pred = (-zeta12_out ** 2 + zeta11_out * zeta23_out) / \
        (-zeta12_out ** 2 + zeta11_out * zeta22_out)

    error_ymxx = 100 * torch.abs(YMxx_pred * sigma0 - parameters['ymxx_gt']) / parameters['ymxx_gt']
    error_ymxx = torch.mean(error_ymxx).item()

    error_ymyy = 100 * torch.abs(YMyy_pred * sigma0 - parameters['ymyy_gt']) / parameters['ymyy_gt']
    error_ymyy = torch.mean(error_ymyy).item()

    error_gxy = 100 * torch.abs(muxy_out * sigma0 - parameters['gxy_gt']) / parameters['gxy_gt']
    error_gxy = torch.mean(error_gxy).item()

    error_gyz = 100 * torch.abs(muyz_out * sigma0 - parameters['gyz_gt']) / parameters['gyz_gt']
    error_gyz = torch.mean(error_gyz).item()

    error_prxy = 100 * torch.abs(PRxy_pred - parameters['prxy_gt']) / parameters['prxy_gt']
    error_prxy = torch.mean(error_prxy).item()

    error_pryz = 100 * torch.abs(PRyz_pred - parameters['pryz_gt']) / parameters['pryz_gt']
    error_pryz = torch.mean(error_pryz).item()

    error_dict = {
        'error_ymxx': error_ymxx,
        'error_ymyy': error_ymyy,
        'error_gxy': error_gxy,
        'error_gyz': error_gyz,
        'error_prxy': error_prxy,
        'error_pryz': error_pryz
    } 

    return error_dict

def train_running_time(
        model, criterion, optimizer, scheduler,
        strain_dict,
        stress_boundary_dict,
        sigma0,
        angles, parameters,
        batch_size, training_duration_max,
        report_freq=100, save_freq=5000):
    '''
    Train the model for a given number of epochs

    Parameters:
    - model (torch.nn.Module): Model to be trained
    - strain_dict (dict): Dictionary containing the strain data
    - stress_boundary_dict (dict): Dictionary containing the stress boundary data
    - sigma0 (float): Normalizing factor for stress
    - angles (dict): Dictionary containing the phi, psi, and theta angles
    - parameters (dict): Dictionary containing the ground truth values for the material parameters
    - batch_size (int): Batch size for training (equal to number of voxels in the volume)
    - training_duration_max (int): Maximum training duration in minutes
    - learning_rate (float): Learning rate for the optimizer
    - report_freq (int): Frequency of reporting the training progress (Defaulted to 100)
    - save_freq (int): Frequency of saving the model (Defaulted to 5000)

    Returns:
    torch.nn.Module: Trained model
    dict: Dictionary containing the loss histories
    '''


    loss_histories = {}
    loss_histories['running_loss_history'] = []
    loss_histories['ymxx_mae_history'] = []
    loss_histories['ymyy_mae_history'] = []
    loss_histories['gxy_mae_history'] = []
    loss_histories['gyz_mae_history'] = []
    loss_histories['prxy_mae_history'] = []
    loss_histories['pryz_mae_history'] = []

    coefficient_instance = GlobalCoefficient(batch_size)
    training_duration = 0
    start = time.time()
    e = 0
    while training_duration < training_duration_max:

        y = model.forward(strain_dict['mean_strain_stack_normalized'].float(),strain_dict['strain_stack_normalized_xloading'].float())

        loss_tensile_xloading = compute_loss(y, criterion, coefficient_instance,
                                    strain_dict['strain_stack_xloading'], stress_boundary_dict['stress_boundaries_xloading'], sigma0,
                                    angles)
        
        y = model.forward(strain_dict['mean_strain_stack_normalized'].float(),strain_dict['strain_stack_normalized_yloading'].float())

        loss_tensile_yloading = compute_loss(y, criterion, coefficient_instance,
                                              strain_dict['strain_stack_yloading'], stress_boundary_dict['stress_boundaries_yloading'], sigma0,
                                              angles)
        
        y = model.forward(strain_dict['mean_strain_stack_normalized'].float(),strain_dict['strain_stack_normalized_zloading'].float())

        loss_tensile_zloading = compute_loss(y, criterion, coefficient_instance,
                                              strain_dict['strain_stack_zloading'], stress_boundary_dict['stress_boundaries_zloading'], sigma0,
                                              angles)
        
        optimizer.zero_grad()
        loss = (loss_tensile_xloading + loss_tensile_yloading + loss_tensile_zloading)/3
        loss.backward()
        optimizer.step()

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        with torch.no_grad():
            error_dict = compute_error_metrics(y, parameters, sigma0)

            loss_histories['ymxx_mae_history'].append(error_dict['error_ymxx'])
            loss_histories['ymyy_mae_history'].append(error_dict['error_ymyy'])
            loss_histories['gxy_mae_history'].append(error_dict['error_gxy'])
            loss_histories['gyz_mae_history'].append(error_dict['error_gyz'])
            loss_histories['prxy_mae_history'].append(error_dict['error_prxy'])
            loss_histories['pryz_mae_history'].append(error_dict['error_pryz'])

        if e % report_freq == 0 or e == 0:

            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")

            print(
                'ymxx_mae: ', "{:.2f}  ".format(error_dict['error_ymxx']),
                'ymyy_mae: ', "{:.2f}  ".format(error_dict['error_ymyy']))
            
            print(
                'gxy_mae: ', "{:.2f}  ".format(error_dict['error_gxy']),
                'gyz_mae: ', "{:.2f}  ".format(error_dict['error_gyz']))
            
            print(
                'prxy_mae: ', "{:.2f}  ".format(error_dict['error_prxy']),
                'pryz_mae: ', "{:.2f}  ".format(error_dict['error_pryz']))

        # the running loss for every epoch is the average of loss associated with the loss from all the iterations from batches in that epoch
        loss_histories['running_loss_history'].append(loss.item())

        scheduler.step(loss)
        if e % save_freq == 0 and not e == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint, filename=f"Temp_checkpoint_at_{e}_epochs.pth.tar")
            plt.plot(
                loss_histories['running_loss_history'], label='training loss')
            plt.yscale('log')
            plt.savefig('Loss_history_at_{e}_epochs.png')
        e += 1
        end = time.time()
        training_duration = (end - start)/60
    print(f'Training took {end-start: .2f}, seconds')
    return model, loss_histories

def train_epoch_number(
        model, criterion, optimizer, scheduler,
        strain_dict,
        stress_boundary_dict,
        sigma0,
        angles, parameters,
        batch_size, max_epochs,
        report_freq=100, save_freq=5000):
    '''
    Train the model for a given number of epochs

    Parameters:
    - model (torch.nn.Module): Model to be trained
    - strain_dict (dict): Dictionary containing the strain data
    - stress_boundary_dict (dict): Dictionary containing the stress boundary data
    - sigma0 (float): Normalizing factor for stress
    - angles (dict): Dictionary containing the phi, psi, and theta angles
    - parameters (dict): Dictionary containing the ground truth values for the material parameters
    - batch_size (int): Batch size for training (equal to number of voxels in the volume)
    - max_epochs (int): Maximum number of epochs for training
    - learning_rate (float): Learning rate for the optimizer
    - report_freq (int): Frequency of reporting the training progress (Defaulted to 100)
    - save_freq (int): Frequency of saving the model (Defaulted to 5000)

    Returns:
    torch.nn.Module: Trained model
    dict: Dictionary containing the loss histories
    '''

    loss_histories = {}
    loss_histories['running_loss_history'] = []
    loss_histories['ymxx_mae_history'] = []
    loss_histories['ymyy_mae_history'] = []
    loss_histories['gxy_mae_history'] = []
    loss_histories['gyz_mae_history'] = []
    loss_histories['prxy_mae_history'] = []
    loss_histories['pryz_mae_history'] = []


    coefficient_instance = GlobalCoefficient(batch_size)
    training_duration = 0
    start = time.time()
    for e in range(max_epochs):
        y = model.forward(strain_dict['mean_strain_stack_normalized'].float(),strain_dict['strain_stack_normalized_xloading'].float())

        loss_tensile_xloading = compute_loss(y, criterion, coefficient_instance,
                                    strain_dict['strain_stack_xloading'], stress_boundary_dict['stress_boundaries_xloading'], sigma0,
                                    angles)
        
        y = model.forward(strain_dict['mean_strain_stack_normalized'].float(),strain_dict['strain_stack_normalized_yloading'].float())

        loss_tensile_yloading = compute_loss(y, criterion, coefficient_instance,
                                              strain_dict['strain_stack_yloading'], stress_boundary_dict['stress_boundaries_yloading'], sigma0,
                                              angles)
        
        y = model.forward(strain_dict['mean_strain_stack_normalized'].float(),strain_dict['strain_stack_normalized_zloading'].float())

        loss_tensile_zloading = compute_loss(y, criterion, coefficient_instance,
                                              strain_dict['strain_stack_zloading'], stress_boundary_dict['stress_boundaries_zloading'], sigma0,
                                              angles)
        
        optimizer.zero_grad()
        loss = (loss_tensile_xloading + loss_tensile_yloading + loss_tensile_zloading)/3
        loss.backward()
        optimizer.step()

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        with torch.no_grad():
            error_dict = compute_error_metrics(y, parameters, sigma0)

            loss_histories['ymxx_mae_history'].append(error_dict['error_ymxx'])
            loss_histories['ymyy_mae_history'].append(error_dict['error_ymyy'])
            loss_histories['gxy_mae_history'].append(error_dict['error_gxy'])
            loss_histories['gyz_mae_history'].append(error_dict['error_gyz'])
            loss_histories['prxy_mae_history'].append(error_dict['error_prxy'])
            loss_histories['pryz_mae_history'].append(error_dict['error_pryz'])

        if e % report_freq == 0 or e == 0:

            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")

            print(
                'ymxx_mae: ', "{:.2f}  ".format(error_dict['error_ymxx']),
                'ymyy_mae: ', "{:.2f}  ".format(error_dict['error_ymyy']))
            
            print(
                'gxy_mae: ', "{:.2f}  ".format(error_dict['error_gxy']),
                'gyz_mae: ', "{:.2f}  ".format(error_dict['error_gyz']))
            
            print(
                'prxy_mae: ', "{:.2f}  ".format(error_dict['error_prxy']),
                'pryz_mae: ', "{:.2f}  ".format(error_dict['error_pryz']))
            
        # the running loss for every epoch is the average of loss associated with the loss from all the iterations from batches in that epoch
        loss_histories['running_loss_history'].append(loss.item())

        scheduler.step(loss)
        if e % save_freq == 0 and not e == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint, filename=f"my_checkpoint_UNet_tmp_TwoOutput.pth.tar")
            
        end = time.time()
        training_duration = (end - start)/60
    
    print(f'Training took {end-start: .2f}, seconds')
    return model, loss_histories