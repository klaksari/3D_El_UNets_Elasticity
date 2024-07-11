# Discovering 3D Hidden Elasticity in Isotropic and Transversely Isotropic Materials with Physics-informed UNets
Data and codes from our paper: 
[Discovering 3D Hidden Elasticity in Isotropic and Transversely Isotropic Materials with Physics-informed UNets](https://www.sciencedirect.com/science/article/pii/S1742706124003532)

We developed a physics-informed UNet-based neural network model (El-UNet) to discover the three-dimensional (3D) internal composition and space-dependent material properties of heterogeneous isotropic and transversely isotropic materials without a priori knowledge of the composition.
This repository currently contains data and sample code from our published paper.

![image](https://github.com/klaksari/3D_El_UNets_Elasticity/assets/60515966/bc6a99fa-11c5-41bd-baee-c168f703461b)

## Running the code
After cloning or downloading this repository, running the inverse script can be done with the desired training parameters.

Isotropic example (Brain directory):
```
!python3  brain_inverse_application.py --noise_percentage 5.0 --epochs 20000 --lr_update_threshold 10000 --loss_report_freq 1000 -pl --unet_num_channels 64 128  --parameter_and_stress --weighted_loss --input_path 'brain_input_domain_data' --output_path 'brain/output_brain_estimation_results'
```

Transversely isotropic example (Cartilage directory):
```
!python3  cartilage_inverse_application.py --noise_percentage 0.0 --epochs 20000 --lr_update_threshold 10000 --loss_report_freq 1000 --unet_num_channels 64 128 --input_path 'cartilage_input_domain_data' --output_path 'output_cartilage_estimation_results'
```

Have a question about implementing the code? contact us at [klaksari@engr.ucr.edu](mailto:klaksari@engr.ucr.edu), [akamali@arizona.edu](mailto:akamali@arizona.edu).
