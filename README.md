# Probabilistic 3D U-Net 
Resources shared as part of the paper - Probabilistic 3D segmentation for aleatoric uncertainty quantification in full 3D medical data.
 [SPIE paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12465/2654255/Probabilistic-3D-segmentation-for-aleatoric-uncertainty-quantification-in-full-3D/10.1117/12.2654255.short?SSO=1), Arxiv paper inc.

Network Architecture:
![alt text](https://github.com/cviviers/prob_3D_segmentation/blob/main/images/Prob3DUnet.PNG?raw=true)

Example predictions:
![alt text](https://github.com/cviviers/prob_3D_segmentation/blob/main/images/full_adjusted.png)

3D visuals:
![](https://github.com/cviviers/prob_3D_segmentation/blob/main/images/gifmovie_prediction.gif )

# Code
Since the repo is based on [https://github.com/wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), most of the code works the same way.

## Setup

install requirements or use the sudochris/3dunet:v3 [docker](https://hub.docker.com/repository/docker/sudochris/3dunet/general) container.

## Train

python train.py --config ./resources/probabilistic_3d_unet/train_config_vanilla_0.yaml
## Validate

python validate.py --config ./resources/probabilistic_3d_unet/val_config_vanilla_0.yaml
# Acknowledgement
The repository is based on the work by [https://github.com/wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), [Kohl et. al.](https://arxiv.org/abs/1806.05034), but adapted from [Valiuddin et. al.](https://arxiv.org/abs/2108.02155) and [PyTorch implemntation](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch).

