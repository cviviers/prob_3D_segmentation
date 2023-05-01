import importlib

import torch.nn as nn
import torch
from torch.distributions import kl
from pytorch3dunet.unet3d.buildingblocks import DoubleConv, create_encoders, \
    create_decoders, Fcomb,  PriorNet, PosteriorNet, PosteriorNetWithNormalizingFlow
from pytorch3dunet.unet3d.utils import number_of_features_per_level
import pytorch3dunet.unet3d.flows as flows


def get_model(model_config):
    def _model_class(class_name):
        modules = ['pytorch3dunet.unet3d.model']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz

    model_class = _model_class(model_config['name'])
    return model_class(**model_config)




class AbstractProb3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        num_classes: the number of classes to predict
        latent_dim: dimension of the latent space
        no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
        no_convs_fcomb: number of layers in feature combination network
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, testing=False,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, latent_dim=6, no_convs_fcomb=4,  prior_layer_order ='gcr', 
                 posterior_layer_order = 'gcr', encoders_f_maps = 32, encoder_num_groups = 16, encoder_num_levels = 4,  flow_type = None, num_flow_steps = None, **kwargs):
        super(AbstractProb3DUNet, self).__init__()

        self.num_classes = out_channels
        self.testing = testing
        self.latent_dim = latent_dim
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}

        if flow_type == 'radial':
            self.flow_type = flows.Radial
        elif flow_type == 'planar':
            self.flow_type = flows.Planar
        else:
            self.flow_type = None

        self.num_flow_steps = num_flow_steps
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        if isinstance(encoders_f_maps, int):
            encoders_f_maps = number_of_features_per_level(encoders_f_maps, num_levels=encoder_num_levels)
        assert isinstance(encoders_f_maps, list) or isinstance(encoders_f_maps, tuple)
        assert len(encoders_f_maps) > 1, "Required at least 2 levels in the encoders for the prior/posterior"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample=True)

        # create prior path
        self.prior = PriorNet(in_channels, encoders_f_maps, basic_module, conv_kernel_size, conv_padding, prior_layer_order, encoder_num_groups, pool_kernel_size, self.latent_dim, posterior=False)

        if self.flow_type != None:
            # create posterior path with normalizing flow
            self.posterior = PosteriorNetWithNormalizingFlow(in_channels+out_channels, encoders_f_maps, basic_module, conv_kernel_size, conv_padding, posterior_layer_order, encoder_num_groups, pool_kernel_size, self.latent_dim, num_flow_steps=self.num_flow_steps, flow_type=self.flow_type, posterior=True)
        else:    
            # create posterior path
            self.posterior = PosteriorNet(in_channels+out_channels, encoders_f_maps, basic_module, conv_kernel_size, conv_padding, posterior_layer_order, encoder_num_groups, pool_kernel_size, self.latent_dim,  posterior=True)

        self.fcomb = Fcomb(f_maps, self.latent_dim, in_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, input, target=None):

        x = input

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        # Always calculate prior
        _, self.prior_latent_space = self.prior.forward(input)

        if target != None: # train or validation, grab posterior
            if self.flow_type != None:
                log_det_j_q, z0_q, z_q, posterior_latent_space = self.posterior.forward(input, target)
            else:
                log_det_j_q, z0_q, z_q = None, None, None
                _, posterior_latent_space = self.posterior.forward(input, target)

        self.unet_features = x

        if self.training:
            if self.flow_type != None:
                z_posterior = z_q
            else:
                z_posterior = posterior_latent_space.rsample()
            self.z_posterior = z_posterior

            # print(f"posterior_latent_space mean: {posterior_latent_space.mean}")
            # print(f"posterior_latent_space variance: {posterior_latent_space.variance}")
            # print(f"mean: {self.prior_latent_space.mean}")
            # print(f"variance: {self.prior_latent_space.variance}")

            #Here we use the posterior sample sampled above
            kl_div = self.kl_divergence(posterior_latent_space, self.prior_latent_space, log_det_j_q, z_q, z0_q)
            return self.fcomb.forward(x, z_posterior), kl_div

        elif self.testing:
            
            # print(f"posterior_latent_space mean: {posterior_latent_space.mean}")
            # print(f"posterior_latent_space variance: {posterior_latent_space.variance}")
            # print(f"mean: {self.prior_latent_space.mean}")
            # print(f"variance: {self.prior_latent_space.variance}")
            z_prior = self.prior_latent_space.sample()
            return self.final_activation(self.fcomb.forward(x, z_prior))

        elif not self.testing and not self.training: # Validation
            
            if self.num_flow_steps != None:
                z_posterior = z_q
            else:
                z_posterior = posterior_latent_space.rsample()
            self.z_posterior = z_posterior
            kl_div = self.kl_divergence(posterior_latent_space, self.prior_latent_space, log_det_j_q, z_q, z0_q)
            return self.fcomb.forward(x, z_posterior), kl_div

    def kl_divergence(self, posterior_latent_space, prior_latent_space, log_det_j_q=None, z_q=None, z0_q=None):

        if log_det_j_q == None and z_q == None and z0_q == None:
            kl_div = torch.sum(kl.kl_divergence(posterior_latent_space, prior_latent_space))
        else:

            log_posterior_prob = posterior_latent_space.log_prob(z0_q) 
            log_prior_prob = prior_latent_space.log_prob(z_q)

            kl_div = (log_posterior_prob - log_prior_prob) # *torch.exp(log_posterior_prob)
            kl_div -= log_det_j_q
            kl_div = kl_div.sum()
        return kl_div

    def sample(self):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if self.testing:
            z_prior = self.prior_latent_space.sample()
            # self.z_prior_sample = z_prior
            # print(f"z_prior: {z_prior}")
            return self.final_activation(self.fcomb.forward(self.unet_features, z_prior))
        else:
            return self.final_activation(self.fcomb.forward(self.unet_features, self.z_posterior.sample()))




class ProbUNet3D(AbstractProb3DUNet):

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, latent_dim = 6,
                                     prior_layer_order ='gcr', 
                                     posterior_layer_order = 'gcr',
                                     encoders_f_maps = 32,
                                     encoder_num_groups = 16,
                                     encoder_num_levels = 4,
                                     no_convs_fcomb = 6,
                                     **kwargs):
                 
        super(ProbUNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     latent_dim=latent_dim,
                                     prior_layer_order = prior_layer_order, 
                                     posterior_layer_order = posterior_layer_order,
                                     encoders_f_maps = encoders_f_maps,
                                     encoder_num_groups = encoder_num_groups,
                                     encoder_num_levels = encoder_num_levels,
                                     no_convs_fcomb = no_convs_fcomb,
                                     **kwargs)

class ProbUNet2D(AbstractProb3DUNet):
    """
    Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, latent_dim = 6,
                                     prior_layer_order ='gcr', 
                                     posterior_layer_order = 'gcr',
                                     encoders_f_maps = 32,
                                     encoder_num_groups = 16,
                                     encoder_num_levels = 4,
                                     no_convs_fcomb = 4,
                                     **kwargs):
        if conv_padding == 1:
            conv_padding = (0, 1, 1)         
        super(ProbUNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=(1, 3, 3),
                                     pool_kernel_size=(1, 2, 2),
                                     conv_padding=conv_padding,
                                     latent_dim=latent_dim,
                                     prior_layer_order = prior_layer_order, 
                                     posterior_layer_order = posterior_layer_order,
                                     encoders_f_maps = encoders_f_maps,
                                     encoder_num_groups = encoder_num_groups,
                                     encoder_num_levels = encoder_num_levels,
                                     no_convs_fcomb = no_convs_fcomb,
                                     **kwargs)
