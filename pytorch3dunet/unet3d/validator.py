import os

import h5py
import numpy as np
import torch
from skimage import measure

from pytorch3dunet.datasets.hdf5 import AbstractHDF5Dataset
from pytorch3dunet.datasets.utils import SliceBuilder
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.utils import remove_halo
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
import time
logger = get_logger('UNetValidator')


def _get_output_file(dataset, suffix='_predictions', output_dir=None):
    input_dir, file_name = os.path.split(dataset.file_path)
    if output_dir is None:
        output_dir = input_dir
    output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + suffix + '.h5')
    return output_file

class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class _AbstractValidator:
    def __init__(self, model, output_dir, config, **kwargs):
        self.model = model
        self.output_dir = output_dir
        self.config = config
        self.validator_config = kwargs

    @staticmethod
    def volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def __call__(self, val_loader):
        raise NotImplementedError


class StandardValidator(_AbstractValidator):
    """
    Applies the model on the given dataset and saves the result as H5 file.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyValidator` instead.
    The output dataset names inside the H5 is given by `dest_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.
    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        output_dir (str): path to the output directory (optional)
        config (dict): global config dict
    """

    def __init__(self, model, output_dir, config, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)

    def __call__(self, val_loader):
        assert isinstance(val_loader.dataset, AbstractHDF5Dataset)

        logger.info(f"Processing '{val_loader.dataset.file_path}'...")
        output_file = _get_output_file(dataset=val_loader.dataset, output_dir=self.output_dir)

        out_channels = self.config['model'].get('out_channels')

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Saving only channel '{prediction_channel}' from the network output")

        device = self.config['device']

        logger.info(f'Running prediction on {len(val_loader)} batches...')
        # dimensionality of the the output predictions
        # volume_shape = self.volume_shape(val_loader.dataset)
        volume_shape = tuple([96, 128, 128])
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        # create destination H5 file
        h5_output_file = h5py.File(output_file, 'w')

        number_samples = self.config["eval_metric"].get('num_samples')

        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              number_samples, h5_output_file)

        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        t0, t1, t2, t3, t4 = [], [],[],[], []

        # print(self.model)
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for input, targets, indices in val_loader:
                
                # send batch to device
                input = input.to(device)
                targets = targets.to(device)

                t0.append(time.perf_counter())
                # forward pass
                self.model.forward(input, targets)
                t1.append(time.perf_counter())

                predictions = []
                for idx in range(number_samples):
                    t2.append(time.perf_counter())
                    predictions.append(self.model.sample())
                    t3.append(time.perf_counter())

                t4.append(time.perf_counter())
                # for each sample
                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                        normalization_masks):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()
                    
                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)
                        
                        

                        logger.info(f'Saving predictions for slice:{index}...')
                        # remove halo in order to avoid block artifacts in the output probability maps
                        # u_prediction, u_index = remove_halo(pred, index, volume_shape, patch_halo)
                        # accumulate probabilities into the output prediction array
                        prediction_map[index] += pred
                        # count voxel visits for normalization
                        normalization_mask[index] += 1

        # save results
        logger.info(f'Saving predictions to: {output_file}')
        self._save_results(prediction_maps, normalization_masks, number_samples, h5_output_file, val_loader.dataset)
        # close the output H5 file
        h5_output_file.close()
        t0 = np.array(t0)
        t1 = np.array(t1)
        t2 = np.array(t2)
        t3 = np.array(t3)
        t4 = np.array(t4)
        print(f"Average forward: {np.mean(t1-t0)}")
        print(f"Sample: {np.mean(t3-t2)}")
        print(f"Average total: {np.mean(t4-t0)}")

    def _allocate_prediction_maps(self, output_shape, num_samples, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(num_samples)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(num_samples)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, number_samples, output_file, dataset):
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        prediction_datasets = self.get_output_dataset_names(number_samples, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask

            if dataset.mirror_padding is not None:
                z_s, y_s, x_s = [_slice_from_pad(p) for p in dataset.mirror_padding]

                logger.info(f'Dataset loaded with mirror padding: {dataset.mirror_padding}. Cropping before saving...')

                prediction_map = prediction_map[:, z_s, y_s, x_s]

            output_file.create_dataset(prediction_dataset, data=prediction_map, compression="gzip")

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"


class LazyValidator(StandardValidator):
    """
        Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
        Predicted patches are directly saved into the H5 and they won't be stored in memory. Since this Validator
        is slower than the `StandardValidator` it should only be used when the predicted volume does not fit into RAM.
        The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
        not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
        of the output head from the network.
        Args:
            model (Unet3D): trained 3D UNet model used for prediction
            output_dir (str): path to the output directory (optional)
            config (dict): global config dict
        """

    def __init__(self, model, output_dir, config, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # allocate datasets for probability maps
        prediction_datasets = self.get_output_dataset_names(output_heads, prefix='predictions')
        prediction_maps = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='float32', chunks=True,
                                       compression='gzip')
            for dataset_name in prediction_datasets]

        # allocate datasets for normalization masks
        normalization_datasets = self.get_output_dataset_names(output_heads, prefix='normalization')
        normalization_masks = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='uint8', chunks=True,
                                       compression='gzip')
            for dataset_name in normalization_datasets]

        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        if dataset.mirror_padding:
            logger.warn(
                f'Mirror padding unsupported in LazyValidator. Output predictions will be padded with pad_width: {dataset.pad_width}')

        prediction_datasets = self.get_output_dataset_names(output_heads, prefix='predictions')
        normalization_datasets = self.get_output_dataset_names(output_heads, prefix='normalization')

        # normalize the prediction_maps inside the H5
        for prediction_map, normalization_mask, prediction_dataset, normalization_dataset in zip(prediction_maps,
                                                                                                 normalization_masks,
                                                                                                 prediction_datasets,
                                                                                                 normalization_datasets):
            # split the volume into 4 parts and load each into the memory separately
            logger.info(f'Normalizing {prediction_dataset}...')

            z, y, x = prediction_map.shape[1:]
            # take slices which are 1/27 of the original volume
            patch_shape = (z // 3, y // 3, x // 3)
            for index in SliceBuilder._build_slices(prediction_map, patch_shape=patch_shape, stride_shape=patch_shape):
                logger.info(f'Normalizing slice: {index}')
                prediction_map[index] /= normalization_mask[index]
                # make sure to reset the slice that has been visited already in order to avoid 'double' normalization
                # when the patches overlap with each other
                normalization_mask[index] = 1

            logger.info(f'Deleting {normalization_dataset}...')
            del output_file[normalization_dataset]