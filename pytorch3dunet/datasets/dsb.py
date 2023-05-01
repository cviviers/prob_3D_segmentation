import collections
import os

import imageio
import numpy as np
import torch

from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats, sample_instances
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('DSB2018Dataset')


def dsb_prediction_collate(batch):
    """
    Forms a mini-batch of (images, paths) during test time for the DSB-like datasets.
    """
    error_msg = "batch must contain tensors or str; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], collections.Sequence):
        # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
        transposed = zip(*batch)
        return [dsb_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class DSB2018Dataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config, mirror_padding=(0, 32, 32), expand_dims=True,
                 instance_ratio=None, random_seed=0):
        assert os.path.isdir(root_dir), f'{root_dir} is not a directory'
        assert phase in ['train', 'val', 'test']

        # use mirror padding only during the 'test' phase
        if phase in ['train', 'val']:
            mirror_padding = None
        if mirror_padding is not None:
            assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        self.mirror_padding = mirror_padding

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, 'images')
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, expand_dims)
        self.file_path = images_dir
        self.instance_ratio = instance_ratio

        min_value, max_value, mean, std = calculate_stats(self.images)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')

        transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                 mean=mean, std=std)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != 'test':
            # load labeled images
            masks_dir = os.path.join(root_dir, 'masks')
            assert os.path.isdir(masks_dir)
            self.masks, _ = self._load_files(masks_dir, expand_dims)
            # prepare for training with sparse object supervision (allow sparse objects only in training phase)
            if self.instance_ratio is not None and phase == 'train':
                assert 0 < self.instance_ratio <= 1
                rs = np.random.RandomState(random_seed)
                self.masks = [sample_instances(m, self.instance_ratio, rs) for m in self.masks]
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                padded_imgs = []
                for img in self.images:
                    padded_img = np.pad(img, pad_width=pad_width, mode='reflect')
                    padded_imgs.append(padded_img)

                self.images = padded_imgs

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase != 'test':
            mask = self.masks[idx]
            return self.raw_transform(img), self.masks_transform(mask)
        else:
            return self.raw_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)
        expand_dims = dataset_config.get('expand_dims', True)
        instance_ratio = phase_config.get('instance_ratio', None)
        random_seed = phase_config.get('random_seed', 0)
        return [cls(file_paths[0], phase, transformer_config, mirror_padding, expand_dims, instance_ratio, random_seed)]

    @staticmethod
    def _load_files(dir, expand_dims):
        files_data = []
        paths = []
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            img = np.asarray(imageio.imread(path))
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths
