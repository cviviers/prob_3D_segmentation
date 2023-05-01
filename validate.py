import importlib
import os

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_validation_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger('UNet3DValidate')


def _get_validator(model, output_dir, config):
    validator_config = config.get('validator', {})
    class_name = validator_config.get('name', 'StandardValidator')

    m = importlib.import_module('pytorch3dunet.unet3d.validator')
    validator_class = getattr(m, class_name)

    return validator_class(model, output_dir, config, **validator_config)


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)

    device = config['device']
    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')

    # create validator instance
    validator = _get_validator(model, output_dir, config)

    for val_loader in get_validation_loaders(config):
        # run the model prediction on the val_loader and save the results in the output_dir
        print(val_loader)
        validator(val_loader)


if __name__ == '__main__':
    main()