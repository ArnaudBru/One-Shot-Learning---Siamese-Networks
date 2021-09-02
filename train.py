""" Main training module"""
import argparse
import os

import torch
from pytorch_lightning import Trainer

from datamodule.datamodule import LFWImageDataset, PairedDataModule
from model.model import SiameseNetwork


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Siamese Network - Face Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_folder", default=None)
    # parser = Trainer.add_argparse_args(parser)
    parser = PairedDataModule.add_model_specific_args(parser)

    args = parser.parse_args()

    if args.data_folder is None:
        data_path = "data\\lfw"
        abs_path = os.path.dirname(os.path.realpath(__file__))
        abs_path = os.path.abspath(os.path.join(abs_path, data_path))
        args.data_folder = abs_path

    min_images = 0

    number_of_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {number_of_gpus}")

    lfw_dataset = LFWImageDataset(args.data_folder, min_files=min_images)
    data_module = PairedDataModule(lfw_dataset)

    model = SiameseNetwork()

    trainer = Trainer(gpus=number_of_gpus)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
