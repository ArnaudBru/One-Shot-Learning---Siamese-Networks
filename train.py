""" Main training module"""
import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule.datamodule import (DataModule, LFWPairedImageDataset,
                                   LFWTripletImageDataset)
from model.model import SiameseNetwork


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Siamese Network - Face Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_folder", default=None)

    # Trainer API reference for possible flags
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html
    parser = Trainer.add_argparse_args(parser)
    parser = SiameseNetwork.add_model_specific_args(parser)

    args = parser.parse_args()

    # TODO: Remove when project is over
    if args.data_folder is None:
        data_path = "data\\lfw"
        abs_path = os.path.dirname(os.path.realpath(__file__))
        abs_path = os.path.abspath(os.path.join(abs_path, data_path))
        args.data_folder = abs_path

    min_images = 0

    if args.gpus is None:
        args.gpus = torch.cuda.device_count()

    lfw_dataset = LFWPairedImageDataset(args.data_folder, min_files=min_images)
    data_module = DataModule(lfw_dataset)

    if args.model_path is not None:
        model = SiameseNetwork.load_from_checkpoint(
            args.model_path, learning_rate=args.learning_rate, margin=args.margin
        )
    else:
        model = SiameseNetwork(args.learning_rate, args.margin)

    # TODO: Remove unecessary callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="siamese-network-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    lr_monitor = LearningRateMonitor()

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.000, patience=10
    )

    tb_logger = TensorBoardLogger("logs/")

    trainer = Trainer.from_argparse_args(
        # args, callbacks=[early_stop_callback, checkpoint_callback, lr_monitor], logger=tb_logger
        args,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=tb_logger
        # args, callbacks=[early_stop_callback], logger=tb_logger
    )
    trainer.fit(model, data_module)

    # trainer.test(...)


if __name__ == "__main__":
    main()
