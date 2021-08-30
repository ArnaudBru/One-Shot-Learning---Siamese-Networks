import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch import Tensor
from torchvision import transforms

from datamodule.datamodule import LFWImageDataset, PairedDataModule
from model.model import SiameseNetwork


def main():
    data_path = "data\\lfw"
    abs_path = os.path.dirname(os.path.realpath(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path, data_path))

    min_images = 0

    def imshow_tensor(image: Tensor) -> None:
        plt.imshow(transforms.ToPILImage()(image), interpolation="bicubic")

    lfw_dataset = LFWImageDataset(abs_path, min_files=min_images)
    data_module = PairedDataModule(lfw_dataset)
    data_module.setup()

    model = SiameseNetwork()

    trainer = Trainer()
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

if __name__=="__main__":
    main()
