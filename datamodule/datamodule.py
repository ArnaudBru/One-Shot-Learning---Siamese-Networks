"""
Dataset & DataModule for Labeled Faces in the Wild,
a public benchmark for face verification, also known as pair matching.
"""
import os
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.io import read_image


class LFWPairedImageDataset(Dataset):

    """

    Attributes:
        files_dico (dict):
        length (int):
        transform (transforms):
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[List[Callable]] = None,
        min_files: Optional[int] = 0,
    ) -> None:

        assert min_files >= 0

        files_dico = {}
        length = 0

        # Add files if the class contains enough samples
        for root, _, files in os.walk(root_dir):
            jpg_files = [
                os.path.join(root, name) for name in files if name.endswith(".jpg")
            ]

            if len(jpg_files) > min_files:
                files_dico[os.path.basename(root)] = jpg_files
                length += len(jpg_files)

        self.files_dico = files_dico
        self.length = length

        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        same_class = not random.getrandbits(1)
        label = int(same_class)

        if same_class:
            multi_image_class = self._select_multi_image_class(self.files_dico)
            image_1, image_2 = random.sample(self.files_dico[multi_image_class], k=2)

        else:
            class_1, class_2 = self._select_distinct_classes(self.files_dico)
            # Select two distinct images
            image_1 = random.choice(self.files_dico[class_1])
            image_2 = random.choice(self.files_dico[class_2])

        image_1 = read_image(image_1)
        image_2 = read_image(image_2)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        else:
            image_1 = transforms.ConvertImageDtype(torch.float)(image_1)
            image_2 = transforms.ConvertImageDtype(torch.float)(image_2)

        return (image_1, image_2), label

    @staticmethod
    def _select_multi_image_class(dictionnary: dict) -> str:
        """Select class containing multiple samples

        Args:
            dictionnary (dict):

        Returns:
            str
        """
        while True:
            multi_image_class = random.choice(list(dictionnary.keys()))
            if len(dictionnary[multi_image_class]) >= 2:
                break
        return multi_image_class

    @staticmethod
    def _select_distinct_classes(dictionnary: dict) -> List[str]:
        """Select two distinct classes from dictionnary's keys

        Args:
            dictionnary (dict):

        Returns:
            List[str]
        """
        return random.sample(list(dictionnary.keys()), k=2)


class LFWTripletImageDataset(Dataset):

    """

    Attributes:
        files_dico (dict):
        length (int):
        transform (transforms):
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[List[Callable]] = None,
        min_files: Optional[int] = 0,
    ) -> None:

        assert min_files >= 0

        files_dico = {}
        length = 0

        # Add files if the class contains enough samples
        for root, _, files in os.walk(root_dir):
            jpg_files = [
                os.path.join(root, name) for name in files if name.endswith(".jpg")
            ]

            if len(jpg_files) > min_files:
                files_dico[os.path.basename(root)] = jpg_files
                length += len(jpg_files)

        self.files_dico = files_dico
        self.length = length

        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        multi_image_class = self._select_multi_image_class(self.files_dico)
        anchor, positive = random.sample(self.files_dico[multi_image_class], k=2)

        negative_class = self._select_other_class(self.files_dico, multi_image_class)
        negative = random.choice(self.files_dico[negative_class])

        anchor = read_image(anchor)
        positive = read_image(positive)
        negative = read_image(negative)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        else:
            anchor = transforms.ConvertImageDtype(torch.float)(anchor)
            positive = transforms.ConvertImageDtype(torch.float)(positive)
            negative = transforms.ConvertImageDtype(torch.float)(negative)

        # We return a label just for compatibility
        return (anchor, positive, negative), 0

    @staticmethod
    def _select_multi_image_class(dictionnary: dict) -> str:
        """Select class containing multiple samples

        Args:
            dictionnary (dict):

        Returns:
            str
        """
        while True:
            multi_image_class = random.choice(list(dictionnary.keys()))
            if len(dictionnary[multi_image_class]) >= 2:
                break
        return multi_image_class

    @staticmethod
    def _select_other_class(dictionnary: dict, anchor_class: str) -> str:
        """Select a class different from anchor_class in dictionnary's keys
        In practice select two distinct classes
        returns the first one if different from anchor class
        else returns the second class

        Args:
            dictionnary (dict):
            anchor_class (str):

        Returns:
            str: other_class
        """
        classes = random.sample(list(dictionnary.keys()), k=2)
        return classes[0] if classes[0] != anchor_class else classes[1]


class DataModule(pl.LightningDataModule):

    """
    Attributes:
        batch_size (int)
        dataset (Dataset):
        test_idx (List):
        train_idx (List):
        train_size (float)
        val_idx (List):
        val_size (float)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 32,
        train_size: Optional[float] = 0.8,
        val_size: Optional[float] = 0.1,
    ) -> None:

        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_size = train_size
        self.val_size = val_size

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Dataset creation, Shuffle & split"""

        shuffled_indices = np.random.permutation(len(self.dataset))

        train_split = int(self.train_size * len(shuffled_indices))
        test_split = int((self.train_size + self.val_size) * len(shuffled_indices))

        self.train_idx, self.val_idx, self.test_idx = np.split(
            shuffled_indices, [train_split, test_split]
        )

    def train_dataloader(self) -> DataLoader:
        train_sampler = SubsetRandomSampler(self.train_idx)
        return DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=train_sampler
        )

    def val_dataloader(self) -> DataLoader:
        val_sampler = SubsetRandomSampler(self.val_idx)
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=val_sampler)

    def test_dataloader(self) -> DataLoader:
        test_sampler = SubsetRandomSampler(self.test_idx)
        return DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=test_sampler
        )


def imshow_tensor(image: Tensor) -> None:
    """Show image from tensor

    Args:
        image (Tensor):
    """
    plt.imshow(transforms.ToPILImage()(image), interpolation="bicubic")


def showcase_dataset(dataset: Dataset, plot_samples: Optional[bool] = True):
    """Turns dataset into DataModule
    Shows different charactiristics
    Optionnaly plots samples

    Args:
        dataset (Dataset):
        plot_samples (Optional[bool], optional):
    """
    data_module = DataModule(dataset)
    data_module.setup()

    for dataloader in [
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader(),
    ]:
        print(f"Length: {len(dataloader)}")
        images_tuples, labels = next(iter(dataloader))
        print(f"Feature batch shape: {images_tuples[0].size()}")
        print(f"Labels batch shape: {labels.size()}")
        label = labels[0]
        print(f"Label: {label}")
        if plot_samples:
            for images in images_tuples:
                image = images[0].squeeze()
                plt.figure()
                imshow_tensor(image)
            plt.show()


def main():
    """Simple script showcasing how the Dataset & DataModule work"""

    data_path = "data\\lfw"
    abs_path = os.path.dirname(os.path.realpath(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path, os.pardir, data_path))

    min_images = 0

    print("Test Pair Dataset")
    print("-----------------")
    pair_dataset = LFWPairedImageDataset(abs_path, min_files=min_images)
    showcase_dataset(pair_dataset)

    print("Test Triplet Dataset")
    print("-----------------")
    triplet_dataset = LFWTripletImageDataset(abs_path, min_files=min_images)
    showcase_dataset(triplet_dataset)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
