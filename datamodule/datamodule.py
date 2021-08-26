"""
Dataset & DataModule for Labeled Faces in the Wild,
a public benchmark for face verification, also known as pair matching.
"""
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.io import read_image


class LFWImageDataset(Dataset):

    """

    Attributes:
        file_list (List[str]):
        image_dir (str):
        label_enc (LabelEncoder):
        label_list (List[int]):
        transform (transforms):
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[List[Callable]] = None,
        min_files: Optional[int] = 0,
    ) -> None:

        assert min_files >= 0

        file_list = []
        str_label_list = []

        for root, _, files in os.walk(root_dir):
            if len(files) > min_files:
                file_list.extend(
                    [
                        os.path.join(root, name)
                        for name in files
                        if name.endswith(".jpg")
                    ]
                )

                str_label_list.extend(
                    [os.path.basename(root) for name in files if name.endswith(".jpg")]
                )

        self.file_list = file_list

        self.label_enc = LabelEncoder()
        self.label_list = torch.as_tensor(self.label_enc.fit_transform(str_label_list))

        self.image_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        image_path = os.path.join(self.file_list[idx])
        image = read_image(image_path)
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label_encoder(self) -> LabelEncoder:
        """

        Returns:
            LabelEncoder
        """
        return self.label_enc


class LFWDataModule(pl.LightningDataModule):

    """

    Attributes:
        batch_size (int):
        data_dir (str):
        label_enc (LabelEncoder):
        lfw_dataset (Dataset):
        train_size (float):
        val_size (float):
    """

    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: Optional[int] = 64,
        train_size: Optional[float] = 0.8,
        val_size: Optional[float] = 0.1,
        min_images: Optional[int] = 0,
    ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.lfw_dataset = None
        self.label_enc = None
        self.batch_size = batch_size
        self.min_images = min_images

        self.train_size = train_size
        self.val_size = val_size

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Dataset creation, Shuffle & split"""

        self.lfw_dataset = LFWImageDataset(self.data_dir, min_files=self.min_images)
        self.label_enc = self.lfw_dataset.get_label_encoder()

        shuffled_indices = np.random.permutation(len(self.lfw_dataset))

        train_split = int(self.train_size * len(shuffled_indices))
        test_split = int((self.train_size + self.val_size) * len(shuffled_indices))

        self.train_idx, self.val_idx, self.test_idx = np.split(
            shuffled_indices, [train_split, test_split]
        )

    def train_dataloader(self) -> DataLoader:
        train_sampler = SubsetRandomSampler(self.train_idx)
        return DataLoader(
            self.lfw_dataset, batch_size=self.batch_size, sampler=train_sampler
        )

    def val_dataloader(self) -> DataLoader:
        val_sampler = SubsetRandomSampler(self.val_idx)
        return DataLoader(
            self.lfw_dataset, batch_size=self.batch_size, sampler=val_sampler
        )

    def test_dataloader(self) -> DataLoader:
        test_sampler = SubsetRandomSampler(self.test_idx)
        return DataLoader(
            self.lfw_dataset, batch_size=self.batch_size, sampler=test_sampler
        )

    def decode_labels(self, labels: Union[List[int], int]) -> List[str]:
        """

        Args:
            labels (Union[List[int], int]):

        Returns:
            List[str]
        """
        if isinstance(labels, int):
            labels = [labels]

        return self.label_enc.inverse_transform(labels)


def main():
    """Simple script showcasing how the Dataset & DataModule work"""

    data_path = "..\\data\\lfw"
    min_images = 30

    def imshow_tensor(image: Tensor) -> None:
        plt.imshow(transforms.ToPILImage()(image), interpolation="bicubic")

    data_module = LFWDataModule(data_path, min_images=min_images)
    data_module.setup()

    for dataloader in [
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader(),
    ]:
        features, labels = next(iter(dataloader))
        print(f"Feature batch shape: {features.size()}")
        print(f"Labels batch shape: {labels.size()}")
        image = features[0].squeeze()
        label = labels[0]
        imshow_tensor(image)
        plt.show()
        print(f"Label: {data_module.decode_labels(label.tolist())}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
