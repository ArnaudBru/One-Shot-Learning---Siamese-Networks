"""Model and Loss definition"""
from argparse import ArgumentParser
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torchmetrics import AUROC


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin
        self.distance = nn.PairwiseDistance(p=2, keepdim=True)

    def forward(self, output1: Tensor, output2: Tensor, label: Tensor) -> float:
        """Calculates Contrastive loss

        Args:
            output1 (Tensor):
            output2 (Tensor):
            label (Tensor):

        Returns:
            float
        """
        euclidean_distance = self.distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class ConvolutionBlock(nn.Module):
    """Convolution Block with ReLU activation, BatchNorm and Max-pooling (Optional)

    Attributes:
        conv (nn.Module):
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Union[int, str] = 'same',
        max_pooling: Optional[bool] = True,
    ) -> None:
        super().__init__()
        if max_pooling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=padding
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(2, stride=2),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Computes output of the block

        Args:
            x (Tensor):

        Returns:
            Tensor
        """
        x = self.conv(x)
        return x


class FullyConnectedBlock(nn.Module):
    """Fully connected block with ReLU activation and BatchNorm
    Flattening can be added prior to the block (Optional)

    Attributes:
        fully_connected (nn.Module):
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        flatten: Optional[bool] = False,
    ) -> None:
        super().__init__()
        if flatten:
            self.fully_connected = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.fully_connected = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Computes output of the block

        Args:
            x (Tensor):

        Returns:
            Tensor
        """
        x = self.fully_connected(x)
        return x


class SiameseNetwork(pl.LightningModule):

    """

    Attributes:
        conv_block_1 (ConvolutionBlock): Convolutional block 1
        conv_block_2 (ConvolutionBlock): Convolutional block 2
        conv_block_3 (ConvolutionBlock): Convolutional block 3
        conv_block_4 (ConvolutionBlock): Convolutional block 4
        criterion (TYPE): Loss function
        fc_block (FullyConnectedBlock): Fully Connected block
        learning_rate (float): learning rate of the optimizer
    """

    def __init__(self, learning_rate: float, margin: float) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.margin = margin
        self.criterion = ContrastiveLoss(margin=1.0)

        self.roc_auc = AUROC(pos_label=1)

        channels = 64

        self.conv_block_1 = ConvolutionBlock(
            3, channels, kernel_size=3, padding=1, max_pooling=False
        )
        self.conv_block_2 = ConvolutionBlock(
            channels, 2 * channels, kernel_size=3, padding=1, max_pooling=True
        )
        self.conv_block_3 = ConvolutionBlock(
            2 * channels, 2 * channels, kernel_size=3, padding=1, max_pooling=False
        )
        self.conv_block_4 = ConvolutionBlock(
            2 * channels, 2 * channels, kernel_size=3, padding=1, max_pooling=True
        )
        self.conv_block_5 = ConvolutionBlock(
            2 * channels, 4 * channels, kernel_size=3, padding=1, max_pooling=True
        )
        self.conv_block_6 = ConvolutionBlock(
            4 * channels, 4 * channels, kernel_size=3, padding=1, max_pooling=True
        )

        self.fc_block = FullyConnectedBlock(256 * 15 * 15, 64, flatten=True)

    def _forward_one_network(self, x: Tensor) -> Tensor:
        # TODO: remove comments
        #print("-----------------")
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        #print(x.shape)
        x = self.fc_block(x)
        #print(x.shape)
        return x

    def forward(
        self, ref_images: Tensor, other_images: Tensor
    ) -> Tuple[Tensor, Tensor]:
        ref_output = self._forward_one_network(ref_images)
        other_output = self._forward_one_network(other_images)
        return ref_output, other_output

    def training_step(self, batch, batch_idx) -> float:
        (ref_images, other_images), labels = batch
        ref_output, other_output = self(ref_images, other_images)
        loss = self.criterion(ref_output, other_output, labels)

        preds = self.criterion.distance(ref_output, other_output).flatten()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_auc", self.roc_auc(preds, labels), on_epoch=True)
        return loss

    # def training_epoch_end(self):
    #     """Computes epoch ROCAUC"""
    #     self.log('train_auc_epoch', self.roc_auc.compute())

    def validation_step(self, batch, batch_idx) -> float:
        (ref_images, other_images), labels = batch
        ref_output, other_output = self(ref_images, other_images)
        loss = self.criterion(ref_output, other_output, labels)

        preds = self.criterion.distance(ref_output, other_output).flatten()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_auc", self.roc_auc(preds, labels), on_epoch=True)
        return loss

    # def validation_epoch_end(self):
    #     """Computes epoch ROCAUC"""
    #     self.log('val_auc_epoch', self.roc_auc.compute())

    def test_step(self, batch, batch_idx) -> float:
        (ref_images, other_images), labels = batch
        ref_output, other_output = self(ref_images, other_images)
        loss = self.criterion(ref_output, other_output, labels)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to the parser

        Args:
            parent_parser (ArgumentParser):

        Returns:
            ArgumentParser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--margin", type=float, default=1.0)
        parser.add_argument("--model_path", type=str, default=None)
        return parser
