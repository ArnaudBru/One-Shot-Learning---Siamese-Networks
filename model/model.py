from typing import Optional

import pytorch_lightning as pl
from torch import nn, optim
import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        max_pooling: Optional[bool] = True,
    ) -> None:
        super(ConvolutionBlock, self).__init__()
        if max_pooling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding
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

    def forward(self, x):
        x = self.conv(x)
        return x

class FullyConnectedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        flatten: Optional[bool] = False,
    ) -> None:
        super(FullyConnectedBlock, self).__init__()
        if flatten:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class SiameseNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.learning_rate = 1e-4
        self.criterion = ContrastiveLoss(margin=1.0)

        channels = 64

        self.conv_block_1 = ConvolutionBlock(
            3, channels, kernel_size=5, padding=2, max_pooling=True
        )
        self.conv_block_2 = ConvolutionBlock(
            channels, 2 * channels, kernel_size=5, padding=2, max_pooling=True
        )
        self.conv_block_3 = ConvolutionBlock(
            2 * channels, 4 * channels, kernel_size=3, padding=1, max_pooling=True
        )
        self.conv_block_4 = ConvolutionBlock(
            4 * channels, 8 * channels, kernel_size=3, padding=1, max_pooling=False
        )

        self.fc_block = FullyConnectedBlock(492032, 8, flatten=True)

    def _forward_one_network(self, x):
        # print('-----------------------')
        # print(x.shape)
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.conv_block_3(x)
        # print(x.shape)
        x = self.conv_block_4(x)
        # print(x.shape)
        x = self.fc_block(x)
        # print(x.shape)
        return x

    def forward(self, ref_images, other_images):
        ref_output = self._forward_one_network(ref_images)
        other_output = self._forward_one_network(other_images)
        return ref_output, other_output

    def training_step(self, batch, batch_idx):
        ref_images, other_images, labels = batch
        ref_output, other_output = self(ref_images, other_images)
        loss = self.criterion(ref_output, other_output, labels)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ref_images, other_images, labels = batch
        ref_output, other_output = self(ref_images, other_images)
        loss = self.criterion(ref_output, other_output, labels)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        ref_images, other_images, labels = batch
        ref_output, other_output = self(ref_images, other_images)
        loss = self.criterion(ref_output, other_output, labels)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
