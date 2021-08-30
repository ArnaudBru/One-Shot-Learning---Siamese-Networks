from typing import Optional

import pytorch_lightning as pl
from torch import nn, optim


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
                    padding=kernel_size,
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class SiameseNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()

        channels = 64

        self.conv_block_1 = ConvolutionBlock(3, channels, kernel_size=5, padding=2, max_pooling=True)
        self.conv_block_2 = ConvolutionBlock(channels, 2*channels, kernel_size=5, padding=2, max_pooling=True)
        self.conv_block_3 = ConvolutionBlock(2*channels, 4*channels, kernel_size=3, padding=1, max_pooling=True)
        self.conv_block_4 = ConvolutionBlock(4*channels, 8*channels, kernel_size=3, padding=1, max_pooling=False)

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*channels * 8*channels, 16*channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16*channels),
        )

    def forward_one_network(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.fc_block(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_one_network(x1)
        output2 = self.forward_one_network(x2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1, output2, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1, output2, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1, output2, y)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
