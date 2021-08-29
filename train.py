import pytorch_lightning as pl
import torch

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)

            nn.Flatten()

            nn.Linear(256 * 512, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024))

    def forward_one_network(self, x):
        return self.net(x)

    def forward(self, x1, x2):
        output1 = self.forward_one_network(x1)
        output2 = self.forward_one_network(x2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass