##### model.py #####
import torch
import lightning.pytorch as pl
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, decollate_batch
from config import CHANNELS, STRIDES, NUM_RES_UNITS, OUT_CHANNELS, LEARNING_RATE

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(spatial_dims=3, in_channels=1, out_channels=OUT_CHANNELS,
                          channels=CHANNELS, strides=STRIDES, num_res_units=NUM_RES_UNITS)
        self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)

