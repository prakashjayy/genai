import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

class DiffusionModel(pl.LightningModule):
    def __init__(self, img_size=64, channels=3, time_steps=1000):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.time_steps = time_steps

        # U-Net architecture
        self.encoder = nn.Sequential(
            self._conv_block(channels, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
        )
        
        self.middle = nn.Sequential(
            self._conv_block(512, 512),
            self._conv_block(512, 512),
        )
        
        self.decoder = nn.Sequential(
            self._conv_block(512 + 512, 512, transpose=True),
            self._conv_block(512 + 256, 256, transpose=True),
            self._conv_block(256 + 128, 128, transpose=True),
            self._conv_block(128 + 64, 64, transpose=True),
        )
        
        self.final = nn.Conv2d(64 + channels, channels, kernel_size=1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

    def _conv_block(self, in_ch, out_ch, transpose=False):
        if not transpose:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        else:
            conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        return nn.Sequential(conv, nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, x, t):
        t = self.time_embed(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        t = t.expand(-1, -1, self.img_size, self.img_size)

        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)

        x4 = x4 + t
        x = self.middle(x4)

        x = self.decoder[0](torch.cat([x, x4], dim=1))
        x = self.decoder[1](torch.cat([x, x3], dim=1))
        x = self.decoder[2](torch.cat([x, x2], dim=1))
        x = self.decoder[3](torch.cat([x, x1], dim=1))

        return self.final(torch.cat([x, x], dim=1))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        t = torch.randint(0, self.time_steps, (x.shape[0],), device=self.device)
        loss = self.p_losses(x, t)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self(x_noisy, t / self.time_steps)
        return F.mse_loss(noise, predicted_noise)

    def q_sample(self, x_start, t, noise):
        # Implement linear beta schedule
        beta_start = 0.0001
        beta_end = 0.02
        beta_t = beta_start + t / self.time_steps * (beta_end - beta_start)
        sqrt_alphas_cumprod_t = torch.sqrt(1. - beta_t.cumsum(dim=0))
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - sqrt_alphas_cumprod_t ** 2)

        return (
            sqrt_alphas_cumprod_t.view(-1, 1, 1, 1) * x_start +
            sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1) * noise
        )

# Data module
class CelebADataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, img_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.celeba = CelebA(self.data_dir, split='train', transform=transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.celeba, batch_size=self.batch_size, shuffle=True)

# Training
model = DiffusionModel()
data_module = CelebADataModule("face_diffusion/data/")
trainer = pl.Trainer(max_epochs=100, gpus=1)
trainer.fit(model, data_module)