import numpy as np 
import torch
import torchvision
import torch.nn as nn
import lightning.pytorch as pl

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import optim

class FMDS(Dataset):
    def __init__(self, name, dtype="train", transforms=None):
        super().__init__()
        self.dsd = load_dataset(name)[dtype]
        self.dtype = dtype 
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dsd)
    
    def __getitem__(self, idx):
        t = self.dsd[idx]
        img, label = t["image"], t["label"]
        #label = self.labels[idx]
        if transforms is not None:
            img = self.transforms(img)
            label = torch.Tensor([label]).long()
        return img, label
    

def load_train_val_dataloaders(name, batch_size=256):    
    ct = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = FMDS(name, dtype="train", transforms=ct)
    #val_ds = FMDS(name, dtype="test", transforms=ct)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    #val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    return train_dl#val_dl 


def conv(ni, nf, ks=3, stride=2, act=True, norm=True):
    res = [nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2)]
    if norm: 
        res = res+[nn.BatchNorm2d(nf)]
    if act: 
        res = res+[nn.LeakyReLU(0.2)]
    return nn.Sequential(*res)


def deconv(ni, nf, ks=3, stride=2, pad=0, act=True, norm=True):
    res = [nn.ConvTranspose2d(ni, nf, stride=stride, kernel_size=ks, padding=pad)]
    if norm: 
        res = res+[nn.BatchNorm2d(nf)]
    if act: 
        res = res+[nn.ReLU()]
    return nn.Sequential(*res)


class Generator(nn.Module):
    def __init__(self, noise_channels, image_channels, features):
        super(Generator, self).__init__()
        """
        In this function the generator model will be defined with all of it layers.
        The generator model uses 4 ConvTranspose blocks. Each block containes 
        a ConvTranspose2d, BatchNorm2d and ReLU activation.
        """
        # define the model
        self.model = nn.Sequential(*[
            deconv(noise_channels, features*8, ks=4, stride=1, pad=0, act=True, norm=False),
            deconv(features*8, features*4, ks=4, stride=2, pad=1),
            deconv(features*4, features*2, ks=4, stride=2, pad=1), 
            #deconv(features*4, features*2, ks=4, stride=2, pad=1),
            deconv(features*2, image_channels, ks=4, stride=2, pad=1, act=False, norm=False)
        ])
        self.out = nn.Tanh()
    
    def forward(self, x):
        out = self.model(x)
        return self.out(out)


class Discriminator(nn.Module):
    def __init__(self, image_channels, features):
        super(Discriminator, self).__init__()
        """
        This function will define the Discriminator model with all the layers needed.
        The model has 5 Conv blocks. The blocks have Conv2d, BatchNorm and LeakyReLU activation.
        """
        self.model = nn.Sequential(*[
            conv(image_channels, features, 4, 2, True, False),
            conv(features, features*2, 4, 2, True, True), 
            conv(features*2, features*4, 4, 2, True, True), 
            conv(features*4, features*8, 4, 2, True, True),
            conv(features*8, 1, 4, 2, False, False) 

        ])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sigmoid()
        # define the model

    def forward(self, x):
        out = self.pool(self.model(x))
        return self.out(out).view(-1, 1)


# Define the GAN Lightning Module
class GAN(pl.LightningModule):
    def __init__(self, latent_dim, lr):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(noise_channels=self.latent_dim, image_channels=1, features=64)
        self.discriminator = Discriminator(image_channels=1, features=64)
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False
        self.z = torch.randn(16, self.latent_dim, 1, 1).to(self.device)

    def forward(self, z):
        return self.generator(z)

    def generator_step(self, batch_size):
        # Genertor creates an image which discriminator says them as 
        # fake but we need to consider them as real and calculate loss. 
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(z)
        fake_labels = torch.ones(batch_size, 1).to(self.device)
        outputs = self.discriminator(fake_images)
        g_loss = self.criterion(outputs, fake_labels)
        return g_loss

    def discriminator_step(self, real_images):
        # take a batch of images and consider their output as 1 
        # take a batch of images from genertor and consider them as 0
        # calculate both the losses and backprop 
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        outputs = self.discriminator(real_images)
        d_loss_real = self.criterion(outputs, real_labels)

        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        with torch.no_grad():
            fake_images = self.generator(z)
        outputs = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        return d_loss

    def training_step(self, batch, batch_idx):
        real_images, _ = batch
        opt_g, opt_d = self.optimizers()

        # Train generator
        self.toggle_optimizer(opt_g)
        g_loss = self.generator_step(real_images.size(0))
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Train discriminator
        self.toggle_optimizer(opt_d)
        d_loss = self.discriminator_step(real_images)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        lr = self.lr
        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]
    
    def on_train_epoch_end(self):
        z = self.z.to(self.device)
        sample_imgs = self.generator(z).detach().cpu()
        grid = torchvision.utils.make_grid(sample_imgs, normalize=True)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        # plt.figure(figsize=(8, 8))
        # plt.imshow(grid.permute(1, 2, 0))
        # plt.title('Generated Images')
        # plt.axis('off')
        # plt.show()



if __name__ == "__main__":
    
    #fake_vec = torch.randn((256, 10, 1, 1))
    #print(fake_vec.shape)

    print("Load_model")
    pipeline = GAN(100, 0.0005*4)
    
    print("Load trainer")
    tqdm_callback = pl.callbacks.TQDMProgressBar(refresh_rate=4)
    trainer = pl.Trainer(max_epochs=500, 
                         accelerator="gpu",
                         enable_model_summary=True,
                         #fast_dev_run=True,
                         logger=[pl.loggers.TensorBoardLogger("logs", name="basic")],
                         callbacks=[tqdm_callback])
    
    print("Load data")
    train_dl = load_train_val_dataloaders("fashion_mnist", batch_size=1024)

    trainer.fit(pipeline, train_dataloaders=train_dl)