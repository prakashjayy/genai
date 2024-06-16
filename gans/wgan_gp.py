# Simple WGAN implementation using Pytorch-Lightning 
# We will train this dataset on celebA dataset 
# A reimplementation of https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
import torch 
import torchvision
import torch.nn as nn 
import lightning.pytorch as pl

from torch import optim
from torch.autograd import Variable
from torch import autograd
from torchvision import transforms
from torchvision.datasets import CelebA

def load_train_val_dataloaders(batch_size=256):    
    train_ds = CelebA(
        root="data/",
        split="all",
        download=False,
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )
    # Create the dataloader
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=32)

    return train_dl#val_dl 

def deconv(ni, nf, ks=3, stride=2, pad=0, act=True, norm=True):
    res = [nn.ConvTranspose2d(ni, nf, stride=stride, kernel_size=ks, padding=pad)]
    if norm: 
        res = res+[nn.BatchNorm2d(nf)]
    if act: 
        res = res+[nn.ReLU()]
    return nn.Sequential(*res)

class Generator(nn.Module):
    def __init__(self, noise_channels, image_channels):
        super(Generator, self).__init__()
        """
        In this function the generator model will be defined with all of it layers.
        The generator model uses 4 ConvTranspose blocks. Each block containes 
        a ConvTranspose2d, BatchNorm2d and ReLU activation.
        """
        #12 million parameters
        self.fe = 128
        self.layer1 = deconv(noise_channels, self.fe*8, ks=4, stride=1, pad=0)
        self.layer2 = deconv(self.fe*8, self.fe*4, ks=4, stride=2, pad=1)
        self.layer3 = deconv(self.fe*4, self.fe*2, ks=4, stride=2, pad=1)
        self.layer4 = deconv(self.fe*2, self.fe, ks=4, stride=2, pad=1)
        self.layer5 = deconv(self.fe, image_channels, ks=4, stride=2, pad=1, act=False, norm=False)
        self.out = nn.Tanh()
    
    def forward(self, x): 
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return self.out(out)

def conv(ni, nf, ks=3, stride=2, pad=0, act=True, norm=True):
    res = [nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=pad)]
    if norm: 
        res = res+[nn.InstanceNorm2d(nf, affine=True)]
    if act: 
        res = res+[nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*res)

class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super(Discriminator, self).__init__()
        """
        This function will define the Discriminator model with all the layers needed.
        The model has 5 Conv blocks. The blocks have Conv2d, BatchNorm and LeakyReLU activation.
        """
        #10 million
        self.fe = 128
        self.model = nn.Sequential(*[
            conv(image_channels, self.fe, 4, 2, 1, True, True),
            conv(self.fe, self.fe*2, 4, 2, 1, True, True), 
            conv(self.fe*2, self.fe*4, 4, 2, 1, True, True),
            conv(self.fe*4, self.fe*8, 4, 2, 1, True, True)
        ])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #self.out = conv(self.fe*8, 1, 4, 1, 0, False, False) #outchannels is 1 because we want a linear for WGAN
        
        # define the model

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)        
        return x


class WGANGP(pl.LightningModule):
    def __init__(self, img_channels, latent_dim, lr,  batch_size=128):
        super(WGANGP, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.img_channels = img_channels
        self.lambda_term = 10
        self.critic_iter = 5
        self.generator = Generator(noise_channels=self.latent_dim, image_channels=self.img_channels)
        self.discriminator = Discriminator(image_channels=self.img_channels)
        self.lr = lr
        self.automatic_optimization = False
        self.z = torch.randn(16, self.latent_dim, 1, 1).to(self.device)

    def forward(self, z):
        return self.generator(z)

    def generator_step(self, batch_size):
        # Genertor creates an image which discriminator says them as 
        # fake but we need to consider them as real and calculate loss. 
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(z)
        #fake_labels = torch.ones(batch_size, 1).to(self.device)
        outputs = self.discriminator(fake_images)
        #g_loss = self.criterion(outputs, fake_labels)
        return torch.mean(outputs)

    def discriminator_step(self, real_images):
        # take a batch of images
        batch_size = real_images.size(0)
        outputs = self.discriminator(real_images)
        d_loss_real = outputs.mean()

        # take a batch of images from genertor
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        with torch.no_grad():
            fake_images = self.generator(z)
        outputs = self.discriminator(fake_images.detach())
        d_loss_fake = outputs.mean()
        gradient_penalty = self.calculate_gradient_penalty(real_images.data, fake_images.data)

        d_loss = - d_loss_real + d_loss_fake + gradient_penalty
        return d_loss
    
    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).to(self.device)
        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device), create_graph=True, retain_graph=True)[0]

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)
        
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def training_step(self, batch, batch_idx):
        real_images, _ = batch
        opt_g, opt_d = self.optimizers()

        # Train discriminator
        for _ in range(self.critic_iter):
            self.toggle_optimizer(opt_d)
            d_loss = self.discriminator_step(real_images)
            self.manual_backward(d_loss)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)
            self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        # Train generator
        self.toggle_optimizer(opt_g)
        g_loss = self.generator_step(real_images.size(0))
        self.manual_backward(-1* g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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


if __name__ == "__main__":
    # dis = Discriminator(3)
    # gan = Generator(100, 3)
    # x = torch.randn((4, 100, 1, 1))
    # gan_img = gan(x)
    # print(gan_img.shape)
    # img = dis(gan_img)
    # print(img.shape)
    print("Load_model")
    batch_size = 1024
    pipeline = WGANGP(3, 100, 0.0005, batch_size)
    
    print("Load trainer")
    tqdm_callback = pl.callbacks.TQDMProgressBar(refresh_rate=4)
    trainer = pl.Trainer(max_epochs=500, 
                         accelerator="gpu",
                         enable_model_summary=True,
                         #fast_dev_run=True,
                         logger=[pl.loggers.TensorBoardLogger("logs", name="wgangp")],
                         callbacks=[tqdm_callback])
    
    print("Load data")
    train_dl = load_train_val_dataloaders(batch_size=batch_size)

    trainer.fit(pipeline, train_dataloaders=train_dl)
