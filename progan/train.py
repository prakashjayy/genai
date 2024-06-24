import os 
import torch
import torchvision
import lightning.pytorch as pl

from torch import optim
from torchvision.utils import save_image

from model import Generator, Discriminator
from utils import gradient_penalty
from dl import get_dl


class PROGAN(pl.LightningModule):
    def __init__(self, hparams):
        super(PROGAN, self).__init__()
        self.hparams.update(hparams)
        self.generator = Generator(self.hparams.z_dim, self.hparams.in_channels, self.hparams.factors, img_channels=self.hparams.image_channels)
        self.discriminator = Discriminator(in_channels = self.hparams.in_channels, factors=self.hparams.factors, img_channels=self.hparams.image_channels)
        self.automatic_optimization = False
        self.z = torch.randn(64, self.hparams.z_dim, 1, 1).to(self.device)
        self.step = 0
        self.alpha = 0
        #self.z = torch.randn(16, self.latent_dim, 1, 1).to(self.device)

    def forward(self, z, alpha, step):
        return self.generator(z, alpha, step)
    
    # def generate_examples(self, steps, save_dir, n=100):
    #     self.generator.eval()
    #     alpha = 1.0
    #     for i in range(n):
    #         with torch.no_grad():
    #             noise = torch.randn(1, self.hparams.z_dim, 1, 1).to(self.device)
    #             img = self.generator(noise, alpha, steps)
    #             #TODO: we need to store them in the run 
    #             if not os.path.exists(save_dir):
    #                 os.makedirs(save_dir)
    #             save_image(img*0.5+0.5, f"{save_dir}/img_{i}.png")
    #     self.generator.train()

    def generator_step(self, batch_size, alpha, step):
        # Genertor creates an image which discriminator says them as 
        # fake but we need to consider them as real and calculate loss. 
        noise = torch.randn(batch_size, self.hparams.z_dim, 1, 1).to(self.device)
        fake = self.generator(noise, alpha, step)
        gen_fake = self.discriminator(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)
        return loss_gen

    def discriminator_step(self, real_images, alpha, step):
        # take a batch of images and consider their output as 1 
        # take a batch of images from genertor and consider them as 0
        # calculate both the losses and backprop 
        cur_batch_size = real_images.shape[0]
        noise = torch.randn(cur_batch_size, self.hparams.z_dim, 1, 1).to(self.device)

        fake = self.generator(noise, alpha, step)
        critic_real = self.discriminator(real_images, alpha, step)
        critic_fake = self.discriminator(fake.detach(), alpha, step)
        gp = gradient_penalty(self.discriminator, real_images, fake, alpha, step, device=self.device)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + self.hparams.lambda_gp * gp
            + (0.001 * torch.mean(critic_real ** 2))
        )
        return loss_critic

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        self.alpha += batch.size(0) / (
            (self.hparams.progressive_epochs[self.step] * 0.5) * self.hparams.total_images
        )
        alpha = min(self.alpha, 1)

        # Train discriminator
        self.toggle_optimizer(opt_d)
        d_loss= self.discriminator_step(batch, alpha, self.step)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Train generator
        self.toggle_optimizer(opt_g)
        g_loss = self.generator_step(batch.size(0), alpha, self.step)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('step', self.step, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('alpha', self.alpha, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('image_size', batch.size(2), on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]
    
    def on_train_epoch_end(self):
        z = self.z.to(self.device)
        sample_imgs = self(z, 1, self.step).detach().cpu()
        grid = torchvision.utils.make_grid(sample_imgs, normalize=True)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
    
    def train_dataloader(self):
        print(f"Loading Dataloader: {self.current_epoch}")
        if self.current_epoch !=0:
            self.step+=1
        image_size = 2**(2+self.step)
        batch_size = self.hparams.batch_sizes[self.step]
        train_dl = get_dl(self.hparams.image_root, (image_size, image_size), batch_size=batch_size)
        self.alpha = 0
        return train_dl

if __name__ == "__main__":
    
    #fake_vec = torch.randn((256, 10, 1, 1))
    #print(fake_vec.shape)
    from mmengine import Config
    cfg = Config.fromfile("config.py")
    print(cfg)


    print("Load_model")
    pipeline = PROGAN(cfg)
    
    print("Load trainer")
    tqdm_callback = pl.callbacks.TQDMProgressBar(refresh_rate=4)
    trainer = pl.Trainer(max_epochs=sum(cfg.progressive_epochs), 
                         devices=1,
                         accelerator="gpu",
                         enable_model_summary=True,
                         reload_dataloaders_every_n_epochs=cfg.progressive_epochs[0],
                         logger=[pl.loggers.TensorBoardLogger("logs", name="progan")],
                         callbacks=[tqdm_callback])
    trainer.fit(pipeline)