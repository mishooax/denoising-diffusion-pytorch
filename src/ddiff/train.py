from ddiff.unet import UNet
from ddiff.gaussdiff import GaussianDiffusion
from ddiff.trainer import Trainer


def train() -> None:
    """Trains a Gaussian diffusion model on an RGB Image dataset"""
    model = UNet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()
    diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000, loss_type="l1").cuda()

    # TODO: build up image dir
    trainer = Trainer(
        diffusion,
        "/data/malexe/cifar10/cifar-10-jpeg",
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
    )

    trainer.train()


if __name__ == "__main__":
    train()
