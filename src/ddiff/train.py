from ddiff.unet import Unet
from ddiff.gaussdiff import GaussianDiffusion
from ddiff.trainer import Trainer


def train() -> None:
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

    diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000, loss_type="l1").cuda()  # number of steps  # L1 or L2

    # TODO: build up image dir
    trainer = Trainer(
        diffusion,
        "path/to/your/images",
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
    )

    trainer.train()


if __name__ == "__main__":
    train()
