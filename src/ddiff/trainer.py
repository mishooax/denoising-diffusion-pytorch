from functools import partial
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.utils import save_image

from ddiff.dataset import ImageDataset
from ddiff.utils import cycle, num_to_groups
from ddiff.layers import EMA
from ddiff.logger import get_logger

LOGGER = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        diffusion_model: nn.Module,
        folder,
        *,
        ema_decay: float = 0.995,
        image_size: int = 128,
        train_batch_size: int = 128,
        train_lr: float = 2e-5,
        train_num_steps: int = 100000,
        gradient_accumulate_every: int = 2,
        # fp16=False,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
        save_and_sample_every: int = 1000,
        results_folder: str = "/data/malexe/scratch/DDIFF/training/results",
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = ImageDataset(folder, image_size)
        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True))
        self.opt = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0
        self.fp16 = False

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        save_data = {"step": self.step, "model": self.model.state_dict(), "ema": self.ema_model.state_dict()}
        torch.save(save_data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        load_data = torch.load(str(self.results_folder / f"model-{milestone}.pt"))
        self.step = load_data["step"]
        self.model.load_state_dict(load_data["model"])
        self.ema_model.load_state_dict(load_data["ema"])

    def train(self):
        # backwards = partial(loss_backwards, self.fp16)

        while self.step < self.train_num_steps:
            for _ in range(self.gradient_accumulate_every):
                batch_data = next(self.dl).cuda()
                loss = self.model(batch_data)
                if self.step % 100 == 0:
                    LOGGER.debug("%d: %.2e", self.step, loss.item())
                (loss / self.gradient_accumulate_every).backward()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(36, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(self.results_folder / f"sample-{milestone}.png"), nrow=6)
                self.save(milestone)

            self.step += 1

        LOGGER.info("--- Training completed. ---")
