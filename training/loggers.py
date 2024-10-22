import wandb
import torch
from PIL import Image
from collections import defaultdict
import os


class WandbLogger:
    def __init__(self, config):
        wandb.login(key=os.environ["WANDB_KEY"].strip())
        self.wandb_args = {
            "id": wandb.util.generate_id(),
            "project": "genai_task",
            "name": config["train"]["model"],
            "config": {"test": "test"},
        }

        wandb.init(**self.wandb_args, resume="allow")

    @staticmethod
    def log_values(values_dict: dict, step: int):
        # TO DO
        # log values to wandb
        wandb.log(values_dict, step=step)

    @staticmethod
    def log_images(images: dict, step: int):
        # TO DO
        # log images
        raise NotImplementedError()


class TrainingLogger:
    def __init__(self, config):
        self.logger = WandbLogger(config)
        self.losses_memory = defaultdict(list)

    def log_train_losses(self, step: int):
        # avarage losses in losses_memory
        # log them and clear losses_memory
        losses_avg = defaultdict(list)
        for loss_name, loss_val in self.losses_memory.items():
            losses_avg[loss_name] = sum(loss_val) / len(loss_val)
        self.logger.log_values(losses_avg, step)
        self.losses_memory = defaultdict(list)

    def log_val_metrics(self, val_metrics: dict, step: int):
        # TO DO
        pass

    def log_batch_of_images(
        self, batch: torch.Tensor, step: int, images_type: str = ""
    ):
        # TO DO
        pass

    def update_losses(self, losses_dict):
        # it is useful to average losses over a number of steps rather than track them at each step
        # this makes training curves smoother
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)
