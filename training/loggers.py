import wandb
import torch
from PIL import Image
from collections import defaultdict
import os
from utils.class_registry import ClassRegistry

loggers_registry = ClassRegistry()


class WandbLogger:
    def __init__(self, config, run_id=None):
        wandb.login(key=os.environ["WANDB_KEY"].strip())
        self.run_id = run_id if run_id is not None else wandb.util.generate_id()
        self.wandb_args = {
            "id": self.run_id,
            "project": "genai_task",
            "name": f"{config['train']['model']}_{self.run_id}",
            "config": config,
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


@loggers_registry.add_to_registry(name="training_logger")
class TrainingLogger:
    def __init__(self, config, run_id=None):
        self.logger = WandbLogger(config, run_id=run_id)
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
        return self.logger.log_values(val_metrics, step)

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
