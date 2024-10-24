from utils.class_registry import ClassRegistry
from training.trainers.base_trainer import BaseTrainer

from models.diffusion_models import diffusion_models_registry
from training.optimizers import optimizers_registry
from datasets.datasets import datasets_registry
from training.loggers import loggers_registry
from metrics.metrics import metrics_registry
from datasets.dataloaders import InfiniteLoader
from tqdm import tqdm

import torch
import os
import shutil
import cv2

diffusion_trainers_registry = ClassRegistry()


@diffusion_trainers_registry.add_to_registry(name="base_diffusion_trainer")
class BaseDiffusionTrainer(BaseTrainer):
    def setup_models(self):
        # TO DO
        # self.unet = ...
        # self.encoder == ... # if needed
        # self.noise_scheduler = ...
        # do not forget to load state from checkpoints if provided
        raise NotImplementedError()

    def setup_optimizers(self):
        # TO DO
        # self.optimizer = ...
        # do not forget to load state from checkpoints if provided
        raise NotImplementedError()

    def setup_losses(self):
        # TO DO
        # self.loss_builder = ...
        raise NotImplementedError()

    def to_train(self):
        # TO DO
        # all trainable modules to .train()
        raise NotImplementedError()

    def to_eval(self):
        # TO DO
        # all trainable modules to .eval()
        raise NotImplementedError()

    def train_step(self):
        # TO DO
        # batch = next(self.train_dataloader)
        # timesteps = ...
        # add noise to images according self.noise_scheduler
        # predict noise via self.unet
        # calculate losses, make oprimizer step
        # return dict of losses to log
        raise NotImplementedError()

    def save_checkpoint(self):
        # TO DO
        # save all necessary parts of your pipeline
        raise NotImplementedError()

    def synthesize_images(self):
        # TO DO
        # synthesize images and save to self.experiment_dir/images
        # synthesized additional batch of images to log
        # return batch_of_images, path_to_saved_pics,
        raise NotImplementedError()


@diffusion_trainers_registry.add_to_registry(name="improved_diffusion_trainer")
class ImprovedDiffusionTrainer(BaseDiffusionTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.run_id = None
        self.metrics = {}

    def __create_model(
        self,
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma,
        class_cond,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        num_classes,
    ):
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return diffusion_models_registry[self.config["train"]["model"]](
            in_channels=3,
            model_channels=num_channels,
            out_channels=(3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(num_classes if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
        )

    def __create_gaussian_diffusion(
        self,
        steps=1000,
        learn_sigma=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
        sigma_small=False,
    ):
        betas = diffusion_models_registry["get_named_beta_schedule"](
            noise_schedule, steps
        )
        LossType = diffusion_models_registry["LossType"]
        ModelMeanType = diffusion_models_registry["ModelMeanType"]
        ModelVarType = diffusion_models_registry["ModelVarType"]
        if use_kl:
            loss_type = LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = LossType.RESCALED_MSE
        else:
            loss_type = LossType.MSE
        if not timestep_respacing:
            timestep_respacing = [steps]
        return diffusion_models_registry[self.config["train"]["diffusion"]](
            use_timesteps=diffusion_models_registry["space_timesteps"](
                steps, timestep_respacing
            ),
            betas=betas,
            model_mean_type=(
                ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )

    def setup_models(self):
        # create model for reconstruction
        self.model = self.__create_model(
            **self.config["model_args"]["reverse_process_args"]
        ).to(self.device)
        # create diffusion process
        self.diffusion = self.__create_gaussian_diffusion(
            **self.config["model_args"]["forward_process_args"]
        )
        # create noise sampler
        self.noise_sampler = diffusion_models_registry[
            self.config["train"]["noise_sampler"]
        ](self.diffusion)
        print("Model setup!")

    def setup_experiment_dir(self):
        experiments_dir = self.config["exp"]["exp_dir"]
        if not os.path.isdir(experiments_dir):
            print(f"Creating dir for experiments: {experiments_dir}")
            os.mkdir(experiments_dir)
        else:
            print(f"Using existing {experiments_dir}")

    def to_train(self):
        self.model.train()

    def to_eval(self):
        self.model.eval()

    def setup_metrics(self):
        for metric_name in self.config["train"]["val_metrics"]:
            self.metrics[metric_name] = metrics_registry[metric_name](
                batch_size=self.config["data"]["val_batch_size"], device=self.device
            )

    def setup_logger(self):
        self.logger = loggers_registry["training_logger"](self.config, self.run_id)

    def setup_losses(self):
        # loss is embedded into forward diffusion process
        pass

    def setup_optimizers(self):
        self.optimizer = optimizers_registry[self.config["train"]["optimizer"]](
            self.model.parameters(), **self.config["optimizer_args"]
        )

    def setup_datasets(self):
        dataset_type = self.config["data"]["dataset"]
        self.train_dataset = datasets_registry[dataset_type](
            self.config["data"]["input_train_dir"], train=True
        )
        self.test_dataset = datasets_registry[dataset_type](
            self.config["data"]["input_val_dir"], train=False
        )

    def setup_dataloaders(self):
        train_batch_size = self.config["data"]["train_batch_size"]
        num_workers = self.config["data"]["workers"]
        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.train_dataloader_iter = iter(self.train_dataloader)

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        images, labels = next(self.train_dataloader_iter)
        images = images.to(self.device)
        labels["y"] = labels["y"].to(self.device).type(torch.int64)
        # sample timesteps for the batch
        # returns timesteps per each sample in the batch and weights for each timestamp
        # in case of uniform scheduler, weights for each timestamp is equal to 1
        t, _ = self.noise_sampler.sample(images.shape[0], self.device)

        # internally this function:
        # > generates random noise to apply to the image
        # > does forward diffusion process
        # > gets the output from the model
        # > calculates loss
        losses = self.diffusion.training_losses(
            model=self.model, x_start=images, t=t, model_kwargs=labels
        )

        # loss: could be KL or MSE, or rescaled ones. Default rescaled MSE
        loss = losses["loss"].mean()
        loss.backward()
        # TODO add annealing
        self.optimizer.step()
        self.global_step += 1
        self.step += 1
        return {"train_loss": loss.item()}

    def save_checkpoint(self):
        experiments_dir = self.config["train"]["checkpoint_save_path"]
        model_name = self.config["train"]["model"]
        save_checkpoint_path = (
            f"{experiments_dir}/checkpoint_{model_name}_step_{self.global_step}.pkl"
        )
        print(f"Saving checkpoint at: {save_checkpoint_path}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "diffusion": self.diffusion,
                "noise_sampler": self.noise_sampler,
                "global_step": self.global_step,
                "run_id": self.logger.logger.run_id,
            },
            save_checkpoint_path,
        )

    def load_checkpoint(self):
        load_checkpoint_path = self.config["train"]["checkpoint_path"]
        print(f"Loading checkpoint: {load_checkpoint_path}")
        if not os.path.isfile(load_checkpoint_path):
            print("Err: no such file, not loading")
            return
        dict = torch.load(load_checkpoint_path)
        self.model.load_state_dict(dict["model_state_dict"])
        self.optimizer.load_state_dict(dict["optimizer_state_dict"])
        self.diffusion = dict["diffusion"]
        self.noise_sampler = dict["noise_sampler"]
        self.global_step = dict["global_step"]
        self.run_id = dict["run_id"]

    def synthesize_images(self):
        # this function just going to produce 20 images per class
        # because that is how many images in test per class
        # and save those in experiment folder
        print("Synthesizing images...")
        experiment_root = self.config["exp"]["exp_dir"]
        synthetic_images_dir = os.path.join(experiment_root, "synthethic")
        if os.path.isdir(synthetic_images_dir):
            shutil.rmtree(synthetic_images_dir)
        os.mkdir(synthetic_images_dir)

        images_per_class = self.config["train"]["val_images_per_class"]
        val_sample_images_num = self.config["train"]["val_sample_images"]
        image_samples = {}
        for label_name, label_value in tqdm(self.train_dataset.classes_to_num.items()):
            condition_dict = {
                "y": torch.tensor(
                    [label_value] * images_per_class,
                    dtype=torch.int64,
                    device=self.device,
                )
            }
            generations = self.inference(
                batch_size=images_per_class, labels=condition_dict
            )
            label_dir = os.path.join(synthetic_images_dir, label_name)
            os.mkdir(label_dir)
            for idx, generation in enumerate(generations):
                cv2.imwrite(
                    os.path.join(label_dir, f"{idx}.jpg"),
                    generation,
                    [cv2.IMWRITE_JPEG_QUALITY, 100],
                )
            if len(image_samples) < val_sample_images_num:
                image_samples[label_name] = generations[0]

        return image_samples, synthetic_images_dir

    @torch.no_grad()
    def inference(self, batch_size, labels=None):
        image_size = self.config["model_args"]["reverse_process_args"]["image_size"]
        generations = self.diffusion.p_sample_loop(
            self.model,
            (batch_size, 3, image_size, image_size),
            model_kwargs=labels,
            device=self.device,
        )
        # back to [0-255] range RGB images
        generations = (generations + 1) * 127.5
        # clamp images to [0-255] and convert to uint8
        generations = generations.clamp(0, 255).type(torch.uint8)
        # permute dimensions from [B, 3, H, W] to [B, H, W, 3] and return as numpy
        return generations.permute([0, 2, 3, 1]).detach().cpu().numpy()
