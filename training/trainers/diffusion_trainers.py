from utils.class_registry import ClassRegistry
from training.trainers.base_trainer import BaseTrainer

from models.diffusion_models import diffusion_models_registry
from training.optimizers import optimizers_registry
from training.losses.diffusion_losses import DiffusionLossBuilder



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
class ImprovedDiffusionTrainer(BaseTrainer):
    def __init__(self, config):
        print(config)
        super().__init__(config)
    
    def __create_model(self,
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
                        num_classes):
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

        return diffusion_models_registry[self.config['train']['model']](
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
        sigma_small=False
    ):
        betas = diffusion_models_registry["get_named_beta_schedule"](noise_schedule, steps)
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
        return diffusion_models_registry[self.config['train']['diffusion']](
            use_timesteps=diffusion_models_registry["space_timesteps"](steps, timestep_respacing),
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
        self.model = self.__create_model(**self.config['model_args']['reverse_process_args'])
        # create diffusion process
        diffusion = self.__create_gaussian_diffusion(**self.config['model_args']['forward_process_args']) 
        # create noise sampler
        self.noise_sampler = diffusion_models_registry[self.config['train']['noise_sampler']](diffusion)

    def setup_experiment_dir(self):
        pass