import torch
from torch import nn
from tqdm import tqdm
from featurevis.ops import GaussianBlur
from .inr import INRTemplates
from .contrastive_loss import SimCLROnGrid
from .utils.training import (
    JitteringGridDatamodule,
    ParamReduceOnPlateau,
    check_activity_requirements,
)
from .utils.img_transforms import StandardizeClip, FixEnergyNormClip
from .utils.mask import apply_mask
from .utils.general import SingleNeuronModel, resolve_device
# from .utils.mei import check_meis


class InvarianceManifold:
    def __init__(
        self,
        response_predicting_model,
        meis_dict,
        device=None,
        # INR template config
        widths=[50] * 4,
        positional_encoding_dim=50,
        positional_encoding_projection_scale=10,
        aux_positional_encoding_dim=50,
        aux_positional_encoding_projection_scale=0.1,
        periodic_invariance=True,
        nonlinearity=nn.Tanh,
        final_nonlinearity=nn.Tanh,
        weights_scale=0.1,
        bias=True,
        # Optimization config
        required_img_norm=None,
        required_img_std=None,
        required_img_pixel_min=None,
        required_img_pixel_max=None,
    ):
        # check that input argument are there and valid
        if required_img_std is None and required_img_norm is None:
            raise ValueError("Either std or norm should be provided.")
        elif required_img_std is not None and required_img_norm is not None:
            raise ValueError("Only one of std or norm should be provided.")

        img_normalized_by = "norm" if required_img_norm is not None else "std"

        self.response_predicting_model = response_predicting_model

        input_shape = list(meis_dict.values())[0]["mei"].shape
        channels, height, width = input_shape
        img_res = (height, width)
        template_config = dict(
            img_res=img_res,
            num_templates=1,
            out_channels=channels,
            widths=widths,
            positional_encoding_projection_scale=positional_encoding_projection_scale,
            positional_encoding_dim=positional_encoding_dim,
            aux_positional_encoding_dim=aux_positional_encoding_dim,
            aux_positional_encoding_projection_scale=aux_positional_encoding_projection_scale,
            periodic_invariance=periodic_invariance,
            nonlinearity=nonlinearity,
            final_nonlinearity=final_nonlinearity,
            weights_scale=weights_scale,
            bias=bias,
        )

        if device is None:
            device = resolve_device()

        self.template = INRTemplates(**template_config).to(device)

        # check_meis() #TODO: implement this function
        self.meis_dict = meis_dict

        self.pixel_min = required_img_pixel_min
        self.pixel_max = required_img_pixel_max
        std = required_img_std
        norm = required_img_norm
        mean = (self.pixel_min + self.pixel_max) / 2

        # select img transform
        if img_normalized_by == "std":
            img_transf_config = dict(
                mean=mean, std=std, pixel_min=self.pixel_min, pixel_max=self.pixel_max
            )
            self.img_transforms = StandardizeClip(**img_transf_config).to(device)
        elif img_normalized_by == "norm":
            img_transf_config = dict(
                mean=mean, norm=norm, pixel_min=self.pixel_min, pixel_max=self.pixel_max
            )
            self.img_transforms = FixEnergyNormClip(**img_transf_config).to(device)
        else:
            raise ValueError("img_transform does not match existing names")

    def learn(
        self,
        template_neuron_idx,
        gaussian_blur_sigma=None,
        steps_per_epoch=1,
        grid_points_per_dim=20,
        neighbor_size=0.1,
        temperature=0.3,
        with_periodic_invariances=True,
        reg_decr=0.8,
        min_epochs=10,
        lr=1e-3,
        additional_epochs=0,
        reg_scale=2,
        requirements=None,
        num_max_epochs=1000,
    ):
        device = next(self.template.parameters()).device

        if gaussian_blur_sigma is not None:
            gaussian_blur = GaussianBlur(gaussian_blur_sigma)
        else:
            gaussian_blur = None

        # Dataloader for Training
        dataloader_config = dict(
            num_invariances=1,
            grid_points_per_dim=grid_points_per_dim,
            steps_per_epoch=steps_per_epoch,
        )
        dm = JitteringGridDatamodule(**dataloader_config)
        dm.train_dataloader()
        grid = dm.grid.to(device)

        # Contrastive Regularization Module
        objective_config = dict(
            num_invariances=1,
            grid_points_per_dim=grid_points_per_dim,
            neighbor_size=neighbor_size,
            temperature=temperature,
            with_periodic_invariances=with_periodic_invariances,
            with_round_neighbor=False,
        )
        grid_reg = SimCLROnGrid(**objective_config).to(device)

        # Contrastive Regularization Strength
        reg_scheduler = ParamReduceOnPlateau(
            factor=reg_decr,
            patience=5,
            threshold=0.005,
            mode="max",
        )

        if requirements is None:
            requirements = dict(avg=0.9, std=1.0, necessary_min=0.85)

        optimizer = torch.optim.Adam(self.template.func.parameters(), lr=lr)
        pbar = tqdm(
            range(num_max_epochs), desc="mean activation will appear after one epoch"
        )

        template_neuron_model = SingleNeuronModel(
            self.response_predicting_model, template_neuron_idx
        )

        template_mei_act = self.meis_dict[template_neuron_idx]["activation"]
        template_rf_mask = self.meis_dict[template_neuron_idx]["mask"]

        ignore_scale_scheduler = False
        self.template.train()

        for epoch in pbar:
            for input_grid in dm.train_dataloader():
                input_grid = input_grid.to(device)
                img_pre, img_post, _acts, _ = self.forward(
                    input_grid,
                    self.template,
                    self.img_transforms,
                    template_neuron_model,
                    return_template=True,
                    gb=gaussian_blur,
                )
                acts = _acts / template_mei_act
                within_rf_mask_similarity = grid_reg.reg_term(
                    apply_mask(
                        img_post,
                        torch.from_numpy(template_rf_mask).to(device),
                        self.pixel_min,
                        self.pixel_max,
                    )
                )
                loss = -acts.mean() - reg_scale * within_rf_mask_similarity
                optimizer.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                optimizer.step()

            if not ignore_scale_scheduler:
                if epoch > min_epochs:
                    reg_scale = reg_scheduler.step(
                        metric_to_monitor=acts.mean(),
                        value_to_reduce=reg_scale,
                    )
                    with torch.no_grad():
                        img_pre, img_post, _acts, _ = self.forward(
                            grid,
                            self.template,
                            self.img_transforms,
                            template_neuron_model,
                            return_template=True,
                            gb=gaussian_blur,
                        )
                        acts = _acts / template_mei_act
                    if check_activity_requirements(acts, requirements):
                        print("requirements_satisfied")
                        ignore_scale_scheduler = True
                        last_scheduling_epoch = epoch

            if ignore_scale_scheduler:
                if epoch >= (last_scheduling_epoch + additional_epochs):
                    break

            pbar.set_description(
                f"Act: mean = {acts.mean().item():.2f} min = {acts.min().item():.2f} std = {acts.std().item():.2f} | Contrastive Reg Scale = {reg_scale:.3f} | Epochs without improving = {reg_scheduler.num_epochs_no_improvement}, BEST = {reg_scheduler.best_metric:.2f}"
            )

        self.template.eval()
        return self.template

    def match(
        self,
        template_manifold,
        target_neuron_idx,
        only_affine_coordinate_transformation=True,
        stochastic_coordinate_transformation=False,
        allow_scale_coordinate_transformation=True,
        allow_shear_coordinate_transformation=True,
        uniform_scale_coordinate_transformation=False,
        init_noise_scale_coordinate_transformation=0.1,
        coordinate_transform_clamp_boundaries=True,
    ):
        num_target_neurons = len(target_neuron_idx)
        self.template = template_manifold

        # Initialize Coordinate Transformation
        self.template.initialize_coordinate_transform(
            num_neurons=num_target_neurons,
            only_affine=only_affine_coordinate_transformation,
            stochastic=stochastic_coordinate_transformation,
            init_noise_scale=init_noise_scale_coordinate_transformation,
            allow_scale=allow_scale_coordinate_transformation,
            allow_shear=allow_shear_coordinate_transformation,
            uniform_scale=uniform_scale_coordinate_transformation,
            clamp_boundaries=coordinate_transform_clamp_boundaries,
        )

    def forward(
        self,
        grid,
        image_generator,
        img_transf,
        encoding_model,
        gb=None,
        return_template=False,
        other_neurons_loc_in_list=None,
        other_neurons_loc_in_model=None,
        centralize_template=False,
    ):
        """forward pass throught the pipeline"""
        _img_pre = image_generator(
            aux=grid,
            return_template=return_template,
            centralize_template=centralize_template,
        )
        if return_template:
            *N, C, H, W = _img_pre.shape
        else:
            D, *N, C, H, W = _img_pre.shape
            if other_neurons_loc_in_model is None:
                raise ValueError(
                    "Please pass the indices of neurons that you are fitting the transform for."
                )

        img_pre = _img_pre.flatten(0, -4)

        img_post = img_transf(img_pre)

        if gb is not None:
            img_post.register_hook(gb)

        acts = encoding_model(img_post)

        img_post_hres = img_post

        if return_template:
            return (
                img_pre.reshape(*N, C, H, W),
                img_post.reshape(*N, C, H, W),
                acts.reshape(*N, -1),
                img_post_hres.reshape(*N, C, H, W),
            )
        else:
            D_selected = len(other_neurons_loc_in_model)
            return (
                img_pre.reshape(D, *N, C, H, W)[other_neurons_loc_in_list],
                img_post.reshape(D, *N, C, H, W)[other_neurons_loc_in_list],
                acts[:, other_neurons_loc_in_model].reshape(D, *N, D_selected)[
                    other_neurons_loc_in_list
                ],
                img_post_hres.reshape(D, *N, C, H, W)[other_neurons_loc_in_list],
            )
