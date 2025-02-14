import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from featurevis.ops import GaussianBlur
import matplotlib.pyplot as plt
from lipstick import GifMaker

from .inr import INRTemplates
from .contrastive_loss import SimCLROnGrid
from .utils.training import (
    JitteringGridDatamodule,
    ParamReduceOnPlateau,
    check_activity_requirements,
    ImprovementChecker,
)
from .utils.img_transforms import StandardizeClip, FixEnergyNormClip
from .utils.mask import apply_mask
from .utils.general import SingleNeuronModel, infer_device
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
        pixel_value_lower_bound=None,
        pixel_value_upper_bound=None,
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
        self.periodic_invariance = periodic_invariance

        if device is None:
            device = infer_device(response_predicting_model)

        self.template = INRTemplates(**template_config).to(device)

        # check_meis() #TODO: implement this function
        self.meis_dict = meis_dict

        self.pixel_min = pixel_value_lower_bound
        self.pixel_max = pixel_value_upper_bound
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
        steps_per_epoch=50,
        grid_points_per_dim=20,
        neighbor_size=0.1,
        temperature=0.3,
        reg_decr=0.8,
        min_epochs=10,
        lr=1e-3,
        additional_epochs=0,
        reg_scale=2,
        requirements=None,
        num_max_epochs=1000,
        verbose=False,
    ):
        device = next(self.template.parameters()).device
        self.template_neuron_idx = template_neuron_idx

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
            with_periodic_invariances=self.periodic_invariance,
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
            requirements = dict(avg=.99, std=1., necessary_min=0.98)

        optimizer = torch.optim.Adam(self.template.func.parameters(), lr=lr)

        template_neuron_model = SingleNeuronModel(
            self.response_predicting_model, template_neuron_idx
        )

        template_mei_act = self.meis_dict[template_neuron_idx]["activation"]
        template_rf_mask = self.meis_dict[template_neuron_idx]["mask"]

        ignore_scale_scheduler = False
        self.template.train()
        pbar = tqdm(range(num_max_epochs))
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
            
            act_desc = f"Act mean = {acts.mean().item():.2f} (min = {acts.min().item():.2f} std = {acts.std().item():.2f})"
            desc = act_desc
            if verbose:
                cont_desc = f"Contrastive Reg Scale = {reg_scale:.3f}"
                improve_desc = f"Epochs without improving = {reg_scheduler.num_epochs_no_improvement} (patience = {reg_scheduler.patience})"
                desc = desc + " | " + cont_desc + " | " + improve_desc

            pbar.set_description(desc)

        self.template.eval()

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

        images_on_template_manifold = img_post.cpu().data.numpy()
        template_neuron_activations = _acts.squeeze().cpu().data.numpy()

        return images_on_template_manifold, template_neuron_activations

    def match(
        self,
        target_neuron_idxs,
        grid_points_per_dim=20,
        steps_per_epoch=1,
        lr=1e-3,
        num_epochs=1000,
        patience=15,
        ignore_diff_smaller_than=1e-3,
        rotate_angle_and_scale=False,
        find_best_translation=False,
        only_affine_coordinate_transformation=True,
        stochastic_coordinate_transformation=False,
        allow_scale_coordinate_transformation=True,
        allow_shear_coordinate_transformation=True,
        uniform_scale_coordinate_transformation=False,
        init_noise_scale_coordinate_transformation=0.1,
        coordinate_transform_clamp_boundaries=True,
        verbose=False,
    ):
        num_target_neurons = len(target_neuron_idxs)
        self.target_neuron_idxs = target_neuron_idxs
        device = next(self.template.parameters()).device

        rf_positions = {neuron_idx: np.array([value["center_pos"]["x"], value["center_pos"]["y"]]).astype(np.float32) for neuron_idx, value in self.meis_dict.items()}
    
        template_rf_mask = self.meis_dict[self.template_neuron_idx]["mask"]
        template_rf_location = rf_positions[self.template_neuron_idx]
        others_rf_mask = np.array([self.meis_dict[idx]["mask"] for idx in target_neuron_idxs])
        others_rf_location = np.array([rf_positions[neuron_idx] for neuron_idx in target_neuron_idxs])
        others_mei_act = np.array([self.meis_dict[idx]["activation"] for idx in target_neuron_idxs])
        others_mei_act = torch.from_numpy(others_mei_act).to(device)
        other_neurons_loc_in_model = target_neuron_idxs
        other_neurons_loc_in_list = np.arange(len(other_neurons_loc_in_model))
       
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

        self.template.register_coords_shifts(
            torch.from_numpy(template_rf_location).to(device), 
            torch.from_numpy(others_rf_location).to(device),
            )

        dataloader_config = dict(
            num_invariances=1,
            grid_points_per_dim=grid_points_per_dim,
            steps_per_epoch=steps_per_epoch,
        )
        grid_dataloader = JitteringGridDatamodule(**dataloader_config)

        template = self.initialize_with_best_angle_and_translation(
            self.template,
            self.response_predicting_model,
            self.img_transforms,
            grid_dataloader,
            other_neurons_loc_in_list,
            other_neurons_loc_in_model,
            template_rf_mask=template_rf_mask,
            others_rf_mask=others_rf_mask,
            rotate_angle_and_scale=rotate_angle_and_scale,
            find_best_translation=find_best_translation,
        )

        optimizer = torch.optim.Adam(template.coordinate_transform.parameters(), lr=lr)
        improvement_checker = ImprovementChecker(
            patience=patience, ignore_diff_smaller_than=ignore_diff_smaller_than
        )
        pbar = tqdm(range(num_epochs))

        # Training Loop
        for epoch in pbar:
            template.train()
            for input_grid in grid_dataloader.train_dataloader():
                input_grid = input_grid.to(device)
                img_pre, img_post, _acts, _ = self.forward(
                    input_grid,
                    template,
                    self.img_transforms,
                    self.response_predicting_model,
                    return_template=False,
                    other_neurons_loc_in_list=other_neurons_loc_in_list,
                    other_neurons_loc_in_model=other_neurons_loc_in_model,
                )
                relevant_acts = _acts[range(len(self.target_neuron_idxs)), :, range(len(self.target_neuron_idxs))]
                acts = relevant_acts.mean(dim=1) / others_mei_act
                loss = -acts.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            is_increasing = improvement_checker.is_increasing(acts.mean().item())
            act_desc = f"Activation mean = {acts.mean().item():.2f} (min = {acts.min().item():.2f} std = {acts.std().item():.2f})"
            desc = act_desc
            if verbose:
                improve_desc = f"Epochs without improving = {improvement_checker.has_not_improved_counter} (patience: {improvement_checker.patience})"
                desc = desc + " | " + improve_desc
            pbar.set_description(desc)

            if not is_increasing:
                break

        template.eval()

        with torch.no_grad():
            img_pre, img_post, _acts, _ = self.forward(
                input_grid,
                template,
                self.img_transforms,
                self.response_predicting_model,
                return_template=False,
                other_neurons_loc_in_list=other_neurons_loc_in_list,
                other_neurons_loc_in_model=other_neurons_loc_in_model,
            )
            relevant_acts = _acts[range(len(self.target_neuron_idxs)), :, range(len(self.target_neuron_idxs))]

        images_on_aligned_manifolds = img_post.cpu().data.numpy()
        target_neurons_activations = relevant_acts.cpu().data.numpy()

        return images_on_aligned_manifolds, target_neurons_activations

    def save_learned_template_as_gif(self, n_images=20, name=None, cmap='Greys_r', fig_kws=None):
        fig_kws = fig_kws if fig_kws is not None else {}
        images_from_learned_template = self.get_images_from_learned_template(n_images=n_images)
        name = name if name is not None else f"learned_template_{self.template_neuron_idx}"
        with GifMaker(name) as g:
            for img in images_from_learned_template:
                fig, ax = plt.subplots()
                vmax = np.abs(img).max()
                vmin = -vmax
                ax.imshow(img[0], vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set(xticks=[], yticks=[])
                g.add(fig)
        return g
    
    def save_matched_template_as_gif(self, target_neuron_idx, n_images=20, name=None, cmap='Greys_r', fig_kws=None):
        fig_kws = fig_kws if fig_kws is not None else {}
        images_from_matched_template = self.get_images_from_matched_template(target_neuron_idx, n_images=n_images)
        name = name if name is not None else f"learned_template_{self.template_neuron_idx}_matched_to_{target_neuron_idx}"
        with GifMaker(name) as g:
            for img in images_from_matched_template:
                fig, ax = plt.subplots(**fig_kws)
                vmax = np.abs(img).max()
                vmin = -vmax
                ax.imshow(img[0], vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set(xticks=[], yticks=[])
                g.add(fig)
        return g
    
    def get_images_from_learned_template(self, n_images=20):
        device = next(self.template.parameters()).device
        dataloader_config = dict(
            num_invariances=1,
            grid_points_per_dim=n_images,
            steps_per_epoch=1,
        )
        dm = JitteringGridDatamodule(**dataloader_config)
        dm.train_dataloader()
        grid = dm.grid.to(device)
        with torch.no_grad():
            return self.get_images(
                grid,
                self.template,
                self.img_transforms,
                return_template=True,
            ).cpu().data.numpy()
    
    def get_images_from_matched_template(self, target_neuron_idx, n_images=20):
        if not isinstance(target_neuron_idx, int):
            raise ValueError("target_neuron_idx should be an integer, referring to a single target neuron.")
        if target_neuron_idx not in self.target_neuron_idxs:
            raise ValueError(f"target_neuron_idx should be one of the target neuron indices: {self.target_neuron_idxs}")
        target_neuron_idx_from_zero = np.where(target_neuron_idx == np.array(self.target_neuron_idxs))[0][0]

        device = next(self.template.parameters()).device
        dataloader_config = dict(
            num_invariances=1,
            grid_points_per_dim=n_images,
            steps_per_epoch=1,
        )
        dm = JitteringGridDatamodule(**dataloader_config)
        dm.train_dataloader()
        grid = dm.grid.to(device)

        channels, height, width = self.meis_dict[self.template_neuron_idx]["mei"].shape
        with torch.no_grad():
            images = self.get_images(
                grid,
                self.template,
                self.img_transforms,
                return_template=False,
            ).reshape(len(self.target_neuron_idxs), n_images, channels, height, width)[target_neuron_idx_from_zero]
            return images.cpu().data.numpy()

    def get_images(
            self, 
            grid,
            image_generator,
            img_transf,
            return_template=False,
            centralize_template=False,
            return_pre=False,
            ):
        
        _img_pre = image_generator(
            aux=grid,
            return_template=return_template,
            centralize_template=centralize_template,
        )

        img_pre = _img_pre.flatten(0, -4)
        img_post = img_transf(img_pre)

        if return_pre:
            return _img_pre, img_post
        else: 
            return img_post
        
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
        _img_pre, img_post = self.get_images(
            grid,
            image_generator,
            img_transf,
            return_template=return_template,
            centralize_template=centralize_template,
            return_pre=True,
            )
        
        if return_template:
            *N, C, H, W = _img_pre.shape
        else:
            D, *N, C, H, W = _img_pre.shape

        if gb is not None:
            img_post.register_hook(gb)

        acts = encoding_model(img_post)

        img_post_hres = img_post
        img_pre = _img_pre.flatten(0, -4)
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

    def initialize_with_best_angle_and_translation(
        self,
        template, 
        encoding_model, 
        img_transforms,
        grid_dataloader,  
        other_neurons_loc_in_list, 
        other_neurons_loc_in_model, 
        template_rf_mask=None, 
        others_rf_mask=None, 
        rotate_angle_and_scale=False,
        find_best_translation=False
    ):
        device = next(template.parameters()).device
        if rotate_angle_and_scale:
            if others_rf_mask is None:
                raise ValueError("others_rf_mask must be provided if rotate_angle_and_scale is True")
            # Initialize the transformation at best rotation angle (and scale) for each target neuron
            mask_size_fraction = others_rf_mask.sum(axis=(1, 2)) / template_rf_mask.sum()
            mask_size_fraction = torch.from_numpy(mask_size_fraction).to(device)

        angles = np.linspace(0, 2*np.pi, int(360/6 + 1))[:-1].astype(np.float32)

        if find_best_translation:
            translations = np.linspace(-.1, .1, 11).astype(np.float32)  # Define a range for translations
            desc = 'Finding best angle and translation for initalization'
        else:
            translations = np.array([0.0]).astype(np.float32)
            desc = 'Finding best angle for initalization'

        # Initialize a 4D array to store activations
        activations_for_transformations = np.zeros((len(angles), len(translations), len(translations), len(other_neurons_loc_in_list)))

        # Try different angles and translations to find the best combination for initialization
        with torch.no_grad():
            input_grid = grid_dataloader.grid.to(device)
            img_posts = []
            for i, angle in enumerate(tqdm(angles, desc=desc)):
                for j, tx in enumerate(translations):
                    for k, ty in enumerate(translations):
                        template.coordinate_transform.transforms.Affine.angles.data[other_neurons_loc_in_list] =  torch.ones_like(template.coordinate_transform.transforms.Affine.angles.data[other_neurons_loc_in_list]) * angle
                        if rotate_angle_and_scale:
                            template.coordinate_transform.transforms.Affine.scalings.data[other_neurons_loc_in_list] = torch.ones_like(template.coordinate_transform.transforms.Affine.scalings.data[other_neurons_loc_in_list]) * 1/mask_size_fraction.reshape(-1, 1)

                        template.coordinate_transform.transforms.Affine.translation.data[other_neurons_loc_in_list, 0, 0] = torch.ones_like(template.coordinate_transform.transforms.Affine.translation.data[other_neurons_loc_in_list, 0, 0]) * tx
                        template.coordinate_transform.transforms.Affine.translation.data[other_neurons_loc_in_list, 0, 1] = torch.ones_like(template.coordinate_transform.transforms.Affine.translation.data[other_neurons_loc_in_list, 0, 1]) * ty

                        img_pre, img_post, _acts, _ = self.forward(input_grid, template, img_transforms, encoding_model, return_template=False, 
                                                            other_neurons_loc_in_list=other_neurons_loc_in_list, other_neurons_loc_in_model=other_neurons_loc_in_model)
                        # relevant_acts = torch.diag(_acts.mean(dim=1)).cpu().data.numpy()  # Get activations for each neuron
                        relevant_acts = _acts[range(len(self.target_neuron_idxs)), :, range(len(self.target_neuron_idxs))]
                        relevant_acts = relevant_acts.mean(dim=1).cpu().data.numpy()

                        # Store activations in the 4D array
                        activations_for_transformations[i, j, k, :] = relevant_acts
                        img_posts.append(img_post.cpu().data.numpy())

        # Find the combination of angle, tx, and ty that maximizes the activation for each neuron
        max_indices = np.argmax(activations_for_transformations.reshape(-1, activations_for_transformations.shape[-1]), axis=0)
        best_angle_indices, best_tx_indices, best_ty_indices = np.unravel_index(max_indices, activations_for_transformations.shape[:3])
        best_angles = angles[best_angle_indices]  # Best angles for each neuron
        best_txs = translations[best_tx_indices]  # Best tx for each neuron
        best_tys = translations[best_ty_indices]  # Best ty for each neuron

        # Set the optimal angle and translation for each neuron
        template.coordinate_transform.transforms.Affine.angles.data[other_neurons_loc_in_list] = torch.from_numpy(best_angles).to(device)
        template.coordinate_transform.transforms.Affine.translation.data[other_neurons_loc_in_list, 0, 0] = torch.from_numpy(best_txs).to(device)
        template.coordinate_transform.transforms.Affine.translation.data[other_neurons_loc_in_list, 0, 1] = torch.from_numpy(best_tys).to(device)

        if rotate_angle_and_scale:
            template.coordinate_transform.transforms.Affine.scalings.data[other_neurons_loc_in_list] = torch.ones_like(template.coordinate_transform.transforms.Affine.scalings.data[other_neurons_loc_in_list]) * 1/mask_size_fraction.reshape(-1, 1)
            
        return template
