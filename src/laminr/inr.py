from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from .coordinate_transforms import CoordinateTransform


class INRTemplates(nn.Module):
    def __init__(
        self,
        img_res,
        num_neurons=0,
        num_templates=1,
        out_channels=1,
        aux_dim=1,
        # CPPN-related arguments
        widths=[15] * 8,
        periodic_invariance=False,
        positional_encoding_dim=None,
        positional_encoding_projection_scale=1.0,
        aux_positional_encoding_dim=None,
        aux_positional_encoding_projection_scale=1.0,
        nonlinearity=nn.Tanh,
        final_nonlinearity=nn.Sigmoid,
        bias=False,
        batchnorm=False,
        weights_scale=1.0,
        jitter_coords=False,
        jitter_coords_scale=1.0,
    ):
        super().__init__()
        self.setup_attributes(
            num_neurons,
            img_res,
            num_templates,
            out_channels,
            aux_dim,
            batchnorm,
            weights_scale,
            periodic_invariance,
            positional_encoding_dim,
            positional_encoding_projection_scale,
            aux_positional_encoding_dim,
            aux_positional_encoding_projection_scale,
            # coordinate_transform_clamp_boundaries,
            jitter_coords,
            jitter_coords_scale,
        )
        self.build_network(widths, nonlinearity, final_nonlinearity, bias)
        self.initialize_buffers()
        self.apply(self.weights_init)

    def forward(
        self, img_res=None, aux=None, return_template=False, centralize_template=False
    ):
        return_template = True if self.num_neurons == 0 else return_template
        device = next(self.parameters()).device
        coords = self.get_coordinates(img_res, device)
        if img_res is None:
            img_res = self.img_res

        coords_transformed = self.apply_coordinate_transform(
            coords, return_template, centralize_template
        )
        aux_transformed = self.transform_auxiliary_input(aux, device)
        coords_transformed_pe, aux_transformed_pe = self.apply_positional_encoding(
            coords_transformed, aux_transformed
        )

        # self.template_input.shape -> T
        # aux_transformed_pe.shape -> N x 1 (or anything depending on the positional encoding)
        # coords_transformed_pe.shape -> D x HW x 2 (or anything depending on the positional encoding)
        # T: num_templates, N: num_images, D: num_neurons, HW: flattened spatial dims

        inputs = combine_multiple_tensors(
            self.template_input, aux_transformed_pe, coords_transformed_pe
        )
        T, N, D, HW = tensors_coordinate_dims(
            self.template_input, aux_transformed_pe, coords_transformed_pe
        )

        assert HW == np.prod(img_res)
        out = (
            self.func(inputs)
            .reshape(T, N, D, *img_res, self.out_channels)
            .permute(0, 2, 1, 5, 3, 4)
        )

        output_aux_dim_unpacked = [round(len(aux) ** (1 / self.aux_dim))] * self.aux_dim

        if return_template:
            return (
                out.squeeze(0)
                .squeeze(0)
                .reshape(*output_aux_dim_unpacked, self.out_channels, *img_res)
            )
        else:
            return out.squeeze(0).reshape(
                D, *output_aux_dim_unpacked, self.out_channels, *img_res
            )

    def setup_attributes(
        self,
        num_neurons,
        img_res,
        num_templates,
        out_channels,
        aux_dim,
        batchnorm,
        weights_scale,
        periodic_invariance,
        positional_encoding_dim,
        positional_encoding_projection_scale,
        aux_positional_encoding_dim,
        aux_positional_encoding_projection_scale,
        # coordinate_transform_clamp_boundaries,
        jitter_coords,
        jitter_coords_scale,
    ):
        self.img_res = img_res
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.num_neurons = num_neurons
        self.num_templates = num_templates
        self.aux_dim = aux_dim
        self.template_dim = 1
        self.weights_scale = weights_scale
        self.periodic_invariance = periodic_invariance
        self.positional_encoding_dim = positional_encoding_dim
        self.positional_encoding_projection_scale = positional_encoding_projection_scale
        self.aux_positional_encoding_dim = aux_positional_encoding_dim
        self.aux_positional_encoding_projection_scale = (
            aux_positional_encoding_projection_scale
        )
        # self.coordinate_transform_clamp_boundaries = (
        #     coordinate_transform_clamp_boundaries
        # )
        self.jitter_coords = jitter_coords
        self.jitter_coords_scale = jitter_coords_scale
        self.calculate_input_dimensions()

    def calculate_input_dimensions(self):
        if self.positional_encoding_dim is not None:
            self.in_dim = 2 * self.positional_encoding_dim + self.template_dim
        else:
            self.in_dim = 2 + self.template_dim

        if self.aux_positional_encoding_dim is not None:
            self.in_dim += self.aux_positional_encoding_dim * 2
        else:
            if self.periodic_invariance:
                self.in_dim += self.aux_dim * 2
            else:
                self.in_dim += self.aux_dim

    def build_network(self, widths, nonlinearity, final_nonlinearity, bias):
        layers = len(widths)
        in_widths = [self.in_dim] + widths[:-1]
        elements = []
        for i in range(layers):
            elements.append(
                (f"layer{i}", nn.Linear(in_widths[i], widths[i], bias=bias))
            )
            if self.batchnorm:
                elements.append((f"batchnorm{i}", nn.BatchNorm1d(widths[i])))
            if nonlinearity is not None:
                elements.append((f"nonlinearity{i}", nonlinearity()))

        elements.append(
            (f"layer{layers}", nn.Linear(widths[-1], self.out_channels, bias=bias))
        )
        elements.append((f"nonlinearity{layers}", final_nonlinearity()))

        self.func = nn.Sequential(OrderedDict(elements))

    def initialize_buffers(self):
        if self.positional_encoding_dim is not None:
            self.register_buffer(
                "B",
                torch.randn(2, self.positional_encoding_dim)
                * self.positional_encoding_projection_scale,
            )
        if self.aux_positional_encoding_dim is not None:
            auxB_shape = (
                (2 * self.aux_dim, self.aux_positional_encoding_dim)
                if self.periodic_invariance
                else (1 * self.aux_dim, self.aux_positional_encoding_dim)
            )
            self.register_buffer(
                "auxB",
                torch.randn(auxB_shape) * self.aux_positional_encoding_projection_scale,
            )

        coords = torch.meshgrid(
            [torch.linspace(-1, 1, linres) for linres in self.img_res], indexing="ij"
        )
        coords = torch.stack(coords, dim=-1)
        self.register_buffer("coords", coords)

        template_input = (
            torch.linspace(-1, 1, self.num_templates)
            if self.num_templates > 1
            else torch.zeros(1)
        )
        self.register_buffer("template_input", template_input)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, self.weights_scale)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def initialize_coordinate_transform(
        self,
        num_neurons,
        only_affine,
        stochastic,
        init_noise_scale,
        allow_scale,
        allow_shear,
        uniform_scale,
        clamp_boundaries,
    ):
        device = next(self.parameters()).device
        self.num_neurons = num_neurons
        self.coordinate_transform = CoordinateTransform(
            self.num_neurons,
            only_affine=only_affine,
            stochastic=stochastic,
            init_noise_scale=init_noise_scale,
            allow_scale=allow_scale,
            allow_shear=allow_shear,
            uniform_scale=uniform_scale,
        ).to(device)
        self.coordinate_transform_clamp_boundaries = clamp_boundaries

    def reset_coordinate_transform(self, num_neurons=None):
        device = next(self.parameters()).device #TODO: use infer_device
        if num_neurons is not None:
            self.num_neurons = num_neurons
        else:
            num_neurons = self.coordinate_transform.n_dim
        only_affine = self.coordinate_transform.only_affine
        stochastic = self.coordinate_transform.stochastic
        init_noise_scale = self.coordinate_transform.init_noise_scale
        allow_scale = self.coordinate_transform.allow_scale
        allow_shear = self.coordinate_transform.allow_shear
        uniform_scale = self.coordinate_transform.uniform_scale

        self.coordinate_transform = CoordinateTransform(
            self.num_neurons,
            only_affine=only_affine,
            stochastic=stochastic,
            init_noise_scale=init_noise_scale,
            allow_scale=allow_scale,
            allow_shear=allow_shear,
            uniform_scale=uniform_scale,
        ).to(device)

    def get_coords_shift_to_center(self, template_rf_center, img_res=None):
        template_rf_center = template_rf_center.flip(dims=[-1])
        device = next(self.parameters()).device
        img_res = img_res if img_res is not None else self.img_res
        img_center_coord = torch.tensor(img_res).to(device) / 2
        coords_shift = (
            template_rf_center.to(device) - img_center_coord
        ) / img_center_coord
        return coords_shift

    def get_coords_shift_from_center(self, template_rf_center, img_res=None):
        shift_to_center = self.get_coords_shift_to_center(
            template_rf_center, img_res=img_res
        )
        return -1 * shift_to_center

    def get_template_center_in_coords_space(self, template_rf_center, img_res=None):
        template_rf_center = template_rf_center.flip(dims=[-1])
        device = next(self.parameters()).device
        img_res = img_res if img_res is not None else self.img_res
        img_max_coord = torch.tensor(img_res).to(device)
        template_center = template_rf_center.to(device) / img_max_coord * 2 - 1
        return template_center

    def register_coords_shifts(self, template_rf_center, target_rf_centers):
        self.register_buffer(
            "coords_shift_to_center",
            self.get_coords_shift_to_center(template_rf_center),
        )
        self.register_buffer(
            "template_center_in_coords_space",
            self.get_template_center_in_coords_space(template_rf_center),
        )
        self.register_buffer(
            "coords_shift_from_center",
            self.get_coords_shift_from_center(target_rf_centers),
        )

    def get_coordinates(self, img_res, device):
        if img_res is None:
            coords = self.coords
            img_res = self.img_res
        else:
            coords = torch.stack(
                torch.meshgrid(
                    [torch.linspace(-1, 1, shape, device=device) for shape in img_res],
                    indexing="ij",
                ),
                dim=-1,
            )
        if self.jitter_coords and self.training:  # only jitter during training
            coords_noise = (torch.rand_like(coords) - 0.5) * torch.tensor(
                [[[2 / img_res[0], 2 / img_res[1]]]]
            ).to(device)
            coords = coords + coords_noise * self.jitter_coords_scale
        return coords

    def apply_coordinate_transform(self, coords, return_template, centralize_template):
        """
        Three important steps here:
        1. Get grid points relative to the center of rotation (template's center) -> this is necessary to apply a local transformation (other than translation)
        2. Apply the local transformation (e.g. rotation) per neuron
        3. Locally transformed coordinates are then shifted to the target neurons' RF location
        """
        coords_shift_to_center = getattr(
            self,
            "coords_shift_to_center",
            torch.zeros(self.num_templates, 2).to(coords.device),
        )
        coords_shifted_to_center = coords.flatten(end_dim=-2) + coords_shift_to_center
        if return_template:
            return (
                coords_shifted_to_center.unsqueeze(0)
                if centralize_template
                else coords.flatten(end_dim=-2).unsqueeze(0)
            )
        else:
            template_center = getattr(
                self,
                "template_center_in_coords_space",
                torch.zeros(self.num_templates, 2).to(coords.device),
            )
            coords_relative_to_template_center = (
                coords.flatten(end_dim=-2) - template_center
            )  # center the grid on the location of interest
            coords_transformed = (
                self.coordinate_transform(coords_relative_to_template_center)
                + template_center
            )  # transform back to original location

            # Apply the final translation to brind the transformed grid points at the location of neurons' RF
            with torch.no_grad():
                coords_shift_from_center = getattr(
                    self,
                    "coords_shift_from_center",
                    torch.zeros(self.num_neurons, 2).to(coords.device),
                )
                coords_shift_from_center_adjusted = coords_shift_from_center.unsqueeze(
                    1
                ) + coords_shift_to_center.unsqueeze(0)
                coords_shift_from_center_transformed = self.coordinate_transform(
                    coords_shift_from_center_adjusted
                )

            coords_transformed_shifted_to_target = (
                coords_transformed + coords_shift_from_center_transformed
            )
            if self.coordinate_transform_clamp_boundaries:
                coords_transformed_shifted_to_target.data.clamp_(-1, 1)
            return coords_transformed_shifted_to_target

    def transform_auxiliary_input(self, aux, device):
        if aux is None:
            aux = torch.tensor([[0.0]]).to(device)
        if len(aux.shape) == 1:
            aux = aux.unsqueeze(1)
        # aux = combine_multiple_tensors(*aux.T)
        if self.periodic_invariance:
            aux_transformed = torch.cat(
                [torch.sin(aux), torch.cos(aux)],
                dim=1,
            )
        else:
            aux_transformed = (
                aux / np.pi - 1
            )  # the assumption is that the aux is between 0 and 2*pi
        return aux_transformed

    def apply_positional_encoding(self, coords_transformed, aux_transformed):
        # Apply positional encoding to coordinates
        if self.positional_encoding_dim is not None:
            coords_transformed_pe = torch.cat(
                [
                    torch.sin(coords_transformed @ self.B),
                    torch.cos(coords_transformed @ self.B),
                ],
                dim=-1,
            )
        else:
            coords_transformed_pe = coords_transformed

        # Apply positional encoding to auxiliary input
        if self.aux_positional_encoding_dim is not None:
            aux_transformed_pe = torch.cat(
                [
                    torch.sin(aux_transformed @ self.auxB),
                    torch.cos(aux_transformed @ self.auxB),
                ],
                dim=-1,
            )
        else:
            aux_transformed_pe = aux_transformed

        return coords_transformed_pe, aux_transformed_pe


def tensor_coordinate_dims(tensor):
    dims = len(tensor.shape)
    if dims == 1:
        return [tensor.shape[0]]
    else:
        return list(tensor.shape[:-1])


def tensors_coordinate_dims(*tensors):
    return [dim for tensor in tensors for dim in tensor_coordinate_dims(tensor)]


def reshape_tensor(tensor):
    dims = len(tensor.shape)
    if dims == 1:
        return tensor.view(-1, 1)
    else:
        return tensor.reshape(-1, tensor.shape[-1])


def cartesian_product(A, B):
    m, n = A.shape
    p, q = B.shape
    A = A[:, None, :]
    B = B[None, :, :]
    A_comb = A.repeat(1, p, 1)
    B_comb = B.repeat(m, 1, 1)
    C = torch.cat((A_comb, B_comb), dim=-1)
    return C.reshape(m * p, n + q)


def combine_multiple_tensors(*tensors):
    reshaped_tensors = [reshape_tensor(tensor) for tensor in tensors]
    result = reshaped_tensors[0]
    for T in reshaped_tensors[1:]:
        result = cartesian_product(result, T)
    return result
