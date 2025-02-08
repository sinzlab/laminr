import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F


class ArbitraryNeuronModel(nn.Module):
    def __init__(self, filters, eps=1e-6):
        super().__init__()
        self.register_buffer("filters", torch.from_numpy(filters.astype(np.float32)))
        self.eps = eps

    def forward(self, x):
        out = torch.einsum("bchw,fhw->bf", x, self.filters)
        max_values, _ = out.max(dim=1)
        return (F.elu(max_values) + 1) / 2 + self.eps


class ArbitraryMultiNeuronModel(nn.Module):
    def __init__(self, neuron_models_list):
        super().__init__()
        self.neuron_models_list = nn.ModuleList(neuron_models_list)

    def forward(self, x):
        preds = []
        for model in self.neuron_models_list:
            preds.append(model(x))
        return torch.stack(preds, dim=1)


class ComplexCell(nn.Module):
    def __init__(self, even_filter_np, odd_filter_np):
        super(ComplexCell, self).__init__()
        self.register_buffer(
            "even_filter", torch.tensor(even_filter_np, dtype=torch.float32)
        )
        self.register_buffer(
            "odd_filter", torch.tensor(odd_filter_np, dtype=torch.float32)
        )

    def forward(self, x):
        even_response = torch.sum(x * self.even_filter, dim=[2, 3])
        odd_response = torch.sum(x * self.odd_filter, dim=[2, 3])
        energy = even_response**2 + odd_response**2
        return torch.sqrt(energy)[:, 0]


def sigmoid(x, temp=1.0):
    return 1 / (1 + np.exp(-x / temp))


def transform_pattern(xx, yy, transformation_mat, center_of_transformation):
    p_x, p_y = center_of_transformation
    xx_r = xx - p_x
    yy_r = yy - p_y
    xy_r = np.stack((xx_r, yy_r)).reshape(2, -1)
    xy_rt = transformation_mat @ xy_r
    xx_rt, yy_rt = xy_rt.reshape(2, *xx.shape)
    xx, yy = xx_rt + p_x, yy_rt + p_y

    return xx, yy


def translate_pattern(xx, yy, shift_from_center):
    dx, dy = shift_from_center
    xx, yy = xx - dx, yy - dy
    return xx, yy


def generate_gabor_filter(
    theta,
    width_x,
    width_y,
    frequency,
    phase,
    center,
    grid_size,
    transformation_matrix=None,
):
    center_of_rotation = center
    shift_from_center = center

    x = np.linspace(-1, 1, grid_size[0])
    y = np.linspace(-1, 1, grid_size[1])
    xx, yy = np.meshgrid(x, y)

    # Apply Translation
    xx, yy = translate_pattern(xx, yy, shift_from_center)

    # Apply local rotation
    dx, dy = shift_from_center
    p_x, p_y = center_of_rotation
    new_center_of_rotation = (p_x - dx, p_y - dy)
    rotation_matrix = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    xx, yy = transform_pattern(xx, yy, rotation_matrix, new_center_of_rotation)

    # Apply goabl transformation
    if transformation_matrix is not None:
        dx, dy = shift_from_center
        p_x, p_y = 0, 0  # center of global transformation is the center
        new_center_of_transformation = (p_x - dx, p_y - dy)
        new_center_of_transformation_rotated = (
            rotation_matrix @ new_center_of_transformation
        )
        xx, yy = transform_pattern(
            xx, yy, transformation_matrix, new_center_of_transformation_rotated
        )

    gaussian_maks = np.exp(-0.5 * (xx**2 / width_x**2 + yy**2 / width_y**2))
    spatial_sinudoid = np.cos(2 * np.pi * frequency * xx + phase)
    gabor_filter = gaussian_maks * spatial_sinudoid
    return gabor_filter


def generate_gabor_filter_span_as_grid(center, grid_size, transformation_matrix=None):
    theta = 0
    width_x = 0.2
    width_y = 0.2

    center = np.array(center)

    x = np.linspace(-1, 1, 100) * np.sqrt(width_x)
    y = np.linspace(-1, 1, 100) * np.sqrt(width_y)
    xx, yy = np.meshgrid(x, y)

    # Apply local transformation
    rotation_matrix = np.array(
        [[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]]
    )
    xx, yy = transform_pattern(xx, yy, rotation_matrix, [0, 0])
    xx, yy = translate_pattern(xx, yy, -center)

    # Apply global transformation
    if transformation_matrix is not None:
        inv_transformation_matrix = np.linalg.inv(transformation_matrix)
        xx, yy = transform_pattern(xx, yy, inv_transformation_matrix, [0, 0])

    w, h = grid_size
    ar = h / w

    return xx, yy * ar


### Visualization
def plot_grid_points(xx, yy, ax=None, cmap="inferno", scatter_kws=None):
    if ax is None:
        ax = plt.subplots()[1]
    if scatter_kws is not None:
        ax.scatter(xx.flatten(), yy.flatten(), **scatter_kws)
    else:
        h, w = xx.shape
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx_c, yy_c = np.meshgrid(x, y)
        norm = np.sqrt(((xx_c + 1) / 2) ** 2 + ((yy_c + 1) / 2) ** 2)
        norm = norm / norm.max()
        ax.scatter(xx.flatten(), yy.flatten(), c=norm.flatten(), cmap=cmap)


def plot_grid_border(xx, yy, ax=None, plot_kws=None):
    if ax is None:
        ax = plt.subplots()[1]
    h, w = xx.shape
    first = [0, 0, h - 1, h - 1, 0]
    second = [0, w - 1, w - 1, 0, 0]
    if plot_kws is not None:
        ax.plot(xx[first, second], yy[first, second], **plot_kws)
    else:
        ax.plot(xx[first, second], yy[first, second], c="k", lw=4)


### Generating random transformation matrices
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def scaling_matrix(s_x, s_y):
    return np.array([[s_x, 0], [0, s_y]])


def shearing_matrix(s_h, s_v):
    return np.array([[1, s_h], [s_v, 1]])


def random_transformation_matrices(n):
    matrices = []
    for _ in range(n):
        theta = np.random.uniform(0, 2 * np.pi)  # Random rotation
        s_x = np.random.uniform(0.5, 1.3)  # Random scaling factors
        s_y = s_x + np.random.uniform(-0.3, 0.3)
        s_h = np.random.uniform(-0.15, 0.15)  # Random horizontal shearing
        s_v = np.random.uniform(-0.15, 0.15)  # Random vertical shearing

        rotation = rotation_matrix(theta)
        scaling = scaling_matrix(s_x, s_y)
        shearing = shearing_matrix(s_h, s_v)

        transformation = np.dot(rotation, np.dot(scaling, shearing))
        matrices.append(transformation)

    return matrices
