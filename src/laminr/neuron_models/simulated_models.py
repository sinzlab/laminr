import numpy as np
from .utils.simulated_model import (
    ArbitraryMultiNeuronModel,
    sigmoid,
    generate_gabor_filter,
    ArbitraryNeuronModel,
    ComplexCell,
    generate_gabor_filter_span_as_grid,
)


def neuron1_generator(loc, img_res, transformation_matrix=None, unit_norm=True):
    sigma = 0.13
    sf = 2.5
    gabor_locs = (
        np.array(
            [
                [0.05, 0.05],
                [-0.05, -0.05],
            ]
        )
        + loc
    )
    n_points = 15
    invariance_axis = np.concatenate(
        [np.linspace(-7.5, 7.5, n_points), np.linspace(-7.5, 7.5, n_points)]
    )

    filters = []
    for filter_idx, x in enumerate(invariance_axis):
        gb1 = generate_gabor_filter(
            theta=-np.pi / 2,
            width_x=sigma,
            width_y=sigma,
            frequency=sf,
            phase=np.pi / 2,
            center=gabor_locs[0],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        temp = 2
        phase = (sigmoid(x, temp=temp) * 2 - 1) * np.pi / 2
        if filter_idx >= n_points:
            phase = (sigmoid(x, temp=temp) * 2 + 1) * np.pi / 2
        gb2 = generate_gabor_filter(
            theta=np.pi / 8,
            width_x=sigma,
            width_y=sigma,
            frequency=sf,
            phase=phase,
            center=gabor_locs[1],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        gb = (gb1 + gb2) / 2

        if unit_norm:
            gb = gb / np.linalg.norm(gb)

        filters.append(gb.astype(np.float32))

    xx, yy = generate_gabor_filter_span_as_grid(
        center=np.array(loc),
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )
    grid = xx, yy

    return ArbitraryNeuronModel(np.stack(filters)), grid


def neuron2_generator(loc, img_res, transformation_matrix=None, unit_norm=True, num_filters=30):
    sigma_x = 0.11
    sigma_y = 0.15
    sf = 3
    min_val = 0.9
    max_val = 1.5
    drange = max_val - min_val
    periods = 1

    gabor_locs = np.array([[-0.09, 0.01], [-0.01, -0.05], [0.11, 0.04]]) + loc

    filters = []
    for x in np.linspace(0, 2 * np.pi * periods, periods * num_filters):
        scale1 = (np.sin(x) + 1) / 2 * drange + min_val
        scale2 = (np.sin(x + np.pi / 3) + 1) / 2 * drange + min_val
        scale3 = (np.sin(x + np.pi / 3 * 2) + 1) / 2 * drange + min_val

        gb1 = generate_gabor_filter(
            theta=-np.pi / 8,
            width_x=sigma_x * scale1,
            width_y=sigma_y * scale1,
            frequency=sf / scale1,
            phase=0,
            center=gabor_locs[0],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        gb2 = generate_gabor_filter(
            theta=0,
            width_x=sigma_x * scale2,
            width_y=sigma_y * scale2,
            frequency=sf / scale2,
            phase=0,
            center=gabor_locs[1],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        gb3 = generate_gabor_filter(
            theta=np.pi / 8,
            width_x=sigma_x * scale3,
            width_y=sigma_y * scale3,
            frequency=sf / scale3,
            phase=0,
            center=gabor_locs[2],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        gb = (gb1 + gb2 + gb3) / 3
        # gb = (gb1 + gb3)/3

        if unit_norm:
            gb = gb / np.linalg.norm(gb)

        filters.append(gb.astype(np.float32))

    xx, yy = generate_gabor_filter_span_as_grid(
        center=np.array(loc),
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )
    grid = xx, yy

    return ArbitraryNeuronModel(np.stack(filters)), grid


def neuron3_generator(loc, img_res, transformation_matrix=None, unit_norm=True, num_filters=15):
    sigma_x = 0.13
    sigma_y = 0.13
    sf = 2.5
    gabor_locs = (
        np.array(
            [
                [0.2, 0.0],  # right
                [0.0, 0.2],  # top
                [-0.2, 0.0],  # left
                [0.0, -0.2],  # bottom
            ]
        )
        * 0.7
        + loc
    )
    n_points = num_filters
    invariance_axis = np.concatenate(
        [np.linspace(0, 2 * np.pi, n_points)] * len(gabor_locs)
    )

    filters = []
    for filter_idx, x in enumerate(invariance_axis):
        if (filter_idx >= n_points * 0) & (filter_idx <= n_points * 1):
            theta1 = -np.pi / 2 + x
        else:
            theta1 = -np.pi / 2

        gb1 = generate_gabor_filter(
            theta=theta1,
            width_x=sigma_x,
            width_y=sigma_y,
            frequency=sf,
            phase=np.pi / 2,
            center=gabor_locs[0],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        if (filter_idx >= n_points * 1) & (filter_idx <= n_points * 2):
            theta2 = x
        else:
            theta2 = 0
        gb2 = generate_gabor_filter(
            theta=theta2,
            width_x=sigma_x,
            width_y=sigma_y,
            frequency=sf,
            phase=np.pi / 2,
            center=gabor_locs[1],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        if (filter_idx >= n_points * 2) & (filter_idx <= n_points * 3):
            theta3 = np.pi / 2 + x
        else:
            theta3 = np.pi / 2
        gb3 = generate_gabor_filter(
            theta=theta3,
            width_x=sigma_x,
            width_y=sigma_y,
            frequency=sf,
            phase=np.pi / 2,
            center=gabor_locs[2],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        if (filter_idx >= n_points * 3) & (filter_idx <= n_points * 4):
            theta4 = x
        else:
            theta4 = 0
        gb4 = generate_gabor_filter(
            theta=theta4,
            width_x=sigma_x,
            width_y=sigma_y,
            frequency=sf,
            phase=-np.pi / 2,
            center=gabor_locs[3],
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )

        gb = (gb1 + gb2 + gb3 + gb4) / 4

        if unit_norm:
            gb = gb / np.linalg.norm(gb)

        filters.append(gb.astype(np.float32))

    xx, yy = generate_gabor_filter_span_as_grid(
        center=np.array(loc),
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )
    grid = xx, yy

    return ArbitraryNeuronModel(np.stack(filters)), grid


def phase_invariant_neuron_generator(
    loc, img_res, transformation_matrix=None, unit_norm=True, num_filters=32
):
    sigma_x = 0.14
    sigma_y = 0.14
    sf = 2.3
    phases = np.linspace(0, 2 * np.pi, num_filters, endpoint=False)
    filters = []
    for _, phase in enumerate(phases):
        gb = generate_gabor_filter(
            theta=0,
            width_x=sigma_x,
            width_y=sigma_y,
            frequency=sf,
            phase=phase,
            center=loc,
            grid_size=img_res,
            transformation_matrix=transformation_matrix,
        )
        if unit_norm:
            gb = gb / np.linalg.norm(gb)

        filters.append(gb.astype(np.float32))

    xx, yy = generate_gabor_filter_span_as_grid(
        center=np.array(loc),
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )
    grid = xx, yy

    return ArbitraryNeuronModel(np.stack(filters)), grid


def even_gabor_neuron_generator(
    loc, img_res, transformation_matrix=None, unit_norm=True
):
    sigma_x = 0.14
    sigma_y = 0.14
    sf = 2.3
    gb = generate_gabor_filter(
        theta=0,
        width_x=sigma_x,
        width_y=sigma_y,
        frequency=sf,
        phase=2 * np.pi / 2,
        center=loc,
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )

    if unit_norm:
        gb = gb / np.linalg.norm(gb)

    xx, yy = generate_gabor_filter_span_as_grid(
        center=np.array(loc),
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )
    grid = xx, yy

    return ArbitraryNeuronModel(np.stack(gb[None])), grid


def odd_gabor_neuron_generator(
    loc, img_res, transformation_matrix=None, unit_norm=True
):
    sigma_x = 0.14
    sigma_y = 0.14
    sf = 2.3
    gb = generate_gabor_filter(
        theta=0,
        width_x=sigma_x,
        width_y=sigma_y,
        frequency=sf,
        phase=1 * np.pi / 2,
        center=loc,
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )

    if unit_norm:
        gb = gb / np.linalg.norm(gb)

    xx, yy = generate_gabor_filter_span_as_grid(
        center=np.array(loc),
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )
    grid = xx, yy

    return ArbitraryNeuronModel(np.stack(gb[None])), grid


def complex_neuron_generator(loc, img_res, transformation_matrix=None, unit_norm=True):
    sigma_x = 0.14
    sigma_y = 0.14
    sf = 2.3
    gb1 = generate_gabor_filter(
        theta=0,
        width_x=sigma_x,
        width_y=sigma_y,
        frequency=sf,
        phase=1 * np.pi / 2,
        center=loc,
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )

    sigma_x = 0.14
    sigma_y = 0.14
    gb2 = generate_gabor_filter(
        theta=0,
        width_x=sigma_x,
        width_y=sigma_y,
        frequency=sf,
        phase=2 * np.pi / 2,
        center=loc,
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )

    if unit_norm:
        gb1 = gb1 / np.linalg.norm(gb1)
        gb2 = gb2 / np.linalg.norm(gb2)

    xx, yy = generate_gabor_filter_span_as_grid(
        center=np.array(loc),
        grid_size=img_res,
        transformation_matrix=transformation_matrix,
    )
    grid = xx, yy

    return ComplexCell(gb1, gb2), grid



def simulated(model_type="demo1", img_res=[100, 100]):

    if model_type == "demo1":
        loc = [0.2, .2]
        neuron_model1, _ = phase_invariant_neuron_generator(loc, img_res, num_filters=20)
        loc = [-0.2, -.2]
        t_mat = np.array([[0.70710678, -0.70710678], [0.70710678,  0.70710678]])
        neuron_model2, _ = phase_invariant_neuron_generator(loc, img_res, transformation_matrix=t_mat, num_filters=20)
        loc = [-0.2, .2]
        # neuron_model3, _ = neuron3_generator(loc, img_res)
        neuron_model3, _ = neuron2_generator(loc, img_res, num_filters=60)
        return ArbitraryMultiNeuronModel([neuron_model1, neuron_model2, neuron_model3])
    
    else:
        raise ValueError(f"Model type {model_type} does not exist.")