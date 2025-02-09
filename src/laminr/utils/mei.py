import torch
import featurevis
import featurevis.ops as ops
from featurevis.utils import Compose
from .mask import create_mask
from .general import SingleNeuronModel, resolve_device


def generate_mei(
    model,
    img_res,
    pixel_min,
    pixel_max,
    std=None,
    norm=None,
    gb=None,
    zscore=0.3,
    step_size=10,
    num_iterations=1000,
    print_iters=1001,
):
    model_items = list(model.parameters()) + list(model.buffers())
    device = model_items[0].device

    if std is None and norm is None:
        raise ValueError("Either std or norm should be provided.")
    elif std is not None and norm is not None:
        raise ValueError("Only one of std or norm should be provided.")

    mean = (pixel_min + pixel_max) / 2

    # get image to optimize
    initial_image = torch.randn(1, 1, *img_res, dtype=torch.float32).to(device) + mean

    if std is not None:
        change_stats_transform = ops.ChangeStats(std=std, mean=mean)
    elif norm is not None:
        change_stats_transform = ops.ChangeNorm(norm=norm, mean=mean)

    post_update = Compose([change_stats_transform, ops.ClipRange(pixel_min, pixel_max)])

    initial_image = post_update(initial_image)

    if gb is not None:
        gb = ops.GaussianBlur(gb)

    # optimization of the image
    mei, act, reg_values = featurevis.gradient_ascent(
        model,
        initial_image,
        step_size=step_size,
        num_iterations=num_iterations,
        post_update=post_update,
        gradient_f=gb,
        print_iters=print_iters,
    )

    # Grab the activation of the last optimization step
    mei_act = act[-1]

    # remove batch dimension
    mei = mei[0]
    channels, height, width = mei.shape

    # get mask and it's center from mei (change zscore_thresh to change mask size)
    # mask, mask_center = create_mask(mei, return_mask_centroid=True, zscore_thresh=.5)
    mask, mask_center = create_mask(
        mei.mean(dim=0), return_mask_centroid=True, zscore_thresh=zscore
    )

    return mei.cpu().data.numpy(), mei_act, mask.cpu().data.numpy(), mask_center


def get_mei_dict(
    response_predicting_model,
    neuron_idxs,
    input_shape,
    required_pixel_min,
    required_pixel_max,
    required_std=None,
    required_norm=None,
    gb=None,
    zscore=0.3,  # TODO: think more about this (either user needs to specify this or we need to find a way to calculate this based on other params)
    step_size=10,
    num_iterations=1000,
    print_iters=1001,
    device=None,
):
    if device is None:
        device = resolve_device()

    channels, height, width = input_shape
    img_res = (height, width)
    meis_dict = {}
    for idx, neuron_idx in enumerate(neuron_idxs):
        print(f"neuron_idx = {neuron_idx}: neuron number {idx}/{len(neuron_idxs)}")
        model = SingleNeuronModel(response_predicting_model, neuron_idx).to(device)
        mei, mei_act, mask, mask_center = generate_mei(
            model,
            img_res,
            std=required_std,
            norm=required_norm,
            pixel_min=required_pixel_min,
            pixel_max=required_pixel_max,
            gb=gb,
            zscore=zscore,
            step_size=step_size,
            num_iterations=num_iterations,
            print_iters=print_iters,
        )
        meis_dict[idx] = {
            "mei": mei,
            "activation": mei_act,
            "mask": mask,
            "center_pos": mask_center,
        }
    return meis_dict


def check_meis(
    meis_dict,
    required_pixel_min,
    required_pixel_max,
    required_std=None,
    required_norm=None,
):
    return True
