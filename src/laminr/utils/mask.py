import numpy as np
import torch
from scipy import ndimage
from skimage import morphology


def rescale(x, in_min, in_max, out_min, out_max):
    in_mean = (in_min + in_max) / 2
    out_mean = (out_min + out_max) / 2
    in_ext = in_max - in_min
    out_ext = out_max - out_min
    gain = out_ext / in_ext
    x_rescaled = (x - in_mean) * gain + out_mean
    return x_rescaled


def apply_mask(mei, mask, pixel_min, pixel_max):
    mei = rescale(mei, pixel_min, pixel_max, -1, 1)
    mask = mask.repeat(len(mei), 1, 1, 1)
    mask = mask.reshape(mei.shape)
    mei_masked = rescale(mei * mask, -1, 1, pixel_min, pixel_max)
    return mei_masked


def create_mask(
    mei,
    zscore_thresh=1.5,
    closing_iters=2,
    gaussian_sigma=1.5,
    return_mask_centroid=False,
):
    """
    Creates a mask for an image using z-score thresholding, binary closing, and Gaussian filtering.

    Parameters:
    - mei (Tensor): The input image (Multi-Exposure Image) to be processed.
    - zscore_thresh (float, optional): The threshold for z-score to identify significant deviations. Default is 1.5.
    - closing_iters (int, optional): Number of iterations for binary closing to close small holes in the mask. Default is 2.
    - gaussian_sigma (float, optional): Sigma value for Gaussian filtering to smooth the mask. Default is 1.5.
    - return_mask_centroid (bool, optional): If True, returns the centroid of the mask along with the mask. Default is False.

    Returns:
    - mask (Tensor): The generated mask for the image.
    - (mask_x, mask_y) (tuple, optional): The centroid of the mask (returned if return_mask_centroid is True).
    """

    # Compute mask centroid
    mei_np = mei.cpu().squeeze().numpy()
    norm_mei_np = (mei_np - mei_np.mean()) / mei_np.std()
    thresholded = (np.abs(norm_mei_np) > zscore_thresh).squeeze()
    closed = ndimage.binary_closing(thresholded, iterations=closing_iters)
    labeled = morphology.label(closed, connectivity=2)
    most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
    oneobject = labeled == most_frequent
    hull = morphology.convex_hull_image(oneobject)
    smoothed = ndimage.gaussian_filter(hull.astype(np.float32), sigma=gaussian_sigma)
    mask = torch.Tensor(smoothed)
    # Compute mask centroid
    if return_mask_centroid:
        py, px = (coords.mean() + 0.5 for coords in np.nonzero(hull))
        # mask_y, mask_x = px_y - mask.shape[0] / 2, px_x - mask.shape[1] / 2,
        return mask, {"y": py, "x": px}
    else:
        return mask
