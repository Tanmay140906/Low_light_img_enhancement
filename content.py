import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix, identity,lil_matrix
from scipy.sparse.linalg import spsolve
from bm3d import bm3d
from typing import Union,Tuple

def bool_image(file_name: str) -> bool:
    """Checks if a file is of 'bmp', 'jpg', 'png' or 'tif' format.

    Returns True if a file name ends with any of these formats, and False otherwise.
    """
    bool_value = file_name[-3:] in ['bmp', 'jpg', 'png', 'tif']
    return bool_value


def brightness_adjust(ill_map: np.ndarray, brightness: Union[int, float]) -> np.ndarray:
    """Adjusts the brightness of the illumination map by adding a constant.

    Returns the shape-(M, N) adjusted illumination map array.
    """
    return ill_map + brightness

def gradient_matrices(illumination_map: np.ndarray) -> Tuple[csr_matrix, csr_matrix]:
    """Generates sparse gradient matrices for computing forward differences in both horizontal and vertical directions.

    Returns the gradient matrices in Compressed Sparse Row (CSR) format.
    """
    image_size = illumination_map.size
    image_x_shape = illumination_map.shape[-1]
    
    dx = lil_matrix((image_size, image_size), dtype=np.float64)
    dy = lil_matrix((image_size, image_size), dtype=np.float64)

    for i in range(image_size - 1):
        if image_x_shape + i < image_size:
            dy[i, i] = -1
            dy[i, image_x_shape + i] = 1
        if (i + 1) % image_x_shape != 0 or i == 0:
            dx[i, i] = -1
            dx[i, i + 1] = 1

    return dx.tocsr(), dy.tocsr()



def partial_derivative(input_matrix: np.ndarray, gradient_matrix: np.ndarray) -> np.ndarray:
    return np.abs(gradient_matrix @ input_matrix.flatten()).reshape(input_matrix.shape)



def laplacian_weight(grad: np.ndarray, size: int, sigma: Union[int, float], epsilon: float) -> np.ndarray:
    radius = int((size - 1) / 2)
    denominator = epsilon + gaussian_filter(np.abs(grad), sigma, mode='constant')
    weights = np.exp(-np.abs(grad)) / denominator

    return weights



def initialize_weights(ill_map: np.ndarray, strategy_n: int, epsilon: float = 0.001) -> np.ndarray:

    if strategy_n == 1:
        weights = np.ones(ill_map.shape)
        weights_x = weights
        weights_y = weights
    elif strategy_n == 2:
        dx, dy = gradient_matrices(ill_map)
        grad_x = partial_derivative(ill_map, dx)
        grad_y = partial_derivative(ill_map, dy)
        weights_x = np.exp(-np.abs(grad_x)) / (epsilon + np.abs(grad_x))
        weights_y = np.exp(-np.abs(grad_y)) / (epsilon + np.abs(grad_y))
    else:
        sigma = 2
        size = 15
        dx, dy = gradient_matrices(ill_map)
        grad_x = partial_derivative(ill_map, dx)
        grad_y = partial_derivative(ill_map, dy)
        weights_x = laplacian_weight(grad_x, size, sigma, epsilon)
        weights_y = laplacian_weight(grad_y, size, sigma, epsilon)

    return weights_x.flatten(), weights_y.flatten()


def optimize_illumination_map(ill_map: np.ndarray, weight_strategy: int = 3) -> np.ndarray:
    """Updates the initial illumination map according to a sped-up solver of the original LIME paper.
    Returns the shape-(M, N) updated illumination map array.
    """
    vectorized_t = ill_map.reshape((ill_map.size, 1))
    epsilon = 0.001
    alpha = 0.15

    d_x_sparse, d_y_sparse = gradient_matrices(ill_map)
    flat_weights_x, flat_weights_y = initialize_weights(ill_map, weight_strategy, epsilon)

    diag_weights_x = diags(flat_weights_x)
    diag_weights_y = diags(flat_weights_y)

    x_term = d_x_sparse.T @ diag_weights_x @ d_x_sparse
    y_term = d_y_sparse.T @ diag_weights_y @ d_y_sparse
    identity_matrix = identity(x_term.shape[0])
    matrix = identity_matrix + alpha * (x_term + y_term)

    updated_t = spsolve(csr_matrix(matrix), vectorized_t)

    return updated_t.reshape(ill_map.shape)


from skimage.restoration import denoise_tv_chambolle
def denoising_tv(image: np.ndarray, cor_ill_map: np.ndarray, weight: float = 0.1) -> np.ndarray:
    # Perform TV denoising
    denoised_image = denoise_tv_chambolle(image, weight=weight)
    
    # Combine the corrected illumination map with the denoised image
    recombined_image = image * cor_ill_map + denoised_image * (1 - cor_ill_map)
    
    # Clip the result to ensure values are in the range [0, 1] and return as float32
    return np.clip(recombined_image, 0, 1).astype("float32")

