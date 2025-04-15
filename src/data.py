import numpy as np
from typing import Tuple, Optional
from skimage.measure import block_reduce
import rasterio


def gaussian_distance_matrix_zero_diag(grid_shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """
    Compute a Gaussian distance matrix for a grid of points with the diagonal set to 0.

    Each entry [i, j] is exp(-d^2 / (2*sigma^2)) for the Euclidean distance d between points i and j.

    Parameters:
        grid_shape (Tuple[int, int]): The shape of the grid (n_rows, n_cols).
        sigma (float): The sigma parameter in the Gaussian function.

    Returns:
        np.ndarray: Gaussian distance matrix with zero diagonal.
    """
    n_rows, n_cols = grid_shape
    y_coords, x_coords = np.indices((n_rows, n_cols))
    points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=-1)
    gauss_matrix = np.exp(-sq_dist / (2 * sigma**2))
    np.fill_diagonal(gauss_matrix, 0)
    return gauss_matrix


def load_fire_data(tiff_file: str,
                   longitude: Optional[Tuple[float, float]] = None,
                   latitude: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Load and optionally crop fire data from a GeoTIFF file.

    Parameters:
        tiff_file (str): Path to the GeoTIFF file.
        longitude (Optional[Tuple[float, float]]): (lon_min, lon_max) bounds.
        latitude (Optional[Tuple[float, float]]): (lat_min, lat_max) bounds.

    Returns:
        np.ndarray: The fire data array, possibly cropped.
    """
    with rasterio.open(tiff_file) as src:
        data = src.read(1)
        if longitude and latitude:
            lon_min, lon_max = longitude
            lat_min, lat_max = latitude

            # Convert geographic coordinates to pixel indices
            top_left = src.index(lon_min, lat_max)
            bottom_right = src.index(lon_max, lat_min)

            row_min, col_min = top_left
            row_max, col_max = bottom_right

            # Ensure correct ordering
            row_min, row_max = sorted([row_min, row_max])
            col_min, col_max = sorted([col_min, col_max])

            cropped_data = data[row_min:row_max, col_min:col_max]
            return cropped_data

        return data

def downsample_fire_data(fire_data: np.ndarray,
                         block_size: Tuple[int, int] = (500, 500),
                         inverted: bool = True) -> np.ndarray:
    """
    Downsample the fire grid using block reduction.

    Parameters:
        fire_data (np.ndarray): Original fire data array.
        block_size (Tuple[int, int]): Block size for downsampling.
        inverted (bool): Whether to invert the downsampled result (251 - value).

    Returns:
        np.ndarray: Downsampled fire data.
    """
    downsampled = block_reduce(fire_data, block_size=block_size, func=np.mean)
    print("Downsampled fire grid shape:", downsampled.shape)
    return 251 - downsampled if inverted else downsampled


def compute_full_matrix(downsampled_fire: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the full matrix combining local fire intensity and influence spread.

    Parameters:
        downsampled_fire (np.ndarray): The downsampled fire data.
        sigma (float): Sigma parameter for the Gaussian distance matrix.

    Returns:
        Tuple containing:
            - gauss_matrix (np.ndarray): Gaussian kernel.
            - fire_vector (np.ndarray): Flattened fire intensities.
            - spread_matrix (np.ndarray): Fire influence across the grid.
            - full_matrix (np.ndarray): Combined fire intensity and influence.
    """
    grid_shape = downsampled_fire.shape
    gauss_matrix = gaussian_distance_matrix_zero_diag(grid_shape, sigma)
    print("Gaussian distance matrix shape:", gauss_matrix.shape)

    fire_vector = downsampled_fire.flatten()
    print("Fire vector shape:", fire_vector.shape)

    spread_matrix = fire_vector[:, np.newaxis] * gauss_matrix
    fire_matrix = np.diag(fire_vector)
    full_matrix = fire_matrix + spread_matrix
    print("Full matrix shape:", full_matrix.shape)

    return gauss_matrix, fire_vector, spread_matrix, full_matrix
