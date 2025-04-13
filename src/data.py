import numpy as np
from typing import Tuple
from skimage.measure import block_reduce
import rasterio

def gaussian_distance_matrix_zero_diag(grid_shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """
    Compute a Gaussian distance matrix for a grid of points with the diagonal set to 0.
    Each entry [i, j] is exp(-d^2 / (2*sigma^2)) for the Euclidean distance d between points i and j,
    with the diagonal entries explicitly set to 0.
    
    Parameters:
        grid_shape (tuple): The shape of the grid (n_rows, n_cols).
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

def load_fire_data(tiff_file: str) -> np.ndarray:
    """
    Load fire data from a GeoTIFF file.
    
    Parameters:
        tiff_file (str): Path to the GeoTIFF file.
        
    Returns:
        np.ndarray: The loaded fire data array.
    """
    with rasterio.open(tiff_file) as src:
        data = src.read(1)
    print("Original fire data shape:", data.shape)
    return data

def downsample_fire_data(fire_data: np.ndarray, block_size: Tuple[int, int]=(500, 500), slice_shape: Tuple[int, int]=(3, 3)) -> np.ndarray:
    """
    Downsample the fire grid using block reduction and then slice a smaller region.
    
    Parameters:
        fire_data (np.ndarray): The original fire data array.
        block_size (tuple): Block size for downsampling.
        slice_shape (tuple): The shape to which the downsampled data is sliced.
        
    Returns:
        np.ndarray: The downsampled (and optionally sliced) fire grid.
    """
    downsampled = block_reduce(fire_data, block_size=block_size, func=np.mean)
    # Slice the result if needed (for example, to reduce to a smaller grid).
    downsampled = downsampled[:slice_shape[0], :slice_shape[1]]
    print("Downsampled fire grid shape:", downsampled.shape)
    return downsampled

def compute_full_matrix(downsampled_fire: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Given the downsampled fire grid, compute the Gaussian distance matrix, and combine
    fire intensities and spread influence into a full matrix.
    
    Parameters:
        downsampled_fire (np.ndarray): The downsampled fire data.
        sigma (float): Sigma parameter for the Gaussian distance matrix.
        
    Returns:
        tuple: A tuple containing the Gaussian matrix, fire_vector, spread_matrix, and full_matrix.
    """
    grid_shape = downsampled_fire.shape
    gauss_matrix = gaussian_distance_matrix_zero_diag(grid_shape, sigma)
    print("Gaussian distance matrix shape:", gauss_matrix.shape)
    print("Gaussian distance matrix:")
    print(gauss_matrix)
    
    # Flatten the fire grid.
    fire_vector = downsampled_fire.flatten()
    print("Fire vector shape:", fire_vector.shape)
    
    # Compute the spread matrix where each row is scaled by corresponding fire intensity.
    spread_matrix = fire_vector[:, np.newaxis] * gauss_matrix
    
    # Create a diagonal matrix from the fire intensities.
    fire_matrix = np.diag(fire_vector)
    
    # Combine to form the full matrix.
    full_matrix = fire_matrix + spread_matrix
    print("Full matrix shape:", full_matrix.shape)
    print("Full matrix:")
    print(full_matrix)
    
    return gauss_matrix, fire_vector, spread_matrix, full_matrix