import numpy as np
import rasterio
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import Tuple

# ------------------------------
# Functions for Fire Data Processing
# ------------------------------

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

def plot_fire_data(original_fire: np.ndarray, downsampled_fire: np.ndarray, full_matrix: np.ndarray):
    """
    Create a visualization with three subplots: original fire data, downsampled fire grid, and full matrix.
    
    Parameters:
        original_fire (np.ndarray): The original fire data.
        downsampled_fire (np.ndarray): The downsampled fire data.
        full_matrix (np.ndarray): The full matrix combining direct and spread fire values.
    """
    plt.figure(figsize=(18, 6))
    
    # Original fire data
    plt.subplot(1, 3, 1)
    plt.imshow(original_fire, cmap='hot')
    plt.title("Original Fire Data")
    plt.colorbar()
    
    # Downsampled fire data
    plt.subplot(1, 3, 2)
    plt.imshow(downsampled_fire, cmap='hot')
    plt.title("Downsampled Fire Grid")
    plt.colorbar()
    
    # Full matrix (fire + spread influence)
    plt.subplot(1, 3, 3)
    plt.imshow(full_matrix, cmap='hot')
    plt.title("Full Matrix (Fire + Spread)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()


# ------------------------------
# Functions for QUBO and Optimization
# ------------------------------

def create_qubo(lambda1: float = 0.5, lambda2: float = 10000, k_constraint: int = 3) -> QuadraticProgram:
    """
    Create a Quadratic Unconstrained Binary Optimization (QUBO) problem.
    It reads Q from a file, creates a constraint matrix K, and constructs the QUBO.
    
    Parameters:
        lambda1 (float): Regularization parameter lambda1.
        lambda2 (float): Regularization parameter lambda2.
        k_constraint (int): Constraint value.
    
    Returns:
        QuadraticProgram: The constructed QUBO.
    """
    # Load Q (assuming the file exists at the given path)
    Q = -np.load("results/full_matrix.npy")
    M = Q.shape[0]
    
    # Create constraint matrix K
    K = np.full((M, M), lambda2)
    diagonal_value_K = -lambda1 + lambda2 * (1 - 2 * k_constraint)
    np.fill_diagonal(K, diagonal_value_K)
    
    qubo = QuadraticProgram()
    # Adjust the number of variables if QUBO size is different from fixed value (9 in original code)
    num_vars = Q.shape[0]
    qubo.binary_var_list(num_vars, name="x")
    qubo.minimize(quadratic=Q + K)
    
    print(qubo.prettyprint())
    return qubo

def run_optimizers(qubo: QuadraticProgram) -> Tuple:
    """
    Run the QAOA optimizer and the exact (NumPyMinimumEigensolver) optimizer on the given QUBO.
    
    Parameters:
        qubo (QuadraticProgram): The QUBO formulation.
    
    Returns:
        tuple: Results from QAOA and exact optimizers.
    """
    # QAOA optimization
    algorithm_globals.random_seed = 10598
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
    qaoa_optimizer = MinimumEigenOptimizer(qaoa_mes)
    qaoa_result = qaoa_optimizer.solve(qubo)
    print("QAOA Result:")
    print(qaoa_result.prettyprint())
    
    # Exact optimization
    exact_mes = NumPyMinimumEigensolver()
    exact_optimizer = MinimumEigenOptimizer(exact_mes)
    exact_result = exact_optimizer.solve(qubo)
    print("Exact Solver Result:")
    print(exact_result.prettyprint())
    
    return qaoa_result, exact_result

def plot_qaoa_histogram(qaoa_result, exact_result):
    """
    Plot a histogram of the QAOA objective values and overlay the exact minimum objective value.
    
    Parameters:
        qaoa_result: The result object from the QAOA optimizer.
        exact_result: The result object from the exact optimizer.
    """
    # Assuming each sample in qaoa_result.samples has an attribute 'fval'
    qaoa_objective_values = [sample.fval for sample in qaoa_result.samples]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(qaoa_objective_values, bins=20, color='blue',
                               alpha=0.7, edgecolor='black', label='QAOA samples')
    
    # Overlay the exact value with a dotted vertical line.
    exact_value = exact_result.fval
    ax.axvline(exact_value, color='red', linestyle='dotted', linewidth=2,
               label=f'Exact value ({exact_value:.2f})')
    
    ax.set_xlabel("Objective Function Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of QAOA Objective Values with Exact Minimum")
    ax.legend()
    plt.show()


# ------------------------------
# Main Execution Function
# ------------------------------

def main():
    # Fire data processing and visualization
    tiff_file = "data/wfpi_data_20250412_20250412.tiff"  # Adjust path as needed.
    fire_data = load_fire_data(tiff_file)
    downsampled_fire = downsample_fire_data(fire_data, block_size=(500, 500), slice_shape=(3, 3))
    
    sigma = 1  # Adjust sigma as required.
    _, _, _, full_matrix = compute_full_matrix(downsampled_fire, sigma)
    plot_fire_data(fire_data, downsampled_fire, full_matrix)
    
    # QUBO creation and optimization
    qubo = create_qubo(lambda1=0.5, lambda2=10000, k_constraint=3)
    qaoa_result, exact_result = run_optimizers(qubo)
    plot_qaoa_histogram(qaoa_result, exact_result)


if __name__ == "__main__":
    main()
