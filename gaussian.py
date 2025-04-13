import numpy as np
import rasterio
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

def gaussian_distance_matrix_zero_diag(grid_shape, sigma=1.0):
    """
    Compute a Gaussian distance matrix for a grid of points, with the diagonal set to 0.
    
    Each entry [i, j] is exp(-d^2 / (2*sigma^2)) for the Euclidean distance d between
    points i and j, but with the diagonal entries explicitly set to 0.
    """
    n_rows, n_cols = grid_shape
    y_coords, x_coords = np.indices((n_rows, n_cols))
    points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=-1)
    gauss_matrix = np.exp(-sq_dist / (2 * sigma**2))
    np.fill_diagonal(gauss_matrix, 0)
    return gauss_matrix

# Load fire data from a GeoTIFF (using Rasterio) and optionally downsample it.
tiff_file = "data/wfpi_data_20250412_20250412.tiff"  # Update path as needed
with rasterio.open(tiff_file) as src:
    # Read the first band
    fire_data = src.read(1)
    print("Original fire data shape:", fire_data.shape)

# Optional: Downsample the fire grid if you need coarser resolution.
# Here we use a block size of (5000, 5000); adjust as needed.
downsampled_fire = block_reduce(fire_data, block_size=(500, 500), func=np.mean)[:5, :5]
print("Downsampled fire grid shape:", downsampled_fire.shape)

# Define the grid shape from the downsampled fire data.
grid_shape = downsampled_fire.shape

# Compute the Gaussian distance matrix over the grid.
sigma = 1 # Adjust sigma as required.
gauss_matrix = gaussian_distance_matrix_zero_diag(grid_shape, sigma)
print("Gaussian distance matrix shape:", gauss_matrix.shape)
print("Gaussian distance matrix:")
print(gauss_matrix)

# Flatten the downsampled fire grid into a vector.
fire_vector = downsampled_fire.flatten()
print("Fire vector shape:", fire_vector.shape)

# Multiply each row of the Gaussian matrix by the corresponding fire intensity.
spread_matrix = fire_vector[:, np.newaxis] * gauss_matrix

# Create a diagonal matrix from the fire intensities.
fire_matrix = np.diag(fire_vector)

# The full matrix combines the direct fire values and the spread influence.
full_matrix = fire_matrix + spread_matrix
print("Full matrix shape:", full_matrix.shape)
print("Full matrix:")
print(full_matrix)

# Visualization: Create a figure with three subplots.
plt.figure(figsize=(18, 6))

# Subplot 1: Original fire data (optionally, you may zoom in if the full image is large)
plt.subplot(1, 3, 1)
# Here we display the full original fire data.
plt.imshow(fire_data, cmap='hot')
plt.title("Original Fire Data")
plt.colorbar()

# Subplot 2: Downsampled fire grid
plt.subplot(1, 3, 2)
plt.imshow(downsampled_fire, cmap='hot')
plt.title("Downsampled Fire Grid")
plt.colorbar()

# Subplot 3: Full matrix (fire + spread influence)
plt.subplot(1, 3, 3)
plt.imshow(full_matrix, cmap='hot')
plt.title("Full Matrix (Fire + Spread)")
plt.colorbar()

plt.tight_layout()
plt.show()
