import numpy as np
import rasterio
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

# Replace with the path to your TIFF file
tiff_file = "data/wfpi_data_20250412_20250412.tiff"

# Open the TIFF file and read the first band
with rasterio.open(tiff_file) as src:
    data = src.read(1)
    print("Original data shape:", data.shape)

# Define the block size (example: average every 5x5 block)
block_size = (1000, 1000)
downsampled_grid = block_reduce(data, block_size=block_size, func=np.mean)
print("Downsampled grid shape:", downsampled_grid.shape)

# Define region boundaries for zoom (using array indices)
# For instance, if you want to focus on rows 100 to 200 and columns 150 to 250:
y_start, y_end = 0, 5  # rows - vertical axis
x_start, x_end = 0, 5  # columns - horizontal axis

# Extract the region from the original data (or downsampled grid if desired)
zoomed_region_original = data[y_start:y_end, x_start:x_end]
zoomed_region_downsampled = downsampled_grid[y_start//block_size[0] : y_end//block_size[0],
                                              x_start//block_size[1] : x_end//block_size[1]]


from gaussian import gaussian_distance_matrix_zero_diag

# Create a Gaussian object with the downsampled grid
gauss = gaussian_distance_matrix_zero_diag(np.diag(downsampled_grid[0 : 2, 0 : 2].flatten()))

# Flatten the 2D grid into a 1D vector
diagonal_vector = zoomed_region_downsampled.flatten()

# Create a diagonal matrix
diagonal_matrix = np.diag(diagonal_vector)

# Print and visualize the zoomed regions

# print("Zoomed region from original data:")
# print(zoomed_region_original)

# print("\nZoomed region from downsampled grid:")
# print(zoomed_region_downsampled)

# Plotting the zoomed region from the original data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='hot')
plt.title("Zoomed Original Region")
plt.colorbar()

# Plotting the zoomed region from the downsampled grid
plt.subplot(1, 2, 2)
plt.imshow(downsampled_grid, cmap='hot')
plt.title("Zoomed Downsampled Region")
plt.colorbar()
plt.show()
