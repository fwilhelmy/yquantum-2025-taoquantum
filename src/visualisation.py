import matplotlib.pyplot as plt
import numpy as np

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