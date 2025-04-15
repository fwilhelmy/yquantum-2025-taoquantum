import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def plot_fire_data(original_fire: np.ndarray,
                   downsampled_fire: np.ndarray,
                   full_matrix: np.ndarray,
                   output_dir: str) -> None:
    """
    Create and save a visualization with three subplots: original fire data,
    downsampled fire grid, and full matrix.

    Parameters:
        original_fire (np.ndarray): The original fire data.
        downsampled_fire (np.ndarray): The downsampled fire data.
        full_matrix (np.ndarray): The full matrix combining direct and spread fire values.
        output_dir (str): Directory to save the resulting plot.
    """
    plt.figure(figsize=(18, 6))

    # Original fire data
    plt.subplot(1, 3, 1)
    plt.imshow(original_fire, cmap='hot')
    plt.xticks([]); plt.yticks([])
    plt.title("Original Data")

    # Downsampled fire data
    plt.subplot(1, 3, 2)
    plt.imshow(downsampled_fire, cmap='hot')
    plt.xticks([]); plt.yticks([])
    plt.title("Downsampled Grid")

    # Full matrix (fire + spread influence)
    plt.subplot(1, 3, 3)
    plt.imshow(full_matrix, cmap='hot')
    plt.xticks([]); plt.yticks([])
    plt.title("Diffusion Matrix of the Fire")
    plt.colorbar()

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "fire_visualization.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved fire visualization to {save_path}")

    plt.show()


def plot_qaoa_histogram(qaoa_result: Any, exact_result: Any, output_dir: str) -> None:
    """
    Plot and save a histogram of the QAOA objective values with the exact minimum value.

    Parameters:
        qaoa_result (Any): Object with a 'samples' attribute, each sample having a 'fval'.
        exact_result (Any): Object with a 'fval' attribute (the exact minimum).
        output_dir (str): Directory to save the resulting plot.
    """
    qaoa_objective_values = [sample.fval for sample in qaoa_result.samples]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(qaoa_objective_values, bins=20, color='blue', alpha=0.7,
            edgecolor='black', label='QAOA samples')

    exact_value = exact_result.fval
    ax.axvline(exact_value, color='red', linestyle='dotted', linewidth=2,
               label=f'Exact value ({exact_value:.2f})')

    xmin, xmax = ax.get_xlim()
    xtick_positions = np.linspace(xmin, xmax, num=6)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f"{tick:.2f}" for tick in xtick_positions])

    ax.set_xlabel("Objective Function Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of QAOA Objective Values with Exact Minimum")
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "qaoa_histogram.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved QAOA histogram to {save_path}")

    plt.show()
