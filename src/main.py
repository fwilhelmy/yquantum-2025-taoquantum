from data import load_fire_data, downsample_fire_data, compute_full_matrix
from qubo import create_qubo
from visualisation import plot_fire_data, plot_qaoa_histogram
from solvers import QAOASolver, NumpySolver

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
    qaoa_result = QAOASolver(qubo)
    exact_result = NumpySolver(qubo)
    plot_qaoa_histogram(qaoa_result, exact_result)

if __name__ == "__main__":
    main()
