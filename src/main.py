import argparse
import os
import numpy as np

from data import load_fire_data, downsample_fire_data, compute_full_matrix
from qubo import create_qubo
from visualisation import plot_fire_data, plot_qaoa_histogram
from solvers import QAOASolver, NumpySolver


def parse_args():
    parser = argparse.ArgumentParser(description="Wildfire Response Optimization")
    parser.add_argument("-f", "--file", required=True, help="Path to the fire heatmap file (GeoTIFF or .npy)")
    parser.add_argument("-r", "--resources", type=int, required=True, help="Number of firefighting resources to dispatch")
    parser.add_argument("-d", "--downsample", nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'),
                        help="Downsampling block size as two integers")
    parser.add_argument("-s", "--spread", type=float, default=0.5, help="Fire diffusion parameter (default: 0.5)")
    parser.add_argument("-o", "--output", default="./results", help="Output directory (default: ./results)")
    parser.add_argument("-c", "--coordinates", nargs=4, type=float, metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
                        help="Longitude and latitude bounds to crop the GeoTIFF")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load fire data
    if args.coordinates:
        lon_min, lat_min, lon_max, lat_max = args.coordinates
        fire_data = load_fire_data(args.file, longitude=(lon_min, lon_max), latitude=(lat_min, lat_max))
    else:
        fire_data = load_fire_data(args.file)

    # Downsampling
    if args.downsample:
        block_size = tuple(args.downsample)
        downsampled_fire = downsample_fire_data(fire_data, block_size=block_size, inverted=True)
    else:
        downsampled_fire = fire_data  # no downsampling

    # Compute influence matrix
    gauss_matrix, fire_vector, spread_matrix, full_matrix = compute_full_matrix(downsampled_fire, sigma=args.spread)

    # Save matrices to .npy files
    np.save(os.path.join(args.output, "gauss_matrix.npy"), gauss_matrix)
    np.save(os.path.join(args.output, "fire_vector.npy"), fire_vector)
    np.save(os.path.join(args.output, "spread_matrix.npy"), spread_matrix)
    np.save(os.path.join(args.output, "full_matrix.npy"), full_matrix)
    print(f"[INFO] Saved matrices to {args.output}")

    # Create QUBO
    qubo = create_qubo(full_matrix, lambda1=0.5, lambda2=10000, k_constraint=args.resources)

    # Solve with QAOA and exact classical solver
    qaoa_result = QAOASolver(qubo)
    exact_result = NumpySolver(qubo)

    # Plot results
    plot_fire_data(fire_data, downsampled_fire, full_matrix, args.output)
    plot_qaoa_histogram(qaoa_result, exact_result, args.output)


if __name__ == "__main__":
    main()