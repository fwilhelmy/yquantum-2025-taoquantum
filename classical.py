import numpy as np
import itertools
import time

def solve_qubo_exhaustive(Q):
    """
    Solves a QUBO problem minimizing f(x) = x^T * Q * x for binary x {0, 1}
    using exhaustive search.

    Args:
        Q (np.ndarray): The square matrix (n x n) defining the QUBO problem.

    Returns:
        tuple: A tuple containing:
            - best_x (np.ndarray): The binary vector x (n x 1) that minimizes the QUBO.
                                   Returns None if Q is not a square matrix.
            - min_value (float): The minimum value of the objective function found.
                                 Returns float('inf') if Q is not square.
            - elapsed_time (float): Time taken for the computation in seconds.
    """
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        print("Error: Q must be a square matrix.")
        return None, float('inf'), 0.0

    n = Q.shape[0]
    num_combinations = 1 << n # Same as 2**n

    print(f"Starting exhaustive search for QUBO of size n={n}.")
    print(f"Total combinations to check: {num_combinations:,}")

    if n > 26: # Add a warning for potentially very long run times
        print("Warning: n > 26, exhaustive search may take a very long time!")

    min_value = float('inf')
    best_x = None
    start_time = time.time()

    # Iterate through all possible binary vectors
    # We can represent each combination by an integer from 0 to 2^n - 1
    for k in range(num_combinations):
        # Efficiently create the binary vector x from integer k
        # Method 1: String formatting (readable)
        binary_string = format(k, f'0{n}b')
        x = np.array([int(bit) for bit in binary_string], dtype=float) # Use float for matmul

        # Method 2: Bit manipulation (potentially faster for very large n, but less clear)
        # x = np.array([(k >> i) & 1 for i in range(n-1, -1, -1)], dtype=float)

        # Calculate the QUBO objective function value: x^T * Q * x
        # In numpy: x @ Q @ x (for 1D vector x, this works as x.T @ Q @ x)
        current_value = x @ Q @ x

        # Update minimum if the current value is lower
        if current_value < min_value:
            min_value = current_value
            best_x = x.astype(int) # Store the best x as integers

        # Optional: Print progress for very long runs
        if n >= 20 and k > 0 and k % (num_combinations // 100) == 0:
             progress = (k / num_combinations) * 100
             print(f"Progress: {progress:.1f}% completed...", end='\r')

    end_time = time.time()
    elapsed_time = end_time - start_time

    if n >= 20: print("\nSearch finished.") # Newline after progress indicator

    print(f"Exhaustive search completed in {elapsed_time:.4f} seconds.")

    return best_x, min_value, elapsed_time

# --- Example Usage ---
N = 25

# Create a random Q matrix for the example.
# QUBO matrices don't need to be symmetric or positive definite.

# Elements can be positive or negative.
np.random.seed(42) # for reproducibility
Q_matrix = np.load("results/full_matrix.npy")
print(Q_matrix)
#Q_matrix = np.diag(Q_matrix.flatten()) # Make it diagonal, using flatten() directly
Q_matrix = Q_matrix * -1

lambda1 = 0.5
lambda2 = 10000
k_constraint = 3
beta = 1000

D_matrix = np.load("results/gauss_matrix.npy")
#D = D_matrix * beta
M = Q_matrix.shape[0] 

# Create K filled with the off-diagonal value lambda2
K = np.full((M, M), lambda2)

# Calculate and set the diagonal value
diagonal_value_K = -lambda1 + lambda2 * (1 - 2 * k_constraint)
np.fill_diagonal(K, diagonal_value_K)



#np.random.randn(N, N)

# Optional: Make Q upper triangular (often the standard form, but not strictly necessary for x^T Q x)
# Q_matrix = np.triu(Q_matrix)

print(f"Solving QUBO for a {N}x{N} matrix.")
# print("Q matrix (first 5x5 block):\n", Q_matrix[:5, :5])
print("-" * 30)

# --- Solve using Exhaustive Search ---
best_x_vector, min_qubo_value, time_taken = solve_qubo_exhaustive(Q_matrix+K-D)

# --- Corrected Verification ---
print("-" * 30)
if best_x_vector is not None:
    print("Optimal binary vector x found:")
    # Use M (actual dimension) for checking size, not N
    if M > 10:
         # Print the whole vector if it's all ones, as that's revealing
        if np.all(best_x_vector == 1):
             print(f"  x = all ones (length {M})")
        elif np.all(best_x_vector == 0):
             print(f"  x = all zeros (length {M})")
        else:
            print(f"  First 5 elements: {best_x_vector[:5]}")
            print(f"  Last 5 elements:  {best_x_vector[-5:]}")
            print(f"  Number of ones:   {np.sum(best_x_vector)}") # Useful info
    else:
        print(f"  x = {best_x_vector}")
        print(f"  Number of ones: {np.sum(best_x_vector)}")

    # Use Q_matrix + K in the print statement label
    print(f"\nMinimum QUBO value f(x) = x^T*(Q_matrix + K)*x: {min_qubo_value:.6f}")

    # Verification: Recalculate value using Q_matrix + K
    # THIS IS THE FIX: Use the same matrix as the solver
    verify_value = best_x_vector @ (Q_matrix + K) @ best_x_vector
    print(f"Verification calculation (using Q_matrix + K): {verify_value:.6f}") # Label correction
    assert np.isclose(min_qubo_value, verify_value), "Verification failed!"
else:
    print("QUBO solving failed (likely input error).")


