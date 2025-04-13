from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from typing import Tuple
from time import time
import numpy as np

def QAOASolver(qubo: QuadraticProgram) -> Tuple:
    algorithm_globals.random_seed = 10598
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
    qaoa_optimizer = MinimumEigenOptimizer(qaoa_mes)
    qaoa_result = qaoa_optimizer.solve(qubo)
    print("QAOA Result:")
    print(qaoa_result.prettyprint())
    return qaoa_result

def ExhaustiveSolver(Q) -> Tuple:
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

def NumpySolver(qubo: QuadraticProgram) -> Tuple:
    exact_mes = NumPyMinimumEigensolver()
    exact_optimizer = MinimumEigenOptimizer(exact_mes)
    exact_result = exact_optimizer.solve(qubo)
    print("Exact Solver Result:")
    print(exact_result.prettyprint())
    return exact_result