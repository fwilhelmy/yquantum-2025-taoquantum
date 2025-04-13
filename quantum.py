import numpy as np
import itertools
import time
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

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
    for k in range(num_combinations):
        # Create the binary vector x from integer k
        binary_string = format(k, f'0{n}b')
        x = np.array([int(bit) for bit in binary_string], dtype=float)

        # Calculate the QUBO objective function value: x^T * Q * x
        current_value = x @ Q @ x

        # Update minimum if the current value is lower
        if current_value < min_value:
            min_value = current_value
            best_x = x.astype(int)

        # Optional: Print progress for very long runs
        if n >= 20 and k > 0 and k % (num_combinations // 100) == 0:
             progress = (k / num_combinations) * 100
             print(f"Progress: {progress:.1f}% completed...", end='\r')

    end_time = time.time()
    elapsed_time = end_time - start_time

    if n >= 20: print("\nSearch finished.")

    print(f"Exhaustive search completed in {elapsed_time:.4f} seconds.")

    return best_x, min_value, elapsed_time

def solve_qubo_quantum(Q, p=1, shots=1024):
    """
    Solves a QUBO problem minimizing f(x) = x^T * Q * x for binary x {0, 1}
    using Quantum Approximate Optimization Algorithm (QAOA).

    Args:
        Q (np.ndarray): The square matrix (n x n) defining the QUBO problem.
        p (int): The QAOA circuit depth (number of layers).
        shots (int): Number of shots for the quantum simulation.

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
    print(f"Starting quantum optimization for QUBO of size n={n} using QAOA (p={p}).")
    
    start_time = time.time()
    
    # Create a quadratic program from the Q matrix
    qp = QuadraticProgram()
    
    # Add binary variables
    for i in range(n):
        qp.binary_var(name=f'x{i}')
    
    # Set the objective function using the Q matrix
    # The QUBO form expected by Qiskit is: min x^T Q x + c^T x + constant
    linear = np.zeros(n)
    quadratic = {}
    
    for i in range(n):
        linear[i] = Q[i, i]  # Diagonal elements
        for j in range(i+1, n):
            # Off-diagonal elements (Qiskit requires the sum of Q[i,j] and Q[j,i])
            if abs(Q[i, j] + Q[j, i]) > 1e-10:  # Avoiding near-zero values
                quadratic[(i, j)] = Q[i, j] + Q[j, i]
    
    qp.minimize(linear=linear, quadratic=quadratic)
    
    # Convert to QUBO if it's not already in QUBO form
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    
    # Set up the quantum instance (simulator)
    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=shots)
    
    # Create QAOA solver
    qaoa = QAOA(optimizer=COBYLA(), reps=p, quantum_instance=quantum_instance)
    
    # Create the minimum eigen optimizer with QAOA
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    try:
        # Solve the problem
        result = qaoa_optimizer.solve(qubo)
        
        # Convert the result to our expected format
        best_x = np.array([result.x[i] for i in range(n)], dtype=int)
        min_value = result.fval
        
    except Exception as e:
        print(f"Quantum optimization failed: {str(e)}")
        return None, float('inf'), 0.0
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Quantum optimization completed in {elapsed_time:.4f} seconds.")
    
    return best_x, min_value, elapsed_time

def compare_methods(Q_matrix, K_matrix=None, D_matrix=None):
    """
    Compare classical exhaustive search with quantum QAOA for solving a QUBO problem.
    
    Args:
        Q_matrix (np.ndarray): The base Q matrix for the QUBO problem.
        K_matrix (np.ndarray, optional): Additional constraint matrix.
        D_matrix (np.ndarray, optional): Additional distance matrix.
    """
    # Determine the full problem matrix
    if K_matrix is not None and D_matrix is not None:
        final_matrix = Q_matrix + K_matrix - D_matrix
        matrix_description = "Q_matrix + K_matrix - D_matrix"
    elif K_matrix is not None:
        final_matrix = Q_matrix + K_matrix
        matrix_description = "Q_matrix + K_matrix"
    elif D_matrix is not None:
        final_matrix = Q_matrix - D_matrix
        matrix_description = "Q_matrix - D_matrix"
    else:
        final_matrix = Q_matrix
        matrix_description = "Q_matrix"
    
    n = final_matrix.shape[0]
    print(f"Problem size: {n}x{n} matrix")
    print(f"Matrix used: {matrix_description}")
    print("-" * 50)
    
    # 1. Solve using Exhaustive Search
    print("METHOD 1: EXHAUSTIVE SEARCH")
    best_x_exhaustive, min_value_exhaustive, time_exhaustive = solve_qubo_exhaustive(final_matrix)
    
    # 2. Solve using Quantum Optimization (QAOA)
    print("\nMETHOD 2: QUANTUM OPTIMIZATION (QAOA)")
    # For large matrices, we may need to reduce p to keep runtime reasonable
    p_value = 1 if n <= 10 else 1  # Use p=1 for larger problems to keep runtime manageable
    best_x_quantum, min_value_quantum, time_quantum = solve_qubo_quantum(final_matrix, p=p_value)
    
    # 3. Compare results
    print("\nRESULTS COMPARISON")
    print("-" * 50)
    
    print(f"Exhaustive Search:")
    if best_x_exhaustive is not None:
        if n > 10:
            print(f"  First 5 elements: {best_x_exhaustive[:5]}")
            print(f"  Last 5 elements:  {best_x_exhaustive[-5:]}")
            print(f"  Number of ones:   {np.sum(best_x_exhaustive)}")
        else:
            print(f"  x = {best_x_exhaustive}")
        print(f"  Minimum value: {min_value_exhaustive:.6f}")
        print(f"  Time taken: {time_exhaustive:.4f} seconds")
    else:
        print("  Failed to find solution")
    
    print(f"\nQuantum QAOA:")
    if best_x_quantum is not None:
        if n > 10:
            print(f"  First 5 elements: {best_x_quantum[:5]}")
            print(f"  Last 5 elements:  {best_x_quantum[-5:]}")
            print(f"  Number of ones:   {np.sum(best_x_quantum)}")
        else:
            print(f"  x = {best_x_quantum}")
        print(f"  Minimum value: {min_value_quantum:.6f}")
        print(f"  Time taken: {time_quantum:.4f} seconds")
    else:
        print("  Failed to find solution")
    
    # Compare solutions if both methods succeeded
    if best_x_exhaustive is not None and best_x_quantum is not None:
        print("\nAccuracy Comparison:")
        
        # Check if solutions are identical
        if np.array_equal(best_x_exhaustive, best_x_quantum):
            print("  Both methods found identical solutions!")
        else:
            print("  Solutions differ between methods")
            
            # Verify that the quantum solution is correct by recalculating its objective value
            verify_quantum = best_x_quantum @ final_matrix @ best_x_quantum
            print(f"  Quantum solution recalculated value: {verify_quantum:.6f}")
            
            # Calculate relative error
            rel_error = abs(min_value_quantum - min_value_exhaustive) / abs(min_value_exhaustive) * 100
            print(f"  Relative error: {rel_error:.4f}%")
        
        # Performance comparison
        speedup = time_exhaustive / time_quantum if time_quantum > 0 else float('inf')
        print(f"\nPerformance Comparison:")
        print(f"  Speedup (Quantum vs Exhaustive): {speedup:.2f}x")
        
        if speedup > 1:
            print(f"  Quantum approach was {speedup:.2f}x faster!")
        else:
            print(f"  Exhaustive search was {1/speedup:.2f}x faster for this problem size")
    
    return {
        'exhaustive': {
            'solution': best_x_exhaustive,
            'value': min_value_exhaustive,
            'time': time_exhaustive
        },
        'quantum': {
            'solution': best_x_quantum,
            'value': min_value_quantum,
            'time': time_quantum
        }
    }

# Main execution code
if __name__ == "__main__":
    # Load matrices
    try:
        Q_matrix = np.load("results/full_matrix.npy")
        D_matrix = np.load("results/gauss_matrix.npy")
    except FileNotFoundError:
        print("Matrix files not found. Creating example matrices instead.")
        # Generate example matrices
        np.random.seed(42)
        N = 5  # Using a smaller size for demonstration
        Q_matrix = np.random.randn(N, N)
        D_matrix = np.random.randn(N, N)
    
    # Parameters
    Q_matrix = Q_matrix * -1
    lambda1 = 0.5
    lambda2 = 10000
    k_constraint = 3
    beta = 1000
    
    # Create the K constraint matrix
    M = Q_matrix.shape[0]
    K = np.full((M, M), lambda2)
    diagonal_value_K = -lambda1 + lambda2 * (1 - 2 * k_constraint)
    np.fill_diagonal(K, diagonal_value_K)
    
    # Compare the methods
    results = compare_methods(Q_matrix, K, D_matrix)
