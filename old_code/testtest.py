import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def solve_QUBO_qaoa(final_matrix, p=1, max_iterations=100):
    """
    Solves the QUBO problem defined by final_matrix using QAOA from Qiskit.
    
    Parameters:
        final_matrix (np.array): QUBO cost matrix (n x n).
        p (int): Number of QAOA layers (repetitions).
        max_iterations (int): Maximum iterations for the classical optimizer.
        
    Returns:
        result: The optimization result from MinimumEigenOptimizer.
        counts (dict): The measurement counts dictionary from the QAOA run.
    """
    n = final_matrix.shape[0]
    
    # Create a QuadraticProgram and add n binary variables.
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f'x{i}')
    
    # Build the objective.
    # The QUBO cost is: sum_i Q[ii]*x_i + sum_{i<j} Q[i,j]*x_i*x_j.
    linear = {f'x{i}': float(final_matrix[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            quadratic[(f'x{i}', f'x{j}')] = float(final_matrix[i, j])
    
    qp.minimize(linear=linear, quadratic=quadratic)
    
    # Set up QAOA through the MinimumEigenOptimizer.
    backend = Aer.get_backend("qasm_simulator")
    qi = QuantumInstance(backend, shots=1000)
    optimizer = COBYLA(maxiter=max_iterations)
    qaoa = QAOA(optimizer=optimizer, reps=p, quantum_instance=qi)
    meo = MinimumEigenOptimizer(qaoa)
    
    # Solve the optimization problem.
    result = meo.solve(qp)
    
    # Extract raw measurement counts from the QAOA run.
    # (They are stored in the private _ret attribute under key "raw_counts".)
    ret = qaoa._ret  if hasattr(qaoa, "_ret") else None
    if ret is not None and "raw_counts" in ret:
        counts = ret["raw_counts"]
    else:
        counts = {}
    
    return result, counts

# Example usage:
# Load your QUBO cost matrix (for instance, from file) or define one.
dummy_Q = np.load("results/full_matrix.npy")

lambda1 = 0.5
lambda2 = 10000
k_constraint = 3
beta = 1000
M = dummy_Q.shape[0] 
K = np.full((M, M), lambda2)

# Calculate and set the diagonal value
diagonal_value_K = -lambda1 + lambda2 * (1 - 2 * k_constraint)
np.fill_diagonal(K, diagonal_value_K)
# In your case, you might have final_matrix = dummy_Q + K (with your constraint modifications).

# Solve the QUBO with QAOA.
result, counts = solve_QUBO_qaoa(dummy_Q+K, p=2, max_iterations=200)

# If there are too many outcomes, show only the top k outcomes.
num_samples = sum(counts.values()) if counts else 0
top_k = 5  # Change to however many outcomes you wish to display
if counts:
    # Sort outcomes by descending frequency.
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_results = sorted_counts[:top_k]
    bitstrings, counts_vals = zip(*top_results)
else:
    bitstrings, counts_vals = [], []

plt.figure(figsize=(8, 5))
plt.bar(bitstrings, counts_vals, color='skyblue', edgecolor='black')
plt.xlabel("Bitstring")
plt.ylabel("Counts")
plt.title("Top {} QAOA Measurement Outcomes".format(top_k))
plt.show()
