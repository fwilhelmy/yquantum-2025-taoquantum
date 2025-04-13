from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import List, Tuple
import numpy as np

# create a QUBO
Q = -np.load("results/full_matrix.npy")
lambda1 = 0.5
lambda2 = 10000
k_constraint = 3
beta = 1000
M = Q.shape[0] 
K = np.full((M, M), lambda2)

# Calculate and set the diagonal value
diagonal_value_K = -lambda1 + lambda2 * (1 - 2 * k_constraint)
np.fill_diagonal(K, diagonal_value_K)
# In your case, you might have final_matrix = dummy_Q + K (with your constraint modifications).

qubo = QuadraticProgram()
qubo.binary_var_list(9, name="x")
qubo.minimize(quadratic=Q+K)
print(qubo.prettyprint())

algorithm_globals.random_seed = 10598
qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA
qaoa_result = qaoa.solve(qubo)
print(qaoa_result.prettyprint())

exact_mes = NumPyMinimumEigensolver()
exact = MinimumEigenOptimizer(exact_mes)  # using the exact classical numpy minimum eigen solver
exact_result = exact.solve(qubo)
print(exact_result.prettyprint())

import matplotlib.pyplot as plt

# Extract objective function values from the QAOA results.
# Here we assume that 'qaoa_result.samples' is a list of objects with an attribute 'fval'
qaoa_objective_values = [sample.fval for sample in qaoa_result.samples]

# Create a histogram of the QAOA objective values
fig, ax = plt.subplots(figsize=(8, 5))
n, bins, patches = ax.hist(qaoa_objective_values, bins=20, color='blue', alpha=0.7, edgecolor='black', label='QAOA samples')

# Overlay the exact value using a dotted vertical line.
exact_value = exact_result.fval  # The exact minimum objective value
ax.axvline(exact_value, color='red', linestyle='dotted', linewidth=2, label=f'Exact value ({exact_value:.2f})')

# Set labels and title
ax.set_xlabel("Objective Function Value")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of QAOA Objective Values with Exact Minimum")
ax.legend()

plt.show()
