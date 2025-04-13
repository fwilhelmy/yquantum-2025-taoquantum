import numpy as np
from qiskit_optimization import QuadraticProgram

def create_qubo(matrix, lambda1: float = 0.5, lambda2: float = 10000, k_constraint: int = 3) -> QuadraticProgram:
    """
    Create a Quadratic Unconstrained Binary Optimization (QUBO) problem.
    It reads Q from a file, creates a constraint matrix K, and constructs the QUBO.
    
    Parameters:
        lambda1 (float): Regularization parameter lambda1.
        lambda2 (float): Regularization parameter lambda2.
        k_constraint (int): Constraint value.
    
    Returns:
        QuadraticProgram: The constructed QUBO.
    """
    # Load Q (assuming the file exists at the given path)
    Q = matrix
    M = Q.shape[0]
    
    # Create constraint matrix K
    K = np.full((M, M), lambda2)
    diagonal_value_K = -lambda1 + lambda2 * (1 - 2 * k_constraint)
    np.fill_diagonal(K, diagonal_value_K)
    
    qubo = QuadraticProgram()
    # Adjust the number of variables if QUBO size is different from fixed value (9 in original code)
    num_vars = Q.shape[0]
    qubo.binary_var_list(num_vars, name="x")
    qubo.minimize(quadratic=Q + K)
    
    print(qubo.prettyprint())
    return qubo