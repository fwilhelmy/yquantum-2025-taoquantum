import numpy as np

def compute_cost(x, Q):
    """
    Compute the cost for a given solution.

    Parameters:
        x (numpy.ndarray): A binary vector (1D array) indicating fire truck placements.
        Q (numpy.ndarray): The quadratic cost matrix (n x n).

    Returns:
        float: The computed cost, x^T * Q * x.
    """
    return x.dot(Q.dot(x))

def find_optimal_solution(Q):
    """
    Find the optimal binary solution vector that minimizes the cost function x^T * Q * x,
    assuming a brute force search over all 2^n possibilities.

    Parameters:
        Q (numpy.ndarray): The quadratic cost matrix (n x n).
    
    Returns:
        tuple: (optimal_x, best_cost)
            optimal_x (numpy.ndarray): The optimal binary vector (1D array) where placing a fire truck reduces the cost.
            best_cost (float): The computed cost for the optimal solution.
    """
    n = Q.shape[0]
    best_cost = float('inf')
    best_solution = None

    # Iterate over all 2^n binary combinations
    for i in range(2 ** n):
        # Convert integer i into a binary vector of size n.
        # np.binary_repr produces a string representing i in binary (with leading zeros given width).
        x_str = np.binary_repr(i, width=n)
        x = np.array([int(bit) for bit in x_str])
        cost = compute_cost(x, Q)
        if cost < best_cost:
            best_cost = cost
            best_solution = x.copy()
    
    return best_solution, best_cost

# ----- Example Usage -----
if __name__ == '__main__':
    # Example: Suppose we have a Q matrix from our fire problem.
    # For instance, this Q could be computed from our wildfire scenario based on current fire intensities and spread factors.
    # Here we create a small dummy 4x4 Q matrix for demonstration.
    Q = np.array([
        [10, 0],
        [0,  8],
    ])
    
    # Compute the optimal solution for fire truck placement (x[i] = 1: place a truck in cell i)
    optimal_x, best_cost = find_optimal_solution(Q)
    
    print("Optimal solution (fire truck placements):", optimal_x)
    print("Minimum cost:", best_cost)
    
    # For a given solution x, you can also compute its cost using compute_cost
    cost_for_optimal = compute_cost(optimal_x, Q)
    print("Verified cost (x^T * Q * x) for optimal solution:", cost_for_optimal)
