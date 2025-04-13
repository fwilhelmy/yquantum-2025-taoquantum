import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

def solve_QUBO_qaoa(final_matrix, p=1, max_iterations=100, stepsize=0.1, init_params=None):
    """
    Solves the given QUBO problem using a QAOA ansatz via PennyLane.
    
    Parameters:
        final_matrix (np.array): The Q matrix for the QUBO problem.
                                 It should be square with dimension n x n.
        p (int): Number of QAOA layers.
        max_iterations (int): Maximum number of optimization steps.
        stepsize (float): Step size for the gradient descent optimizer.
        init_params (np.array, optional): Initial parameters for QAOA.
                                          If None, parameters are initialized uniformly.
    
    Returns:
        params (np.array): The optimized QAOA parameters.
        energy (float): The expected cost (energy) using the optimized parameters.
        probs (np.array): The final measurement probability distribution.
        sampling_circuit (function): The QNode for sampling the circuit.
    """
    # Number of qubits equals the dimension of Q
    n = final_matrix.shape[0]

    # Convert the QUBO matrix into a cost Hamiltonian.
    # Using the substitution: x_i = (1 - Z_i)/2, we get:
    #   C = sum_{i,j} Q_{ij} * ((1 - Z_i)(1 - Z_j))/4.
    # Expanding yields single-qubit Z terms and two-qubit Z_i Z_j interactions.
    coeffs = []
    ops = []
    
    for i in range(n):
        coeff = -0.5 * np.sum(final_matrix[i])
        if np.abs(coeff) > 1e-8:
            coeffs.append(coeff)
            ops.append(qml.PauliZ(i))
    
    for i in range(n):
        for j in range(i+1, n):
            coeff = final_matrix[i, j] / 2.0
            if np.abs(coeff) > 1e-8:
                ops.append(qml.operation.Tensor(qml.PauliZ(i), qml.PauliZ(j)))
                coeffs.append(coeff)
                
    H = qml.Hamiltonian(coeffs, ops)

    # Set up the device.
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(params):
        # Prepare an equal superposition.
        for i in range(n):
            qml.Hadamard(wires=i)
        gammas = params[:p]
        betas = params[p:]
        for k in range(p):
            qml.ApproxTimeEvolution(H, gammas[k], 1)
            for i in range(n):
                qml.RX(2 * betas[k], wires=i)
        return qml.expval(H)

    if init_params is None:
        init_params = np.random.uniform(0, np.pi, 2 * p)
    params = init_params.copy()
    opt = qml.GradientDescentOptimizer(stepsize)
    for _ in range(max_iterations):
        params = opt.step(lambda v: circuit(v), params)
    energy = circuit(params)
    
    # Define the sampling QNode to get the probability distribution.
    @qml.qnode(dev)
    def sampling_circuit(params):
        for i in range(n):
            qml.Hadamard(wires=i)
        gammas = params[:p]
        betas = params[p:]
        for k in range(p):
            qml.ApproxTimeEvolution(H, gammas[k], 1)
            for i in range(n):
                qml.RX(2 * betas[k], wires=i)
        return qml.probs(wires=range(n))
    
    probs = sampling_circuit(params)
    return params, energy, probs, sampling_circuit

# Example usage:
# Create a small dummy Q matrix representing your QUBO problem.
dummy_Q = np.load("results/full_matrix.npy")

params_opt, energy_opt, outcome_probs, sampling_circ = solve_QUBO_qaoa(dummy_Q, p=2, max_iterations=200, stepsize=0.2)

print("Optimized parameters:\n", params_opt)
print("Minimum energy found:", energy_opt)
print("Final measurement probabilities:\n", outcome_probs)

# To create a histogram, we sample many times from the output distribution.
num_samples = 1000
n_qubits = dummy_Q.shape[0]
# Generate the computational basis states as integers 0,1,..., 2^n - 1.
basis_states = np.arange(2 ** n_qubits)
# Use NumPy's random.choice with outcome_probs as the probability weights.
samples = np.random.choice(basis_states, size=num_samples, p=outcome_probs)

# Count occurrences of each basis state.
unique, counts = np.unique(samples, return_counts=True)

# Sort outcomes by descending frequency (highest counts first).
sorted_indices = np.argsort(counts)[::-1]
unique_sorted = unique[sorted_indices]
counts_sorted = counts[sorted_indices]

# Select the top k outcomes.
top_k = 5
top_unique = unique_sorted[:top_k]
top_counts = counts_sorted[:top_k]

# Convert the integer basis states to bitstring labels.
bitstring_labels = [format(state, '0{}b'.format(n_qubits)) for state in top_unique]

# Plot the histogram showing only the top k values.
plt.figure(figsize=(8, 5))
plt.bar(bitstring_labels, top_counts, color='skyblue', edgecolor='black')
plt.xlabel("Bitstring")
plt.ylabel("Counts")
plt.title("Top {} QAOA Measurement Outcomes".format(top_k))
plt.show()