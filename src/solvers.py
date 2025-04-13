from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from typing import Tuple

def QAOASolver(qubo: QuadraticProgram) -> Tuple:
    algorithm_globals.random_seed = 10598
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
    qaoa_optimizer = MinimumEigenOptimizer(qaoa_mes)
    qaoa_result = qaoa_optimizer.solve(qubo)
    print("QAOA Result:")
    print(qaoa_result.prettyprint())
    return qaoa_result

def ExhaustiveSolver(qubo: QuadraticProgram) -> Tuple:
    return None

def NumpySolver(qubo: QuadraticProgram) -> Tuple:
    exact_mes = NumPyMinimumEigensolver()
    exact_optimizer = MinimumEigenOptimizer(exact_mes)
    exact_result = exact_optimizer.solve(qubo)
    print("Exact Solver Result:")
    print(exact_result.prettyprint())
    return exact_result