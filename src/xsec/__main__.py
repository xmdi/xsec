import sys
from matrix_runner.math_core import solve_sparse_system
import numpy as np

def main():
    print("--- Matrix Runner CLI ---")
    
    # Example dummy data for a quick test run
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    
    solution = solve_sparse_system(A, b)
    print(f"Computed Solution: {solution}")

if __name__ == "__main__":
    main()
