import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def solve_sparse_system(A, b):
    """Solves a sparse linear system Ax = b."""
    # Convert to Compressed Sparse Row format for efficiency
    A_sparse = csr_matrix(A)
    return spsolve(A_sparse, b)
