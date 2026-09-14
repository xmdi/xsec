import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def test_sparse_solver():
    A = csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = np.array([1.0, 2.0])
    x = spsolve(A, b)
    
    expected = np.array([1.0/11.0, 7.0/11.0])
    np.testing.assert_allclose(x, expected, rtol=1e-5)
