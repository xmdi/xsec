import numpy as np

def evaluate_jacobian(element_type, nodes, gp):
    """Evaluates and returns the Jacobian matrix (maps isoparametric element to global coordinates defined in nodes array) at a given Gauss integration point. All inputs should be np.float64"""
    if element_type == 3: # bilinear quad
        # N1 = 1/4*(1-e)(1-n)
        #   dN1/de = 1/4*(n-1)
        #   dN1/dn = 1/4*(e-1)
        # N2 = 1/4*(1+e)(1-n)
        #   dN2/de = 1/4*(1-n)
        #   dN2/dn = 1/4*(-1-e)
        # N3 = 1/4*(1+e)(1+n)
        #   dN3/de = 1/4*(1-n)
        #   dN3/dn = 1/4*(1-e)
        # N4 = 1/4*(1-e)(1+n)
        #   dN4/de = 1/4*(-1-n)
        #   dN4/dn = 1/4*(1-e)
        # J0 = [dN1/de dN2/de dN3/de dN4/de
        #       dN1/dn dN2/dn dN3/dn dN4/dn] (2x4)
        # nodes = [x1 y1
        #          x2 y2
        #          x3 y3
        #          x4 y4] (4x2)
        # J = J0 * nodes (2x2)
        J0 = np.zeros((2, 4), dtype = np.float64)
        J0[0, :] = [gp[1] - 1, 1 - gp[1], 1 - gp[1], -1 - gp[1]]
        J0[1, :] = [gp[0] - 1, -1 - gp[0], 1 - gp[0], 1 - gp[0]]
        J = J0 @ nodes
    else:
        raise(f"Unsupported element_type ({element_type}) in evaluate_jacobian().")
    return J
