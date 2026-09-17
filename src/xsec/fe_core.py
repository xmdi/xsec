import numpy as np

def evaluate_gauss_point(element_type, nodes, gp):
    """Evaluates and returns all matrices used in the Gaussian integration. All inputs should be np.float64"""
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
        # N = [N1 0  0  N2 0  0  N3 0  0  N4 0  0
        #      0  N1 0  0  N2 0  0  N3 0  0  N4 0
        #      0  0  N1 0  0  N2 0  0  N3 0  0  N4]
        N1 = 0.25 * (1 - gp[0]) * (1 - gp[1])
        dN1_de = 0.25 * (gp[1] - 1)
        dN1_dn = 0.25 * (gp[0] - 1)
        N2 = 0.25 * (1 + gp[0]) * (1 - gp[1])
        dN2_de = 0.25 * (1 - gp[1])
        dN2_dn = 0.25 * (-1 - gp[0])
        N3 = 0.25 * (1 + gp[0]) * (1 + gp[1])
        dN3_de = 0.25 * (1 - gp[1])
        dN3_dn = 0.25 * (1 - gp[0])
        N4 = 0.25 * (1 - gp[0]) * (1 + gp[1])
        dN4_de = 0.25 * (-1 - gp[1])
        dN4_dn = 0.25 * (1 - gp[0])
        N = np.zeros((3, 12), dtype = np.float64)
        N[0, :] = [N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0]
        N[1, :] = [0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0]
        N[2, :] = [0, 0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4]
        # dN = [dN1_de 0 0 DN2_de 0 0 DN3_de 0 0 DN4_de 0 0 
        #       0 dN1_dn 0 0 DN2_dn 0 0 DN3_dn 0 0 DN4_dn 0
        #       dN1_dn dN1_de 0 DN2_dn DN2_de 0 DN3_dn DN3_de 0 DN4_dn DN4_de 0
        #       0 0 DN1_de 0 0 DN2_de 0 0 DN3_de 0 0 DN4_de
        #       0 0 DN1_dn 0 0 DN2_dn 0 0 DN3_dn 0 0 DN4_dn
        #       0 0 0 0 0 0 0 0 0 0 0 0]
        # (dN is usually called the B matrix)
        dN = np.zeros((6, 12), dtype = np.float64)
        dN[0, :] = [dN1_de, 0, 0, DN2_de, 0, 0, DN3_de, 0, 0, DN4_de, 0, 0]
        dN[1, :] = [0, dN1_dn, 0, 0, DN2_dn, 0, 0, DN3_dn, 0, 0, DN4_dn, 0]
        dN[2, :] = [dN1_dn, dN1_de, 0, DN2_dn, DN2_de, 0, DN3_dn, DN3_de, 0, DN4_dn, DN4_de, 0]
        dN[3, :] = [0, 0, dN1_de, 0, 0, DN2_de, 0, 0, DN3_de, 0, 0, DN4_de]
        dN[4, :] = [0, 0, dN1_dn, 0, 0, DN2_dn, 0, 0, DN3_dn, 0, 0, DN4_dn]
        dN[5, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # J0 = [dN1/de dN2/de dN3/de dN4/de
        #       dN1/dn dN2/dn dN3/dn dN4/dn] (2x4)
        # nodes = [x1 y1
        #          x2 y2
        #          x3 y3
        #          x4 y4] (4x2)
        # J = J0 * nodes (2x2)
        J0 = np.zeros((2, 4), dtype = np.float64)
        J0[0, :] = [dN1_de, dN2_de, DN3_de DN4_de]
        J0[1, :] = [dN1_dn, dN2_dn, DN3_dn DN4_dn]
        J = J0 @ nodes
        # det(J) = J[0, 0] * J[1, 1] - J[1, 0] * J[0, 1]
        # inv(J) = [J[1, 1], -J[0, 1]; -J[1, 0], J[0, 0]] / det(J)
        det_J = J[0, 0] * J[1, 1] - J[1, 0] * J[0, 1]
        J_inv = np.zeros((2, 2), dtype = np.float64)
        J_inv[0, :] = [J[1, 1], -J[0, 1]] / det_J
        J_inv[1, :] = [-J[1, 0], J[0, 0]] / det_J
   else:
        raise(f"Unsupported element_type ({element_type}) in evaluate_gauss_point().")
    return N, dN, J, det_J, J_inv

def constitutive_matrix_isotropic(E, v):
    """Generates isotropic elastic constitutive matrix. All inputs should be np.float64"""
    G = E / (2 * (1 + v))
    Q1 = (E * (1 - v)) / ((1 + v) * (1 - 2 * v))
    Q2 = (E * v) / ((1 + v) * (1 - 2 * v))
    D = np.zeros((6, 6), dtype = np.float64)
    D[0:2,0:2] = [[Q1, Q2, Q2], [Q2, Q1, Q2], [Q2, Q2, Q1]]
    D[3:5,3:5] = [[G, 0, 0], [0, G, 0], [0, 0, G]]
    return D
