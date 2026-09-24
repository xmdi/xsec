import numpy as np
import gmsh
from scipy.sparse import lil_matrix

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
        dN3_de = 0.25 * (1 + gp[1])
        dN3_dn = 0.25 * (1 + gp[0])
        N4 = 0.25 * (1 - gp[0]) * (1 + gp[1])
        dN4_de = 0.25 * (-1 - gp[1])
        dN4_dn = 0.25 * (1 - gp[0])
        N = np.zeros((3, 12))
        N[0, :] = [N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0]
        N[1, :] = [0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0]
        N[2, :] = [0, 0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4]
        
        # J0 = [dN1/de dN2/de dN3/de dN4/de
        #       dN1/dn dN2/dn dN3/dn dN4/dn] (2x4)
        # nodes = [x1 y1
        #          x2 y2
        #          x3 y3
        #          x4 y4] (4x2)
        # J = J0 * nodes (2x2)
        J0 = np.zeros((2, 4))
        J0[0, :] = [dN1_de, dN2_de, dN3_de, dN4_de]
        J0[1, :] = [dN1_dn, dN2_dn, dN3_dn, dN4_dn]
        J = J0 @ nodes
        # det(J) = J[0, 0] * J[1, 1] - J[1, 0] * J[0, 1]
        # inv(J) = [J[1, 1], -J[0, 1]; -J[1, 0], J[0, 0]] / det(J)
        det_J = J[0, 0] * J[1, 1] - J[1, 0] * J[0, 1]
        J_inv = np.zeros((2, 2))
        J_inv[0, :] = [J[1, 1], -J[0, 1]] / det_J
        J_inv[1, :] = [-J[1, 0], J[0, 0]] / det_J

        # dN = [dN1_de 0 0 DN2_de 0 0 DN3_de 0 0 DN4_de 0 0 
        #       0 dN1_dn 0 0 DN2_dn 0 0 DN3_dn 0 0 DN4_dn 0
        #       dN1_dn dN1_de 0 DN2_dn DN2_de 0 DN3_dn DN3_de 0 DN4_dn DN4_de 0
        #       0 0 DN1_de 0 0 DN2_de 0 0 DN3_de 0 0 DN4_de
        #       0 0 DN1_dn 0 0 DN2_dn 0 0 DN3_dn 0 0 DN4_dn
        #       0 0 0 0 0 0 0 0 0 0 0 0]
        # (dN is usually called the B matrix)
        # chain rule:
        dN_dx = np.zeros((4, 2))
        dN_dx[0, :] = J_inv @ np.array([dN1_de, dN1_dn])
        dN_dx[1, :] = J_inv @ np.array([dN2_de, dN2_dn])
        dN_dx[2, :] = J_inv @ np.array([dN3_de, dN3_dn])
        dN_dx[3, :] = J_inv @ np.array([dN4_de, dN4_dn])
        
        dN = np.zeros((6, 12))
        dN[0, :] = [dN_dx[0,0], 0, 0, dN_dx[1,0], 0, 0, dN_dx[2,0], 0, 0, dN_dx[3,0], 0, 0]
        dN[1, :] = [0, dN_dx[0,1], 0, 0, dN_dx[1,1], 0, 0, dN_dx[2,1], 0, 0, dN_dx[3,1], 0]
        dN[2, :] = [dN_dx[0,1], dN_dx[0,0], 0, dN_dx[1,1], dN_dx[1,0], 0, dN_dx[2,1], dN_dx[2,0], 0, dN_dx[3,1], dN_dx[3,0], 0]
        dN[3, :] = [0, 0, dN_dx[0,0], 0, 0, dN_dx[1,0], 0, 0, dN_dx[2,0], 0, 0, dN_dx[3,0]]
        dN[4, :] = [0, 0, dN_dx[0,1], 0, 0, dN_dx[1,1], 0, 0, dN_dx[2,1], 0, 0, dN_dx[3,1]]
        dN[5, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        raise ValueError(f"Unsupported element_type ({element_type}) in evaluate_gauss_point().")
    return N, dN, det_J

def constitutive_matrix_isotropic(E, v):
    """Generates isotropic elastic constitutive matrix. All inputs should be np.float64"""
    G = E / (2 * (1 + v))
    Q1 = (E * (1 - v)) / ((1 + v) * (1 - 2 * v))
    Q2 = (E * v) / ((1 + v) * (1 - 2 * v))
    D = np.zeros((6, 6))
    normal_idx = [0, 1, 5]   # eps11, eps22, eps33
    for i in normal_idx:
        for j in normal_idx:
            D[i, j] = Q1 if i == j else Q2
    D[2, 2] = G   # gamma12
    D[3, 3] = G   # gamma13
    D[4, 4] = G   # gamma23
    return D

def constitutive_matrix_orthotropic(E1, E2, E3, v12, v13, v23, G12, G13, G23):
    """Generates orthotropic elastic constitutive matrix. All inputs should be np.float64"""
    v21 = v12 * E2 / E1
    v31 = v13 * E3 / E1
    v32 = v23 * E3 / E2
    C_normal = np.zeros((3, 3))
    C_normal[0, :] = [1 / E1, -v21 / E2, -v31 / E3]
    C_normal[1, :] = [-v12 / E1, 1 / E2, -v32 / E3]
    C_normal[2, :] = [-v13 / E1, -v23 / E2, 1 / E3]
    D_normal = np.linalg.inv(C_normal)
    D = np.zeros((6, 6))
    normal_idx = [0, 1, 5]   # eps11, eps22, eps33
    for a, i in enumerate(normal_idx):
        for b, j in enumerate(normal_idx):
            D[i, j] = D_normal[a, b]
    D[2, 2] = G12
    D[3, 3] = G13
    D[4, 4] = G23
    return D

def elemental_matrices(element_type,D,nodes):
    """Generates elemental matrices Mi, Ci, Ei, Li & Ri."""
    S = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]])
    DS = D @ S
    SDS = S.T @ DS
    if element_type == 3: # bilinear quad
        gp0 = 1.0 / np.sqrt(3.0)
        gps = np.array([
            [-gp0, -gp0],
            [gp0,  -gp0],
            [gp0,   gp0],
            [-gp0,  gp0]])
        Mi = np.zeros((12,12))
        Ci = np.zeros((12,12))
        Ei = np.zeros((12,12))
        Li = np.zeros((12,6))
        Ri = np.zeros((12,6))
        for gp in gps:
            N, dN, det_J = evaluate_gauss_point(element_type, nodes, gp)
            Nvals = N[0, 0::3] # grab [N1,N2,N3,N4]
            x1_gp = Nvals @ nodes[:, 0]
            x2_gp = Nvals @ nodes[:, 1]
            Z = np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, -x2_gp],
                [0.0, 1.0, 0.0, 0.0, 0.0, x1_gp],
                [0.0, 0.0, 1.0, x2_gp, -x1_gp, 0.0]])
            Mi += N.T @ SDS @ N * det_J # * w but gauss point weight = 1.0
            Ci += dN.T @ DS @ N * det_J # * w but gauss point weight = 1.0
            Ei += dN.T @ D @ dN * det_J # * w but gauss point weight = 1.0
            Li += N.T @ SDS @ Z * det_J # * w but gauss point weight = 1.0
            Ri += dN.T @ DS @ Z * det_J # * w but gauss point weight = 1.0
    else:
        raise ValueError(f"Unsupported element_type ({element_type}) in elemental_matrices().")

    return Mi, Ci, Ei, Li, Ri

def evaluate_SC_TC(K):
    """Evaluates shear center and tension center."""
    Kt = K[0:2, 0:2]
    Kc = K[0:2, 3:5]
    Kt_inv = np.linalg.inv(Kt)
    Y = -Kt_inv @ Kc
    xs = np.array([-Y[1, 2], Y[0, 2]])
    xt = np.array([Y[2, 1], -Y[2, 0]])
    return xs, xt

def evaluate_stiffness():
    """Main wrapper function."""
    node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords_flat.reshape(-1, 3)[:, :2]
    node_id_to_index = {tag: i for i, tag in enumerate(node_tags)}
    n_dof = 3 * len(node_tags)

    #M_global = np.zeros((n_dof, n_dof))
    M_global = lil_matrix((n_dof, n_dof), dtype=np.float64)
    #C_global = np.zeros((n_dof, n_dof))
    C_global = lil_matrix((n_dof, n_dof), dtype=np.float64)
    #E_global = np.zeros((n_dof, n_dof))
    E_global = lil_matrix((n_dof, n_dof), dtype=np.float64)
    #L_global = np.zeros((n_dof, 6))
    L_global = lil_matrix((n_dof, 6), dtype=np.float64)
    #R_global = np.zeros((n_dof, 6))
    R_global = lil_matrix((n_dof, 6), dtype=np.float64)

    for dim, group_tag in gmsh.model.getPhysicalGroups(2):
        name = gmsh.model.getPhysicalName(dim, group_tag)
        # D = material_D_matrices[name]   # look up this component's D once per group
        D = constitutive_matrix_isotropic(10e9, .3)

        for surf_tag in gmsh.model.getEntitiesForPhysicalGroup(dim, group_tag):
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim, surf_tag)
            for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
                nodes_per_elem = len(enodes) // len(etags)
                enodes = enodes.reshape(-1, nodes_per_elem)

                for et, local_node_tags in zip(etags, enodes):
                    local_idx = [node_id_to_index[t] for t in local_node_tags]
                    elem_coords = node_coords[local_idx]

                    Mi, Ci, Ei, Li, Ri = elemental_matrices(3, D, elem_coords)

                    dof_map = []
                    for ni in local_idx:
                        dof_map.extend([3*ni, 3*ni+1, 3*ni+2])

                    for a in range(12):
                        for b in range(12):
                            M_global[dof_map[a], dof_map[b]] += Mi[a, b]
                            C_global[dof_map[a], dof_map[b]] += Ci[a, b]
                            E_global[dof_map[a], dof_map[b]] += Ei[a, b]
                        for b in range(6):
                            L_global[dof_map[a], b] += Li[a, b]
                            R_global[dof_map[a], b] += Ri[a, b]

    M_global = M_global.tocsr()
    C_global = C_global.tocsr()
    E_global = E_global.tocsr()
    L_global = L_global.tocsr()
    R_global = R_global.tocsr()

    np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=2, suppress=True)
    print(E_global)

    #np.savetxt("global_stiffness_matrix.txt", E_global, fmt="%8.2f")

    return
