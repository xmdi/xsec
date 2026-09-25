import numpy as np
import gmsh
from scipy.sparse import csc_matrix, bmat, lil_matrix
from scipy.sparse.linalg import splu

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

negative_count=0

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
            dA = abs(det_J)
            Nvals = N[0, 0::3] # grab [N1,N2,N3,N4]
            x1_gp = Nvals @ nodes[:, 0]
            x2_gp = Nvals @ nodes[:, 1]
            Z = np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, -x2_gp],
                [0.0, 1.0, 0.0, 0.0, 0.0, x1_gp],
                [0.0, 0.0, 1.0, x2_gp, -x1_gp, 0.0]])
            Mi += N.T @ SDS @ N * dA # * w but gauss point weight = 1.0
            Ci += dN.T @ DS @ N * dA # * w but gauss point weight = 1.0
            Ei += dN.T @ D @ dN * dA # * w but gauss point weight = 1.0
            Li += N.T @ SDS @ Z * dA # * w but gauss point weight = 1.0
            Ri += dN.T @ DS @ Z * dA # * w but gauss point weight = 1.0
            
            global negative_count
            if det_J < 0:
                negative_count += 1
    else:
        raise ValueError(f"Unsupported element_type ({element_type}) in elemental_matrices().")

    return Mi, Ci, Ei, Li, Ri

def evaluate_SC_TC(K):
    """Evaluates shear center and tension center."""
    Kt = K[0:3, 0:3]
    Kc = K[0:3, 3:6]
    Kt_inv = np.linalg.inv(Kt)
    Y = -Kt_inv @ Kc
    xs = np.array([-Y[1, 2], Y[0, 2]])
    xt = np.array([Y[2, 1], -Y[2, 0]])
    return xs, xt

def solve_bordered(lu,rhs_n,n):
    rhs = np.zeros(n + 4)
    rhs[:n] = rhs_n
    sol = lu.solve(rhs)
    return sol[:n], sol[n:]   # (d, lambda)

def rigid_rotation_mode(node_coords, axis):
    """axis='e1' -> rotation about e1; axis='e2' -> rotation about e2"""
    n_nodes = node_coords.shape[0]
    n_dof = 3 * n_nodes
    c0 = np.zeros(n_dof)
    for k in range(n_nodes):
        x1, x2 = node_coords[k]
        idx = 3*k
        if axis == 'e2':
            c0[idx+0] = -0     # rotation about e2: u3 = -x1*omega... but x3 not in-plane
            # Rotation about e2 axis: displacement field is u1 = x3*omega (not in-plane var),
            # u3 = -x1*omega. Since we're at a fixed cross-section (x3=0 reference), the
            # in-plane rigid-rotation-about-e2 mode affects only u3:
            c0[idx+2] = -x1
        elif axis == 'e1':
            c0[idx+2] = x2     # rotation about e1: u3 = x2*omega
    return c0

def build_b(M, H, c10, c20, c0_i1, c0_i2):
    rhs = -M @ c0_i1 + H @ c0_i2
    return np.array([c10 @ rhs, c20 @ rhs])

def evaluate_stiffness():
    """Main wrapper function."""
    node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords_flat.reshape(-1, 3)[:, :2]
    node_id_to_index = {tag: i for i, tag in enumerate(node_tags)}
    n_dof = 3 * len(node_tags)
    n_nodes = len(node_tags)

    #M_global = np.zeros((n_dof, n_dof))
    M_global = lil_matrix((n_dof, n_dof), dtype=np.float64)
    #C_global = np.zeros((n_dof, n_dof))
    C_global = lil_matrix((n_dof, n_dof), dtype=np.float64)
    #E_global = np.zeros((n_dof, n_dof))
    E_global = lil_matrix((n_dof, n_dof), dtype=np.float64)
    L_global = np.zeros((n_dof, 6))
    #L_global = lil_matrix((n_dof, 6), dtype=np.float64)
    R_global = np.zeros((n_dof, 6))
    #R_global = lil_matrix((n_dof, 6), dtype=np.float64)

    for dim, group_tag in gmsh.model.getPhysicalGroups(2):
        name = gmsh.model.getPhysicalName(dim, group_tag)
        # D = material_D_matrices[name]   # look up this component's D once per group
        D = constitutive_matrix_isotropic(10e9, .3)

        for surf_tag in gmsh.model.getEntitiesForPhysicalGroup(dim, group_tag):
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim, surf_tag)
            print(len(elem_tags))
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

    #M_global = M_global.tocsr()
    #C_global = C_global.tocsr()
    #E_global = E_global.tocsr()
    #L_global = L_global.tocsr()
    #R_global = R_global.tocsr()

    M = csc_matrix(M_global)
    E = csc_matrix(E_global)
    H = csc_matrix(C_global - C_global.T)     # Eq. (8): H = C - C^T
    C = csc_matrix(C_global)
    #L = np.asarray(L_global.todense()) if hasattr(L, "todense") else np.asarray(L)
    #R = np.asarray(R_global.todense()) if hasattr(R, "todense") else np.asarray(R)

    c10 = np.zeros(n_dof)  # translation along e3
    c20 = np.zeros(n_dof)  # rotation about e3
    c30 = np.zeros(n_dof)  # translation along e1
    c40 = np.zeros(n_dof)  # translation along e2

    for k in range(n_nodes):
        x1, x2 = node_coords[k]
        idx = 3*k
        c10[idx+2] = 1.0            # u3 = 1
        c20[idx+0] = -x2            # u1 = -x2*omega
        c20[idx+1] =  x1            # u2 =  x1*omega
        c30[idx+0] = 1.0            # u1 = 1
        c40[idx+1] = 1.0            # u2 = 1

    U = np.vstack([c10, c20, c30, c40])   # (4, n) constraint matrix

    for name, c0 in [("c10", c10), ("c20", c20), ("c30", c30), ("c40", c40)]:
        resid = np.linalg.norm(E @ c0)
        print(f"{name}: |E @ c0| = {resid:.3e} (should be ~0)")

    U_sp = csc_matrix(U)
    K_bordered = bmat([[E, U_sp.T], [U_sp, None]], format='csc')

    lu = splu(K_bordered)

    c11, lam11 = solve_bordered(lu, -H @ c10, n_dof)   # Eq. 12 for traction chain
    c21, lam21 = solve_bordered(lu, -H @ c20, n_dof)   # Eq. 12 for torsion chain

    c0_31 = rigid_rotation_mode(node_coords, axis='e2')   # bending chain 3 (starts from c30)
    c0_41 = rigid_rotation_mode(node_coords, axis='e1')   # bending chain 4 (starts from c40)

    c0_31, lam0_31 = solve_bordered(lu, -H @ c30, n_dof)
    c0_41, lam0_41 = solve_bordered(lu, -H @ c40, n_dof)
    print("|lam0_31|:", np.linalg.norm(lam0_31), " |lam0_41|:", np.linalg.norm(lam0_41))
    print("c0_31 Eq(12) residual:", np.linalg.norm(E @ c0_31 - (-H @ c30)))
    print("c0_41 Eq(12) residual:", np.linalg.norm(E @ c0_41 - (-H @ c40)))


    # sanity check: these should also be exactly in the nullspace of E
    print("E @ c0_31:", np.linalg.norm(E @ c0_31))
    print("E @ c0_41:", np.linalg.norm(E @ c0_41))

    c0_32, lam0_32 = solve_bordered(lu, -H @ c0_31 + M @ c30, n_dof)
    c0_42, lam0_42 = solve_bordered(lu, -H @ c0_41 + M @ c40, n_dof)

    # Build A per Eq. (15)
    top = np.vstack([c10, c20])                       # (2, n)
    MH  = np.hstack([M.toarray() if hasattr(M,'toarray') else M,
                  -H.toarray() if hasattr(H,'toarray') else -H])  # (n, 2n) -- careful with shapes

    Mc10, Mc20 = M @ c10, M @ c20
    Hc11, Hc21 = H @ c11, H @ c21
    col0 = Mc10 - Hc11   # M*c10 - H*c11  (matches [M -H] @ [c10; c11])
    col1 = Mc20 - Hc21   # M*c20 - H*c21
    A = np.array([
        [c10 @ col0, c10 @ col1],
        [c20 @ col0, c20 @ col1]])

    b1 = build_b(M, H, c10, c20, c0_31, c0_32)
    b2 = build_b(M, H, c10, c20, c0_41, c0_42)

    n1 = np.linalg.solve(A, b1)   # Eq. 14
    n2 = np.linalg.solve(A, b2)

        # Update per Eq. (17)
    c31 = c0_31 + c10*n1[0] + c20*n1[1]
    c32 = c0_32 + c11*n1[0] + c21*n1[1]

    c41 = c0_41 + c10*n2[0] + c20*n2[1]
    c42 = c0_42 + c11*n2[0] + c21*n2[1]

    c33, lam33 = solve_bordered(lu, -H @ c32 + M @ c31, n_dof)
    c43, lam43 = solve_bordered(lu, -H @ c42 + M @ c41, n_dof)

    print("det(A):", np.linalg.det(A))
    print("cond(A):", np.linalg.cond(A))
    print("|n1|:", np.linalg.norm(n1), " |n2|:", np.linalg.norm(n2))
    print("|c31|:", np.linalg.norm(c31), " |c32|:", np.linalg.norm(c32))
    print("|c41|:", np.linalg.norm(c41), " |c42|:", np.linalg.norm(c42))
    print("|c33|:", np.linalg.norm(c33), " |c43|:", np.linalg.norm(c43))

    print("|c10|:", np.linalg.norm(c10), " |c11|:", np.linalg.norm(c11))
    print("|c20|:", np.linalg.norm(c20), " |c21|:", np.linalg.norm(c21))
    print("|c0_31|:", np.linalg.norm(c0_31), " |c0_32|:", np.linalg.norm(c0_32))
    print("|c0_41|:", np.linalg.norm(c0_41), " |c0_42|:", np.linalg.norm(c0_42))

    E_dense = E.toarray()
    eigE = np.linalg.eigvalsh(E_dense)
    print("6 smallest E eigenvalues:", np.sort(eigE)[:6])

    u3_block  = np.column_stack([c10, c20, c31, c32, c41, c42])   # (n,6)
    u_block = np.column_stack([c11, c21, c32, c33, c42, c43])   # (n,6)
    Qd = np.vstack([u3_block, u_block])                           # (2n,6), matches [u,3; u]

    Block = bmat([[M, C.T], [C, E]], format='csc')   # (2n,2n)
    BlockQd = Block @ Qd                              # (2n,6)

    lhs = Qd.T @ BlockQd                              # (6,6)
    LR = np.vstack([L_global, R_global])                            # (2n,6)
    rhs = Qd.T @ LR                                   # (6,6)

    G = np.linalg.solve(lhs, rhs)

    K = G.T @ lhs @ G   # (6,6), reusing lhs = Qd.T @ Block @ Qd from above
  
    zeros = np.zeros_like(c10)

    Qr = np.vstack([
        # top block: u,3
        np.column_stack([zeros, zeros, zeros, c30, zeros, c40]),
        # bottom block: u
        np.column_stack([c10,   c20,   c30,   c0_31, c40,  c0_41])
    ])

    BlockQr = Block @ Qr
    print("Rigid energy check (should be ~0):", np.abs(Qr.T @ BlockQr).max())

    eig_lhs = np.linalg.eigvalsh(lhs)
    print("lhs eigenvalues:", eig_lhs) 

    #assert np.allclose(K, K.T, atol=1e-6*np.abs(K).max())
    eigvals = np.linalg.eigvalsh(K)
    #assert np.all(eigvals > 0), f"Non-PD stiffness: {eigvals}"
    print("K33 (axial stiffness EA):", K[2,2])   # per this paper's t=[t1,t2,t3] ordering

    #np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=2, suppress=True)
    print(K)

    # Verify c0_42 actually solves its defining equation (Eq. 13, i=2, chain 4)
    resid_42 = E @ c0_42 - (-M @ c40 + H @ c0_41)  # or +H depending on your exact Eq(13) sign
    print("chain4 c0_42 residual:", np.linalg.norm(resid_42))

    # Compare directly against chain 3's equivalent
    resid_32 = E @ c0_32 - (-M @ c30 + H @ c0_31)
    print("chain3 c0_32 residual:", np.linalg.norm(resid_32))

    print("|lam0_32|:", np.linalg.norm(lam0_32))
    print("|lam0_42|:", np.linalg.norm(lam0_42))

    resid_c0_31 = E @ c0_31 - (-H @ c30)
    resid_c0_41 = E @ c0_41 - (-H @ c40)
    print("c0_31 Eq(12) residual:", np.linalg.norm(resid_c0_31), " relative to |E@c0_31|:", np.linalg.norm(E@c0_31))
    print("c0_41 Eq(12) residual:", np.linalg.norm(resid_c0_41), " relative to |E@c0_41|:", np.linalg.norm(E@c0_41))

    print("|H @ c0_31|:", np.linalg.norm(H @ c0_31))
    print("|H @ c0_41|:", np.linalg.norm(H @ c0_41))
    print("|M @ c30|:", np.linalg.norm(M @ c30))
    print("|M @ c40|:", np.linalg.norm(M @ c40))

    def check_orthogonality(vec, label):
        print(f"{label}: c10·v={c10@vec:.3e}  c20·v={c20@vec:.3e}  c30·v={c30@vec:.3e}  c40·v={c40@vec:.3e}")

    check_orthogonality(-H @ c30, "-H@c30")
    check_orthogonality(-H @ c40, "-H@c40")
   

    xs, xt = evaluate_SC_TC(K)
    print(xs)
    print(xt)

    #np.savetxt("global_stiffness_matrix.txt", E_global, fmt="%8.2f")

    return
