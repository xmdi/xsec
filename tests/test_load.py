import gmsh
import math
import numpy as np
from xsec.fe_core import evaluate_stiffness, constitutive_matrix_isotropic, evaluate_strain_stress
import time
import pyvista as pv

def test_load():
    """Evaluates a thin-walled semicircular steel tube under load."""

    mesh_start = time.perf_counter()

    gmsh.initialize()
    gmsh.model.add("semicircle")
    gmsh.option.setNumber("General.Terminal", 1)

    R = 1
    thickness = 0.1
    circumferential_elements = 100
    thickness_elements = 5

    left_IML_pt = gmsh.model.occ.addPoint(-R + thickness / 2, 0, 0)
    left_OML_pt = gmsh.model.occ.addPoint(-R - thickness / 2, 0, 0)
    right_IML_pt = gmsh.model.occ.addPoint(R - thickness / 2, 0, 0)
    right_OML_pt = gmsh.model.occ.addPoint(R + thickness / 2, 0, 0)
    center_pt = gmsh.model.occ.addPoint(0, 0, 0)

    IML = gmsh.model.occ.addCircleArc(left_IML_pt, center_pt, right_IML_pt)
    OML = gmsh.model.occ.addCircleArc(left_OML_pt, center_pt, right_OML_pt)
    left_edge = gmsh.model.occ.addLine(left_IML_pt, left_OML_pt) 
    right_edge = gmsh.model.occ.addLine(right_IML_pt, right_OML_pt) 

    boundary = gmsh.model.occ.addWire([left_edge, IML, right_edge, OML])
    surface = gmsh.model.occ.addPlaneSurface([boundary])

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setTransfiniteCurve(IML, circumferential_elements + 1)
    gmsh.model.mesh.setTransfiniteCurve(OML, circumferential_elements + 1)
    gmsh.model.mesh.setTransfiniteCurve(left_edge, thickness_elements + 1)
    gmsh.model.mesh.setTransfiniteCurve(right_edge, thickness_elements + 1)
    gmsh.model.mesh.setTransfiniteSurface(surface)

    gmsh.model.mesh.setRecombine(2, surface)
    gmsh.model.mesh.generate(2)

    mat_id="steel"

    gmsh.model.addPhysicalGroup(2, [surface], name = mat_id)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.model.mesh.renumberElements()

    """
    elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(dim = 2)
    for e_type, tags, nodes in zip(elementTypes, elementTags, nodeTags):
        if e_type == 3: # quads
            quad_tags = np.array(tags)
            quad_nodes = np.array(nodes).reshape(-1, 4)
            number_elements = len(quad_tags)
            for idx in range(len(quad_tags)):
                element_id = quad_tags[idx]
                real_node_ids = quad_nodes[idx]
                print(f"Quad Element ID: {element_id} | Node IDs: {real_node_ids.tolist()}")
                for node_id in real_node_ids:
                    coord, parametricCoord, dim, tag = gmsh.model.mesh.getNode(node_id)
                    print(f"Node Element ID: {node_id} @ ({coord[0]}, {coord[1]}, {coord[2]})")
    
    np.testing.assert_allclose([number_elements], [500], rtol=1e-5)
    """

    mesh_done = time.perf_counter()

    # steel props
    E = 200e9 # modulus
    v = .3 # poisson's ratio
    D = constitutive_matrix_isotropic(E, v)
    D_list={mat_id:D}

    K, G, Qd, mesh_elements, tag_to_idx, node_coords, n_dof = evaluate_stiffness(D_list)

    stiffness_done = time.perf_counter()

    theta = np.array([0, 0, 0, 1000000, 0, 0])
    results, u_global, u3_global = evaluate_strain_stress(theta, K, G, Qd, node_coords, mesh_elements, D_list, tag_to_idx, n_dof)

    scale = 1000

    u_per_node = u_global.reshape(-1, 3)   # (n_nodes, 3) -- [u1, u2, u3] per node, global order
    deformed_coords_2d = node_coords + scale * u_per_node[:, :2]   # (n_nodes, 2)
    node_coords_3d = np.column_stack([node_coords, np.zeros(len(node_coords))])   # add x3=0 reference
    deformed_coords_3d = node_coords_3d + scale * u_per_node   # (n_nodes, 3)

    print("max |u1|:", np.abs(u_per_node[:,0]).max())
    print("max |u2|:", np.abs(u_per_node[:,1]).max())
    print("max |u3|:", np.abs(u_per_node[:,2]).max())

 
    print(u_per_node)

    cells_2d = np.array([
        np.concatenate([[4], tag_to_idx[local_node_tags]])
        for etype, mat_name, local_node_tags in mesh_elements
        ], dtype=np.int64)
    cells_flat = cells_2d.flatten()   # 1D, required by pv.UnstructuredGrid

    load_done = time.perf_counter()

    mesh_time = mesh_done - mesh_start
    stiffness_time = stiffness_done - mesh_done
    load_time = load_done - stiffness_done

    print(f"Meshing took {mesh_time:.6f} seconds to complete.")
    print(f"Sectional analysis took {stiffness_time:.6f} seconds to complete.")
    print(f"Load application took {load_time:.6f} seconds to complete.")
  
    gmsh.finalize()

    cell_types = np.ones(len(cells_2d), dtype=np.uint8) * pv.CellType.QUAD
    grid_undeformed = pv.UnstructuredGrid(cells_flat, cell_types, node_coords_3d)
    grid_deformed = pv.UnstructuredGrid(cells_flat, cell_types, deformed_coords_3d)

    element_values = np.sin(grid_undeformed.cell_centers().points[:, 0]) * np.cos(grid_undeformed.cell_centers().points[:, 1])

    grid_undeformed.cell_data["Breakdown Value"] = element_values

    plotter = pv.Plotter()

    plotter.add_mesh(
            grid_undeformed,
            color = "lightgray",
            opacity = 0.9,
            show_edges = True,
            edge_color = "gray",
            label = "Undeformed"
    )

    plotter.add_mesh(
            grid_deformed,
            scalars = element_values,
            cmap = "turbo",
            show_edges = True,
            edge_color = "black",
            label = "Deformed"
    )

    critical_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 5.0]])
    plotter.add_points(critical_points, color = "red", point_size = 15, render_points_as_spheres = True)

    labels = ["Origin Layer 1", "Origin Layer 2"]
    plotter.add_point_labels(critical_points, labels, font_size = 30, point_color = "red", text_color = "white")
    
    depths = [0.0, 3.0, 6.0]

    for z_depth in depths:
        triad = pv.AxesAssembly(
                position = (0, 0, z_depth),
                show_labels = False,
                shaft_radius = 0.015,
                tip_radius = 0.05
        )
        triad.scale = (0.4, 0.4, 0.4)
        plotter.add_actor(triad)

    u_per_node = u_global.reshape(-1, 3)
    deformed = node_coords + scale * u_per_node[:, :2]   # in-plane deformed shape only

    tol = 1e-6
    max_err = 0.0
    for i in range(len(node_coords)):
        x1, x2 = node_coords[i]
        if x1 <= 0:
            continue
        matches = np.where((np.abs(node_coords[:,0] + x1) < tol) & (np.abs(node_coords[:,1] - x2) < tol))[0]
        if len(matches) == 0:
            continue
        j = matches[0]

        # expected mirror relationship for m1 bending (antisymmetric u3, TBD for u1/u2):
        d1 = deformed[i,0] - (-deformed[j,0])   # is deformed[i].x1 = -deformed[j].x1 ?
        d2 = deformed[i,1] - deformed[j,1]       # is deformed[i].x2 = deformed[j].x2 ?
        max_err = max(max_err, abs(d1), abs(d2))

    print("max in-plane mirror-symmetry error (scaled):", max_err)
    print("relative to scale factor:", max_err/scale)

    print("max |u1|, |u2| (unscaled):", np.abs(u_per_node[:,0]).max(), np.abs(u_per_node[:,1]).max())
    print("max_err relative to actual displacement magnitude:", max_err/scale / np.abs(u_per_node[:,:2]).max())

    print("K[0,3]:", K[0,3], " K[3,0]:", K[3,0])  # should be ~0 for mirror-symmetric section
    print("relative to K.max():", K[0,3]/np.abs(K).max())

    print("K=",K)

    # For each mirror pair, check if the UNDEFORMED mesh coordinates themselves are exactly symmetric
    mesh_asym = []
    for i in range(len(node_coords)):
        x1, x2 = node_coords[i]
        if x1 <= 0:
            continue
        matches = np.where((np.abs(node_coords[:,0] + x1) < 1e-6) & (np.abs(node_coords[:,1] - x2) < 1e-6))[0]
        if len(matches) == 0:
            mesh_asym.append((i, x1, x2))   # no exact mirror partner found at all!

    print(f"Number of nodes with no exact mirror partner: {len(mesh_asym)} out of {len(node_coords)}")

    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.add_legend()
    plotter.show()





if __name__ == "__main__":
    test_load()
