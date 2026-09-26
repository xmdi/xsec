import gmsh
import math
import numpy as np
from xsec.fe_core import evaluate_stiffness, constitutive_matrix_isotropic, evaluate_SC_TC, evaluate_decoupled_K
import time

#import cProfile, pstats

def test_fe_core():
    """Evaluates a thin-walled semicircular steel tube against analytical values."""

    mesh_start = time.perf_counter()

    gmsh.initialize()
    gmsh.model.add("semicircle")
    gmsh.option.setNumber("General.Terminal", 1)

    R = 1
    thickness = 0.01
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
 
    #profiler = cProfile.Profile()
    #K = profiler.runcall(evaluate_stiffness, D_list)
    #profiler.dump_stats('profile_out')

    #p = pstats.Stats('profile_out')
    #p.sort_stats('cumulative').print_stats(15)

    K = evaluate_stiffness(D_list)

    stiffness_done = time.perf_counter()


    mesh_time = mesh_done - mesh_start
    stiffness_time = stiffness_done - mesh_done

    print(f"Meshing took {mesh_time:.6f} seconds to complete.")
    print(f"Sectional analysis took {stiffness_time:.6f} seconds to complete.")
  
    gmsh.finalize()

    # analytical values, for comparison
    A = math.pi * R * thickness # area
    x_bar = 0 # x-centroid (symmetric)
    y_bar = 2 * R / math.pi # y-centroid
    xt_theory = [x_bar, y_bar] # tension center is the mass center for homogenous isotropic cross-section
    Ixx = (math.pi / 2 - 4 / math.pi) * R**3 * thickness # vertical moment of inertia about the centroid
    #Ixx_origin = Ixx + A * y_bar**2
    Iyy = math.pi * R**3 * thickness / 2 # horizontal moment of inertia about the centroid (symmetric)
    x_sc = 0 # x shear center
    y_sc = 4 * R / math.pi # y shear center

    Asx = A / 2  # effective shear area in x
    Asy = (3 * (math.pi**2 -8)**2)/(2*math.pi*(5*math.pi**2 - 48)) * R * thickness
    xs_theory = [x_sc, y_sc] # shear center coordinates
    G = E / (2 * (1 + v))
    GJ = G * math.pi * R * thickness**3 / 3 

    T = np.array([ # transformation matrix to get values about the centroid
                [1.0, 0.0, 0.0, 0.0, 0.0, y_bar],
                [0.0, 1.0, 0.0, 0.0, 0.0, -x_bar],
                [0.0, 0.0, 1.0, -y_bar, x_bar, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
   
    # translate computed stiffness about the part centroid
    Kc = T.T @ K @ T

    K_prime = evaluate_decoupled_K(K)
    
    """
    print(f"Shear Stiffness in e1: {Kc[0,0]:.2e}, should be: {G*Asx:.2e}, error: {(Kc[0,0]-G*Asx)/(G*Asx)*100:.2f}%")
    print(f"Shear Stiffness in e2: {Kc[1,1]:.2e}, should be: {G*Asy:.2e}, error: {(Kc[1,1]-G*Asy)/(G*Asy)*100:.2f}%")
    print(f"Axial Stiffness in e3: {Kc[2,2]:.2e}, should be: {E*A:.2e}, error: {(Kc[2,2]-E*A)/(E*A)*100:.2f}%")
    print(f"Bending Stiffness in e1: {Kc[3,3]:.2e}, should be: {E*Ixx:.2e}, error: {(Kc[3,3]-E*Ixx)/(E*Ixx)*100:.2f}%")
    print(f"Bending Stiffness in e2: {Kc[4,4]:.2e}, should be: {E*Iyy:.2e}, error: {(Kc[4,4]-E*Iyy)/(E*Iyy)*100:.2f}%")
    print(f"Torsional Stiffness in e3: {K_prime[5,5]:.2e}, should be: {GJ:.2e}, error: {(K_prime[5,5]-GJ)/GJ*100:.2f}%")
    """
    
    xs, xt = evaluate_SC_TC(K)

    """print(f"Shear Center: {xs}, should be: {xs_theory}")
    print(f"Tension Center: {xt}, should be: {xt_theory}")
    """
    # print(K)
    # print(K_prime)

    np.testing.assert_allclose([xs,xt], [xs_theory,xt_theory], atol=1e-2)
    np.testing.assert_allclose([Kc[0,0],Kc[1,1],Kc[2,2],Kc[3,3],Kc[4,4],K_prime[5,5]], [G*Asx, G*Asy, E*A, E*Ixx, E*Iyy, GJ], rtol=2e-3)


if __name__ == "__main__":
    test_fe_core()
