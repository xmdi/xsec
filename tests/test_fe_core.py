import gmsh
import math
import numpy as np
from xsec.fe_core import evaluate_stiffness

def test_fe_core():

    gmsh.initialize()
    gmsh.model.add("semicircle")
    gmsh.option.setNumber("General.Terminal", 1)

    thickness = 0.1
    circumferential_elements = 100
    thickness_elements = 5

    left_IML_pt = gmsh.model.occ.addPoint(-1, 0, 0)
    left_OML_pt = gmsh.model.occ.addPoint(-1 - thickness, 0, 0)
    right_IML_pt = gmsh.model.occ.addPoint(1, 0, 0)
    right_OML_pt = gmsh.model.occ.addPoint(1 + thickness, 0, 0)
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

    gmsh.model.addPhysicalGroup(2, [surface], name = "semicircle_quads")
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.model.mesh.renumberElements()

    gmsh.write("semicircle.msh") # use this for the pyvista test as well

    number_elements = -1;

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


    evaluate_stiffness()

    gmsh.finalize()

if __name__ == "__main__":
    test_fe_core()
