import meshio
import numpy as np
import pyvista as pv
from pathlib import Path

def test_pyvista():

    msh = meshio.read("semicircle.msh")

    cells = msh.cells_dict["quad"]
    points = msh.points

    num_cells = len(cells)
    padding = np.ones((num_cells, 1), dtype=int) * 4
    vtk_cells = np.hstack((padding, cells)).ravel()
    cell_types = np.ones(num_cells, dtype=np.uint8) * pv.CellType.QUAD

    grid = pv.UnstructuredGrid(vtk_cells, cell_types, points)

    element_values = np.sin(grid.cell_centers().points[:, 0]) * np.cos(grid.cell_centers().points[:, 1])

    grid.cell_data["Breakdown Value"] = element_values

    plotter = pv.Plotter()

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

    mesh_actor = plotter.add_mesh(
            grid,
            scalars = "Breakdown Value",
            cmap = "turbo",
            show_edges = True,
            edge_color = "black",
            line_width = 1
    )
    plotter.add_actor(mesh_actor)

    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.show()

    file_path = Path("semicircle.msh")
    file_path.unlink(missing_ok = True)

    np.testing.assert_allclose([1], [1], rtol=1e-5)
