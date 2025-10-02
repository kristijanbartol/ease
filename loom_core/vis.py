import numpy as np
import pyvista as pv


def render_circumference_bands(
    base_mesh,              # (V,F) or pv.PolyData
    slice_polylines,        # list of (k_i,3) arrays
    diffs,                  # len == len(slice_polylines), e.g. percent deltas
    stride=1,               # draw every Nth slice to reduce clutter
    tube_radius=None,       # if None, auto (fraction of bbox diag)
    offset_factor=0.004,    # offset as fraction of bbox diag (push bands off surface)
    cmap_name="coolwarm_r",
    show=True
):
    # -- Build/accept PolyData for the base mesh
    if isinstance(base_mesh, pv.PolyData):
        base_pv = base_mesh
    else:
        V, F = base_mesh.vertices, base_mesh.faces
        if F.ndim != 2 or F.shape[1] not in (3, 4):
            raise ValueError("Faces F must be (m,3) triangles or (m,4) quads.")
        faces = np.hstack([np.full((F.shape[0], 1), F.shape[1], dtype=np.int64), F]).ravel()
        base_pv = pv.PolyData(V, faces)

    # Geometry scale for sensible defaults
    xmin, xmax, ymin, ymax, zmin, zmax = base_pv.bounds
    diag = np.linalg.norm([xmax-xmin, ymax-ymin, zmax-zmin])
    if tube_radius is None:
        #tube_radius = 0.008 * diag          # ~0.8% of bbox diagonal (≈ 1–2 cm on a human)
        tube_radius = 0.003 * diag
    offset = float(offset_factor) * diag

    diffs = np.asarray(diffs, dtype=float)
    vmax = float(np.max(np.abs(diffs))) / 2 if diffs.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    vmax = 0.1
    clim = (-vmax, vmax)

    # Base mesh: faint + silhouette to frame the scene
    pl = pv.Plotter()
    pl.set_background("white")
    pl.add_mesh(base_pv, color="#cfcfcf", smooth_shading=True, opacity=0.18)
    pl.add_mesh(base_pv, color="black", silhouette=True, line_width=1.0)

    # Normals for offsetting slice points
    if "Normals" not in base_pv.point_data:
        base_pv = base_pv.compute_normals(cell_normals=False, point_normals=True)
    base_normals = base_pv.point_data["Normals"]

    def offset_points_along_surface(pts):
        # push points slightly along the base-mesh normals of nearest vertices
        offs = np.empty_like(pts)
        for i, p in enumerate(pts):
            idx = int(base_pv.find_closest_point(p))
            n = base_normals[idx]
            offs[i] = p + offset * n
        return offs

    first_bar = True
    for i, (pts, d) in enumerate(zip(slice_polylines, diffs)):
        if i % max(1, stride) != 0:
            continue
        if pts is None or len(pts) < 2:
            continue

        pts_off = offset_points_along_surface(np.asarray(pts))
        line = pv.Spline(pts_off, n_points=len(pts_off))
        tube = line.tube(radius=tube_radius, capping=True, n_sides=24)
        tube.point_data["diff"] = np.full(tube.n_points, d, dtype=float)

        pl.add_mesh(
            tube,
            scalars="diff",
            cmap=cmap_name,
            clim=clim,
            lighting=False,         # keep colors crisp, not shaded
            opacity=1.0,
            show_scalar_bar=first_bar,
            scalar_bar_args=dict(
                title="Circumference Δ (%)",
                vertical=True,
                n_labels=5,
            ),
        )
        first_bar = False

    # Rendering quality for translucency/overdraw
    pl.enable_depth_peeling()
    pl.enable_eye_dome_lighting()
    pl.add_axes(line_width=2)
    pl.enable_parallel_projection()   # orthographic = fewer perspective artifacts

    if show:
        pl.show()
    return pl

