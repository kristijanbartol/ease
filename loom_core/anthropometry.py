import numpy as np
import trimesh
import os
from smplx import SMPL
import torch
from itertools import product

from loom.const import a_pose


SMPL_START_V = 3500
SMPL_STOP_V = 6599
SMPL_START_OFFSET = 0.03

SCAN_START_V = 9198
SCAN_STOP_V = 26868
SCAN_START_OFFSET = 0.025

SLICE_OFFSET = 0.02

N_SLICES = 100

SEARCH_GRID = [
    [0.79], # 1.15 #torch.linspace(0.95, 0.96, 100), #torch.linspace(0.79, 0.81, 250), #[0.7980],
    [1.1], # 0.6 #torch.linspace(0.55, 0.79, 100), #torch.linspace(0.865, 0.88, 15), #[0.8737],
    [0.7], # 0.7 #torch.linspace(-0.69, -0.71, 20),    # -0.7020
    [2.6], #torch.linspace(2.0, 3.0, 100), #torch.linspace(-0.2, 3.0, 100), #torch.linspace(0.0, 2.0, 100),
    [0.5], #torch.linspace(0.5, 2.0, 100), #torch.linspace(0.0, 2.0, 100), #torch.linspace(-1.0, 1.0, 100),
    [0.0], #torch.linspace(-0.6, 0.3, 20), #torch.linspace(0.0, 0.0, 1),
    [-0.45], #torch.linspace(-0.7, 0.0, 50),
    [-0.6], #torch.linspace(-2.0, 0.0, 20),
    [0.0],
    [0.0],
]
# Normalize to 1D float tensors (no data duplication)
grids = [
    g if torch.is_tensor(g) else torch.tensor(g, dtype=torch.float32)
    for g in SEARCH_GRID
]
grid_lengths = [len(g) for g in grids]

SMPL_DIR = '/home/kristijan/data/smpl/models/'
SMPL_MALE = SMPL(
    model_path=os.path.join(SMPL_DIR, f'SMPL_MALE.pkl'), 
    gender='male'
)


def slice_mesh(mesh, y0, normal_dir=1):
    sliced_verts, sliced_faces, _ = trimesh.intersections.slice_faces_plane(mesh.vertices, mesh.faces, plane_normal=[0, normal_dir, 0], plane_origin=[0, y0, 0])
    sliced_mesh = trimesh.Trimesh(vertices=sliced_verts, faces=sliced_faces)
    parts = sliced_mesh.split(only_watertight=False)
    cut = max(parts, key=lambda m: m.area)
    return cut


def circumference_at_y(mesh, y):
    cs = mesh.section([0, 1, 0], [0, y, 0])
    return cs.to_2D()[0].length


def circumference_at_y_debug(mesh, y):
    cs = mesh.section([0, 1, 0], [0, y, 0])
    verts_3d = cs.to_2D()[0].to_3D().vertices
    return cs.to_2D()[0].length, verts_3d


def extract_slice_locations(verts, start_v, stop_v, start_offset):
    start_y = verts[start_v][1] + start_offset
    stop_y = verts[stop_v][1]
    return start_y, stop_y


def rotate_scan(mesh):
    R = trimesh.transformations.rotation_matrix(
        -np.pi / 2, [1, 0, 0], mesh.centroid
    )
    mesh.apply_transform(R)
    return mesh


def loss_mean(circums_a, in_seam_a, circums_b, in_seam_b):
    return float(np.mean(np.abs(circums_a - circums_b))), float(abs(in_seam_a - in_seam_b))


def extract_measurements(mesh, start_v=SMPL_START_V, stop_v=SMPL_STOP_V, start_offset=SMPL_START_OFFSET, slice_offset=SLICE_OFFSET, n_slices=N_SLICES):
    scan_start_y, scan_stop_y = extract_slice_locations(mesh.vertices, start_v, stop_v, start_offset)
    sliced_mesh = slice_mesh(mesh, scan_start_y + 0.02, -1)
    sliced_mesh = slice_mesh(sliced_mesh, scan_stop_y, 1)

    mesh_circums = []
    for y in np.linspace(scan_start_y, scan_stop_y, n_slices):
        mesh_circums.append(circumference_at_y(sliced_mesh, y))

    in_seam_height = scan_start_y - scan_stop_y
    return np.array(mesh_circums), in_seam_height


def get_smpl_mesh(betas, pose=a_pose(), smpl_model=SMPL_MALE):
    smpl_verts = smpl_model(
        body_pose=pose,
        betas=betas
    ).vertices[0].cpu().detach().numpy()
    return trimesh.Trimesh(vertices=smpl_verts, faces=smpl_model.faces)


def make_objective(scan_mesh, loss_f):
    circums_scan, in_seam_height_scan = extract_measurements(scan_mesh, start_v=SCAN_START_V, stop_v=SCAN_STOP_V, start_offset=SCAN_START_OFFSET, slice_offset=SLICE_OFFSET, n_slices=N_SLICES)
    def objective(beta):
        smpl_mesh = get_smpl_mesh(beta)
        circums_smpl, in_seam_height_smpl = extract_measurements(smpl_mesh)
        return loss_f(circums_scan, in_seam_height_scan, circums_smpl, in_seam_height_smpl)
    return objective


def grid_search():
    # Lazy Cartesian product over indices to avoid big Python lists
    for idxs in product(*[range(n) for n in grid_lengths]):
        betas = torch.stack([grids[d][i] for d, i in enumerate(idxs)]).unsqueeze(0)
        circums_loss, in_seam_loss = objective(betas)
        print(betas[0, 7], circums_loss, in_seam_loss)
        get_smpl_mesh(betas).export('fit.ply')


def coord_descent(objective, beta_hat, L=3, passes=3, tol=1e-3):
    def golden(f1d, a, b, iters=20):
        phi = (1 + 5**0.5) / 2
        c = b - (b-a)/phi; d = a + (b-a)/phi
        fc, fd = f1d(c), f1d(d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - (b-a)/phi; fc = f1d(c)
            else:
                a, c, fc = c, d, fd
                d = a + (b-a)/phi; fd = f1d(d)
        return (a+b)/2

    for _ in range(passes):
        beta_prev = beta_hat.clone()
        for i in range(len(beta_hat)):
            def f1d(t):
                x = beta_hat.clone()
                x[i] = max(-L, min(L, t))   # clamp
                return objective(x)
            beta_hat[i] = golden(f1d, -L, L)
        if torch.max(torch.abs(beta_hat - beta_prev)) < tol:
            break

        print(beta_hat)
    return beta_hat


if __name__ == '__main__':
    body_scan = rotate_scan(trimesh.load('/home/kristijan/Documents/eg-2026/Kristijan/3D-Mesh_001_M/A-POSE/PLY/ply_from_obj.ply'))
    objective = make_objective(body_scan, loss_mean)
    #coord_descent(objective, torch.zeros((1, 10)))
    grid_search()
