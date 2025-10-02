import polyscope as ps
import numpy as np
import os
import trimesh

palette = {
    # 1) Slate (bolder blue)
    "solo-female_base_hoodie_long-base-1.1-1.1-0.0-0.0": [(0.10, 0.30, 0.95), (0.35, 0.55, 0.98)],
    "solo-female-ss_base_hoodie_long-base-1.1-1.1-0.0-0.0": [(0.10, 0.30, 0.95), (0.35, 0.55, 0.98)],
    "solo-female-ll_base_hoodie_long-base-1.1-1.1-0.0-0.0": [(0.10, 0.30, 0.95), (0.35, 0.55, 0.98)],

    # 2) Teal (punchier cyan-green)
    "solo-female_base_hoodie_short-base_short-1.25-1.1-0.0-0.0": [(0.00, 0.70, 0.66), (0.20, 0.86, 0.82)],
    "solo-female-ss_base_hoodie_short-base_short-1.25-1.1-0.0-0.0": [(0.00, 0.70, 0.66), (0.20, 0.86, 0.82)],
    "solo-female-ll_base_hoodie_short-base_short-1.15-1.1-0.0-0.0": [(0.00, 0.70, 0.66), (0.20, 0.86, 0.82)],

    # 3) Olive (richer yellow-green)
    "solo-female_base-short_skirt-1.1-1.0-0.0-2.5": [(0.60, 0.72, 0.06), (0.80, 0.88, 0.18)],
    "solo-female-ss_base-short_skirt-1.1-1.0-0.0-2.2": [(0.60, 0.72, 0.06), (0.80, 0.88, 0.18)],
    "solo-female-ll_base-short_skirt-1.1-1.0-0.0-2.8": [(0.60, 0.72, 0.06), (0.80, 0.88, 0.18)],

    # 4) Amber (deeper orange)
    "solo-female_base_shoulderless-skirt-1.1-1.0-0.0-1.5": [(0.98, 0.56, 0.05), (0.99, 0.76, 0.18)],
    "solo-female-ss_base_shoulderless-skirt-1.1-1.0-0.0-1.5": [(0.98, 0.56, 0.05), (0.99, 0.76, 0.18)],
    "solo-female-ll_base_shoulderless-skirt-1.1-1.0-0.0-1.5": [(0.98, 0.56, 0.05), (0.99, 0.76, 0.18)],

    # 5) Plum (more saturated violet)
    "solo-female_dress_sleeveless_decolte-base-1.1-1.0-0.0-0.0": [(0.62, 0.12, 0.78), (0.80, 0.36, 0.90)],
    "solo-female-ss_dress_sleeveless_decolte-base-1.15-1.0-0.0-0.0": [(0.62, 0.12, 0.78), (0.80, 0.36, 0.90)],
    "solo-female-ll_dress_sleeveless_decolte-base-1.1-1.0-0.0-0.0": [(0.62, 0.12, 0.78), (0.80, 0.36, 0.90)],

    # 6) Rust (stronger red-orange)
    "solo-female_dress_sleeveless_decolte-base-1.1-1.0-2.0-0.0": [(0.92, 0.26, 0.16), (0.96, 0.50, 0.38)],
    "solo-female-ss_dress_sleeveless_decolte-base-1.15-1.0-1.8-0.0": [(0.92, 0.26, 0.16), (0.96, 0.50, 0.38)],
    "solo-female-ll_dress_sleeveless_decolte-base-1.1-1.0-2.0-0.0": [(0.92, 0.26, 0.16), (0.96, 0.50, 0.38)],
}


body_color = (0.88, 0.88, 0.88)

OFFSET_DICT = {
    'solo-female-ss_base_hoodie_long-base-1.1-1.1-0.0-0.0': np.array([-2.2, 0.0, 7.0]),
    'solo-female-ss_base_hoodie_short-base_short-1.25-1.1-0.0-0.0': np.array([-3.5, 0.0, 7.0]),
    'solo-female-ss_base-short_skirt-1.1-1.0-0.0-2.2': np.array([-2.4, 0.0, 9.5]),
    'solo-female-ss_base_shoulderless-skirt-1.1-1.0-0.0-1.5': np.array([-3.5, 0.0, 9.5]),
    'solo-female-ss_dress_sleeveless_decolte-base-1.15-1.0-0.0-0.0': np.array([-5.0, 0.0, 7.0]),
    'solo-female-ss_dress_sleeveless_decolte-base-1.15-1.0-1.8-0.0': np.array([-2.5, 0.0, 12.0]),

    'solo-female_base_hoodie_long-base-1.1-1.1-0.0-0.0': np.array([1.0, 0.0, 7.0]),
    'solo-female_base_hoodie_short-base_short-1.25-1.1-0.0-0.0': np.array([0.0, 0.0, 7.0]),
    'solo-female_base-short_skirt-1.1-1.0-0.0-2.5': np.array([0.5, 0.0, 9.5]),
    'solo-female_base_shoulderless-skirt-1.1-1.0-0.0-1.5': np.array([-0.5, 0.0, 9.5]),
    'solo-female_dress_sleeveless_decolte-base-1.1-1.0-0.0-0.0': np.array([-1.0, 0.0, 7.0]),
    'solo-female_dress_sleeveless_decolte-base-1.1-1.0-2.0-0.0': np.array([0.0, 0.0, 12.0]),

    'solo-female-ll_base_hoodie_long-base-1.1-1.1-0.0-0.0': np.array([5.0, 0.0, 7.0]),
    'solo-female-ll_base_hoodie_short-base_short-1.15-1.1-0.0-0.0': np.array([3.5, 0.0, 7.0]),
    'solo-female-ll_base-short_skirt-1.1-1.0-0.0-2.8': np.array([3.5, 0.0, 9.5]),
    'solo-female-ll_base_shoulderless-skirt-1.1-1.0-0.0-1.5': np.array([2.4, 0.0, 9.5]),
    'solo-female-ll_dress_sleeveless_decolte-base-1.1-1.0-0.0-0.0': np.array([2.2, 0.0, 7.0]),
    'solo-female-ll_dress_sleeveless_decolte-base-1.1-1.0-2.0-0.0': np.array([2.5, 0.0, 12.0])
}

def render_scale_up_part():
    ps.init()
    ps.set_background_color((1, 1, 1))
    rootdir = 'results/sim/'
    for dir_idx, dirname in enumerate([x for x in os.listdir(rootdir) if 'solo-female' in x]):
        dirpath = os.path.join(rootdir, dirname)
        body_mesh_path = os.path.join(dirpath, 'target-00-with-offset.ply')
        upper_garment_path = os.path.join(dirpath, 'base_upper_uv.ply')
        lower_garment_path = os.path.join(dirpath, 'base_lower_uv.ply')

        body_mesh = trimesh.load(body_mesh_path)
        upper_mesh = trimesh.load(upper_garment_path)
        lower_mesh = trimesh.load(lower_garment_path) if os.path.exists(lower_garment_path) else None

        body_mesh.vertices += OFFSET_DICT[dirname]
        upper_mesh.vertices += OFFSET_DICT[dirname]
        if lower_mesh:
            lower_mesh.vertices += OFFSET_DICT[dirname]

        ps.register_surface_mesh(f"body-{dir_idx}", body_mesh.vertices, body_mesh.faces, smooth_shade=True, color=body_color)
        ps.register_surface_mesh(f"upper-{dir_idx}", upper_mesh.vertices, upper_mesh.faces, smooth_shade=True, color=palette[dirname][0])
        if lower_mesh:
            ps.register_surface_mesh(f"lower-{dir_idx}", lower_mesh.vertices, lower_mesh.faces, smooth_shade=True, color=palette[dirname][1])

    ps.reset_camera_to_home_view()
    params = ps.get_view_camera_parameters()
    center = ps.get_view_center()
    pos = np.array(params.get_position())
    new_pos = center + (pos - center) * 3.5 + (0.0, 2.0, 0.0)
    ps.look_at(tuple(new_pos), tuple(center))

    ps.screenshot('teaser_scale-up-new.png')


def vis_patches():
    body_mesh = trimesh.load('data/meshes/ref.ply')
    body_color = (0.85, 0.85, 0.85)

    # compute a scene scale to define epsilon offset
    scene_scale = np.linalg.norm(body_mesh.bounding_box.extents)
    eps = 2e-3 * scene_scale   # adjust if needed

    for garment_part in ['upper', 'lower']:
        ps.init()
        ps.remove_all_structures()
        ps.set_ground_plane_mode("none")     # no floor
        ps.set_background_color((1, 1, 1))   # white background

        # register body
        ps.register_surface_mesh(
            'body',
            body_mesh.vertices,
            body_mesh.faces,
            smooth_shade=True,
            color=body_color
        )

        patch_rootdir = f'data/patches/{garment_part}'
        for patch_dirname in os.listdir(patch_rootdir):
            patch_path = os.path.join(patch_rootdir, patch_dirname, 'ref.ply')
            patch_mesh = trimesh.load(patch_path)

            # offset patch vertices along normals
            patch_normals = patch_mesh.vertex_normals
            patch_vertices_offset = patch_mesh.vertices + eps * patch_normals

            ps.register_surface_mesh(
                f'{garment_part}-{patch_dirname}',
                patch_vertices_offset,
                patch_mesh.faces,
                smooth_shade=True
            )

        # rotate view a bit to the left
        ps.reset_camera_to_home_view()
        params = ps.get_view_camera_parameters()
        center = ps.get_view_center()
        pos = np.array(params.get_position())
        theta = np.radians(-30)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0,             1, 0],
            [-np.sin(theta),0, np.cos(theta)]
        ])
        new_pos = center + R @ (pos - center)
        ps.look_at(tuple(new_pos), tuple(center))

        # save screenshot
        ps.screenshot(f'{garment_part}_patches.svg')



if __name__ == '__main__':
    render_scale_up_part()
    #vis_patches()
