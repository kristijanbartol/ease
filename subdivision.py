from smplx import SMPL
import open3d as o3d


if __name__ == '__main__':
    smpl_model = SMPL(model_path='/data/hierprob3d/smpl/SMPL_FEMALE.pkl')
    verts = o3d.utility.Vector3dVector(smpl_model().vertices[0].cpu().detach().numpy())
    faces = o3d.utility.Vector3iVector(smpl_model.faces)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = verts
    mesh.triangles = faces
    mesh.compute_vertex_normals()

    # Mesh subdivision and smoothing
    for subdiv_iter in range(0, 5):
        for smooth_iter in range(0, 5):
            current_mesh = mesh.subdivide_loop(number_of_iterations=subdiv_iter)
            if smooth_iter > 0:
                current_mesh = current_mesh.filter_smooth_simple(number_of_iterations=smooth_iter)
            current_mesh.compute_vertex_normals()

            mesh_path = f'output/subdiv-{subdiv_iter}-smooth-{smooth_iter}.ply'
            print(f'Storing {mesh_path}...')
            o3d.io.write_triangle_mesh(
                mesh_path, 
                current_mesh
            )
