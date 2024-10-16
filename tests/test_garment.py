import unittest
import numpy as np
from collections import defaultdict

# Import your functions here
from src.garment import point_side, map_vertex_idx_to_face_idxs, get_adjacent_faces, update_face, create_dart

class TestMeshCuttingFunctions(unittest.TestCase):

    def setUp(self):
        # Set up some sample data for testing
        self.vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=np.float64)
        self.faces = np.array([
            [0, 1, 2], [1, 3, 2],  # Bottom face
            [4, 5, 6], [5, 7, 6],  # Top face
            [0, 4, 2], [2, 4, 6],  # Left face
            [1, 5, 3], [3, 5, 7],  # Right face
            [0, 1, 4], [1, 5, 4],  # Front face
            [2, 3, 6], [3, 7, 6]   # Back face
        ])

    def test_point_side(self):
        edge = (self.vertices[0], self.vertices[1])  # Edge along x-axis
        p_above = np.array([0.5, 0, 0.1])
        p_below = np.array([0.5, 0, -0.1])
        
        self.assertGreater(point_side(p_above, edge, -1), 0)
        self.assertLess(point_side(p_below, edge, -1), 0)
        self.assertLess(point_side(p_above, edge, 1), 0)
        self.assertGreater(point_side(p_below, edge, 1), 0)

    def test_map_vertex_idx_to_face_idxs(self):
        vertex_to_faces = map_vertex_idx_to_face_idxs(self.faces)
        self.assertEqual(set(vertex_to_faces[0]), {0, 4, 8})
        self.assertEqual(set(vertex_to_faces[1]), {0, 1, 6, 8, 9})

    def test_get_adjacent_faces(self):
        face1 = {0, 1, 2}
        face2 = {1, 2, 3}
        self.assertEqual(get_adjacent_faces(face1, face2), {1, 2})

    def test_update_face(self):
        face = [0, 1, 2]
        updated_face = update_face(face, 1, 10)
        self.assertEqual(updated_face, [0, 10, 2])

    def test_create_dart(self):
        selected_vidxs = [0, 1, 3]  # Cut along the bottom edge and up
        dart_orient = -1
        
        original_vertices = self.vertices.copy()
        original_faces = self.faces.copy()
        
        updated_vertices, updated_faces, dart_pairs = create_dart(
            self.vertices, self.faces, selected_vidxs, dart_orient
        )
        
        # Check if new vertices were added
        expected_new_vertices = len(selected_vidxs) - 1
        self.assertEqual(len(updated_vertices), len(self.vertices) + expected_new_vertices)
        
        # Check if the number of faces remains the same
        self.assertEqual(len(updated_faces), len(self.faces))
        
        # Check if dart_pairs are correct
        self.assertEqual(len(dart_pairs), len(selected_vidxs))
        self.assertEqual(dart_pairs[0], selected_vidxs[-1])  # Last selected vertex should be unchanged
        
        # Check the structure of dart_pairs
        for i, pair in enumerate(dart_pairs[1:], start=1):
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0], selected_vidxs[-i-1])  # Original vertex
            self.assertGreaterEqual(pair[1], len(self.vertices))  # New vertex index
        
        # Verify that new vertices have the same coordinates as their original counterparts
        new_vertex_start = len(self.vertices)
        for i, v_idx in enumerate(selected_vidxs[:-1]):
            new_v_idx = new_vertex_start + i
            self.assertTrue(np.allclose(self.vertices[v_idx], updated_vertices[new_v_idx], atol=0.00001))
        
        # Analyze face changes
        face_differences = np.setdiff1d(updated_faces, original_faces)
        if len(face_differences) == 0:
            print("Warning: No differences found in faces")
        else:
            print(f"Face differences:\n{face_differences}")
        
        # Check if any faces use the new vertices
        new_vertex_indices = set(range(len(original_vertices), len(updated_vertices)))
        faces_with_new_vertices = [face for face in updated_faces if any(v in new_vertex_indices for v in face)]
        
        if not faces_with_new_vertices:
            print("Warning: No faces use the new vertices")
        else:
            print(f"Faces using new vertices:\n{faces_with_new_vertices}")
        
        # Detailed comparison of original and updated faces
        for i, (orig_face, updated_face) in enumerate(zip(original_faces, updated_faces)):
            if not np.array_equal(orig_face, updated_face):
                print(f"Face {i} changed: {orig_face} -> {updated_face}")
        
        # Assert that at least some change has occurred
        self.assertFalse(np.array_equal(original_faces, updated_faces), 
                        "No changes were made to the faces")
                

if __name__ == '__main__':
    unittest.main()