import numpy as np
import trimesh
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

@dataclass
class Point:
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + 
                      (self.y - other.y)**2 + 
                      (self.z - other.z)**2)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

@dataclass
class Edge:
    v1_idx: int
    v2_idx: int
    
    def __hash__(self):
        # Order-independent hash
        return hash(tuple(sorted([self.v1_idx, self.v2_idx])))
    
    def __eq__(self, other):
        return (self.v1_idx == other.v1_idx and self.v2_idx == other.v2_idx) or \
               (self.v1_idx == other.v2_idx and self.v2_idx == other.v1_idx)
               
@dataclass
class Triangle:
    p1: Point
    p2: Point
    p3: Point
    
    def calculate_area(self) -> float:
        # Calculate area using cross product
        v1 = np.array([self.p2.x - self.p1.x, self.p2.y - self.p1.y, self.p2.z - self.p1.z])
        v2 = np.array([self.p3.x - self.p1.x, self.p3.y - self.p1.y, self.p3.z - self.p1.z])
        cross = np.cross(v1, v2)
        return 0.5 * np.sqrt(np.sum(cross**2))
    
    def contains_point(self, point: Point) -> float:
        """
        Returns the barycentric coordinates if point is in triangle.
        The returned value can be used to determine partial containment.
        """
        def compute_barycentric(p: Point) -> Tuple[float, float, float]:
            v0 = np.array([self.p2.x - self.p1.x, self.p2.y - self.p1.y, self.p2.z - self.p1.z])
            v1 = np.array([self.p3.x - self.p1.x, self.p3.y - self.p1.y, self.p3.z - self.p1.z])
            v2 = np.array([p.x - self.p1.x, p.y - self.p1.y, p.z - self.p1.z])
            
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            
            return u, v, w
            
        u, v, w = compute_barycentric(point)
        if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
            return u + v + w  # Should be approximately 1 if point is in triangle
        return 0.0

class DartRegion:
    def __init__(self, inner_triangle: Triangle, outer_triangle: Triangle):
        self.inner_triangle = inner_triangle
        self.outer_triangle = outer_triangle
        self.inner_area = inner_triangle.calculate_area()
        self.outer_area = outer_triangle.calculate_area()
        
    def get_containment_factor(self, face_triangle: Triangle) -> Tuple[float, float]:
        """
        Returns (inner_containment, outer_containment) factors for a face.
        Values between 0 and 1 indicate partial containment.
        """
        # Sample points from the face triangle to determine containment
        sample_points = self._generate_sample_points(face_triangle)
        
        inner_count = 0
        outer_count = 0
        total_points = len(sample_points)
        
        for point in sample_points:
            inner_containment = self.inner_triangle.contains_point(point)
            if inner_containment > 0:
                inner_count += 1
            
            outer_containment = self.outer_triangle.contains_point(point)
            if outer_containment > 0:
                outer_count += 1
        
        return (inner_count / total_points, outer_count / total_points)
    
    def _generate_sample_points(self, triangle: Triangle, num_samples: int = 10) -> List[Point]:
        """Generate sample points within the triangle for containment testing"""
        points = []
        for i in range(num_samples):
            for j in range(num_samples - i):
                # Barycentric coordinates
                a = i / num_samples
                b = j / num_samples
                c = 1 - a - b
                
                x = a * triangle.p1.x + b * triangle.p2.x + c * triangle.p3.x
                y = a * triangle.p1.y + b * triangle.p2.y + c * triangle.p3.y
                z = a * triangle.p1.z + b * triangle.p2.z + c * triangle.p3.z
                
                points.append(Point(x, y, z))
        return points

class DartMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, stretch_values: np.ndarray):
        self.vertices = vertices
        self.faces = faces
        self.stretch_values = stretch_values
        self.edges = self._build_edge_structure()
        self.vertex_faces = self._build_vertex_face_map()
        
    def _build_edge_structure(self) -> Dict[Edge, Set[int]]:
        """Build edge to face mapping"""
        edge_to_faces = {}
        for face_idx, face in enumerate(self.faces):
            edges = [
                Edge(face[0], face[1]),
                Edge(face[1], face[2]),
                Edge(face[2], face[0])
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = set()
                edge_to_faces[edge].add(face_idx)
        return edge_to_faces
    
    def _build_vertex_face_map(self) -> Dict[int, Set[int]]:
        """Build vertex to face mapping"""
        vertex_faces = {i: set() for i in range(len(self.vertices))}
        for face_idx, face in enumerate(self.faces):
            for vertex_idx in face:
                vertex_faces[vertex_idx].add(face_idx)
        return vertex_faces

    def _calculate_edge_direction(self, point: Point, edge_vertex_idx: int) -> Optional[np.ndarray]:
        """Calculate edge direction at the given point"""
        # Find the edge that contains this vertex
        connected_edges = [edge for edge in self.edges.keys() 
                         if edge_vertex_idx in (edge.v1_idx, edge.v2_idx)]
        
        if not connected_edges:
            return None
        
        # Get the direction of each edge
        edge_directions = []
        for edge in connected_edges:
            v1 = self.vertices[edge.v1_idx]
            v2 = self.vertices[edge.v2_idx]
            direction = v2 - v1
            direction = direction / np.linalg.norm(direction)
            edge_directions.append(direction)
        
        # Average the directions (for vertices with multiple connected edges)
        avg_direction = np.mean(edge_directions, axis=0)
        return avg_direction / np.linalg.norm(avg_direction)
    
    def _calculate_surface_normal(self, point: Point, edge_vertex_idx: int) -> np.ndarray:
        """Calculate surface normal at the given point using adjacent faces"""
        # Get all faces connected to this vertex
        adjacent_faces = self.vertex_faces[edge_vertex_idx]
        
        # Calculate weighted normal
        normal = np.zeros(3)
        for face_idx in adjacent_faces:
            face = self.faces[face_idx]
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            
            # Calculate face normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            face_normal = np.cross(edge1, edge2)
            
            # Weight by face area
            area = np.linalg.norm(face_normal) / 2
            normal += face_normal * area
        
        return normal / np.linalg.norm(normal)
    
    def update_stretch_values(self, dart_region: DartRegion):
        """Update stretch values based on dart region containment"""
        # Calculate the average stretch value for the outer region
        total_outer_value = 0
        total_outer_weight = 0
        
        for face_idx, face in enumerate(self.faces):
            face_triangle = Triangle(
                Point(*self.vertices[face[0]]),
                Point(*self.vertices[face[1]]),
                Point(*self.vertices[face[2]])
            )
            
            inner_factor, outer_factor = dart_region.get_containment_factor(face_triangle)
            
            if outer_factor > 0:
                # Face is at least partially in outer triangle
                original_value = self.stretch_values[face_idx]
                if inner_factor > 0:
                    # Face is partially in inner triangle
                    weight = outer_factor * (1 - inner_factor)
                else:
                    weight = outer_factor
                    
                total_outer_value += original_value * weight
                total_outer_weight += weight
        
        if total_outer_weight > 0:
            average_outer_value = total_outer_value / total_outer_weight
            
            # Apply the updates
            for face_idx, face in enumerate(self.faces):
                face_triangle = Triangle(
                    Point(*self.vertices[face[0]]),
                    Point(*self.vertices[face[1]]),
                    Point(*self.vertices[face[2]])
                )
                
                inner_factor, outer_factor = dart_region.get_containment_factor(face_triangle)
                
                if outer_factor > 0:
                    if inner_factor > 0:
                        # Face is partially in both regions
                        self.stretch_values[face_idx] = (
                            average_outer_value * (1 - inner_factor) +
                            0 * inner_factor
                        )
                    else:
                        # Face is only in outer region
                        self.stretch_values[face_idx] = (
                            average_outer_value * outer_factor +
                            self.stretch_values[face_idx] * (1 - outer_factor)
                        )

    def create_dart(self, edge_vertex_idx: int, inner_point: Point, 
                   inner_distance: float, outer_distance: float) -> Optional[DartRegion]:
        """Create a dart starting from edge vertex towards inner point"""
        edge_point = Point(*self.vertices[edge_vertex_idx])
        
        # Calculate edge direction
        edge_direction = self._calculate_edge_direction(edge_point, edge_vertex_idx)
        if edge_direction is None:
            return None
        
        # Calculate surface normal
        surface_normal = self._calculate_surface_normal(edge_point, edge_vertex_idx)
        
        # Calculate perpendicular direction in the surface plane
        perpendicular = np.cross(edge_direction, surface_normal)
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        # Generate points for inner and outer triangles
        inner_points = self._generate_edge_points(edge_point, perpendicular, inner_distance)
        outer_points = self._generate_edge_points(edge_point, perpendicular, outer_distance)
        
        # Create triangles
        inner_triangle = Triangle(inner_points[0], inner_points[1], inner_point)
        outer_triangle = Triangle(outer_points[0], outer_points[1], inner_point)
        
        return DartRegion(inner_triangle, outer_triangle)
    
    def _generate_edge_points(self, start_point: Point, 
                            perpendicular: np.ndarray, 
                            distance: float) -> Tuple[Point, Point]:
        """Generate two points at specified distance perpendicular to the edge"""
        p1 = Point(
            start_point.x + distance * perpendicular[0],
            start_point.y + distance * perpendicular[1],
            start_point.z + distance * perpendicular[2]
        )
        
        p2 = Point(
            start_point.x - distance * perpendicular[0],
            start_point.y - distance * perpendicular[1],
            start_point.z - distance * perpendicular[2]
        )
        
        return (p1, p2)


if __name__ == '__main__':
    upper_front_mesh = trimesh.load_mesh('data/embedded/latest/upper_front/init.ply')
    mesh = DartMesh(upper_front_mesh.vertices, upper_front_mesh.faces, stretch_values)
    inner_point = Point(x, y, z)  # Point on the surface where the dart should point
    dart_region = mesh.create_dart(
        edge_vertex_idx=vertex_id,  # Index of the vertex on the edge
        inner_point=inner_point,
        inner_distance=2.0,  # 2cm
        outer_distance=5.0   # 5cm
    )

    if dart_region:
        mesh.update_stretch_values(dart_region)
