##### find midpoint module #####

import numpy as np

def insert_front_midline_point(verts, faces, v_idx):
    y = verts[v_idx, 1]
    # Build unique edges
    edges = np.unique(
        np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0
    )

    best_p, best_i0, best_i1 = None, None, None
    best_score = np.inf
    for i0, i1 in edges:
        v0, v1 = verts[i0], verts[i1]
        if v0[2] <= 0 or v1[2] <= 0:
            continue
        t = (y - v0[1]) / (v1[1] - v0[1])
        if 0 <= t <= 1:
            # Pick the edge whose vertices are both closest to the midline X=0
            score = max(abs(v0[0]), abs(v1[0]))
            if score < best_score:
                best_score, best_p, best_i0, best_i1 = score, v0 + t * (v1 - v0), i0, i1

    verts_old = verts
    verts = np.vstack((verts, best_p[None]))
    new_idx = len(verts) - 1

    def orient(tri, n_ref):
        # Ensure new triangles keep the original face orientation
        a, b, c = tri
        if np.dot(np.cross(verts[b] - verts[a], verts[c] - verts[a]), n_ref) < 0:
            return [a, c, b]
        return tri

    new_faces = []
    for f in faces:
        if best_i0 in f and best_i1 in f:
            # Split the face along the new vertex
            n_ref = np.cross(verts_old[f[1]] - verts_old[f[0]], verts_old[f[2]] - verts_old[f[0]])
            third = next(v for v in f if v not in (best_i0, best_i1))
            new_faces.append(orient([best_i0, new_idx, third], n_ref))
            new_faces.append(orient([new_idx, best_i1, third], n_ref))
        else:
            new_faces.append(f)

    return verts, np.asarray(new_faces, int), new_idx


def insert_midline_point(verts, faces, v_idx, front=True):
    y = verts[v_idx, 1]
    sign = 1.0 if front else -1.0
    # unique undirected edges
    edges = np.unique(
        np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0
    )

    best = (np.inf, None, None, None)  # (score, point, i0, i1)
    for i0, i1 in edges:
        v0, v1 = verts[i0], verts[i1]
        if v0[2] * sign <= 0 or v1[2] * sign <= 0:
            continue
        t = (y - v0[1]) / (v1[1] - v0[1])
        if 0 <= t <= 1:
            # minimize how far BOTH edge endpoints are from X=0
            score = max(abs(v0[0]), abs(v1[0]))
            if score < best[0]:
                best = (score, v0 + t * (v1 - v0), i0, i1)

    _, p, i0, i1 = best
    verts_old = verts
    verts = np.vstack((verts, p[None]))
    new_idx = len(verts) - 1

    def orient(tri, n_ref):  # keep original face orientation
        a, b, c = tri
        if np.dot(np.cross(verts[b] - verts[a], verts[c] - verts[a]), n_ref) < 0:
            return [a, c, b]
        return tri

    new_faces = []
    for f in faces:
        if i0 in f and i1 in f:
            n_ref = np.cross(verts_old[f[1]] - verts_old[f[0]], verts_old[f[2]] - verts_old[f[0]])
            third = next(v for v in f if v not in (i0, i1))
            new_faces.append(orient([i0, new_idx, third], n_ref))
            new_faces.append(orient([new_idx, i1, third], n_ref))
        else:
            new_faces.append(f)

    return verts, np.asarray(new_faces, int), new_idx



def horizontal_plane_cut(vertices, faces, y_cut):
    plane_normal = np.array([0, -1, 0])
    plane_point = np.array([0, y_cut, 0])
    plane_d = -np.dot(plane_normal, plane_point)

    def point_plane_distance(v):
        return np.dot(plane_normal, v) + plane_d

    def is_intersecting(v1, v2):
        return point_plane_distance(v1) * point_plane_distance(v2) < 0

    def intersection_point(v1, v2):
        d1 = point_plane_distance(v1)
        d2 = point_plane_distance(v2)
        t = d1 / (d1 - d2)
        return v1 + t * (v2 - v1)

    def compute_barycentric_coordinates(p, tri):
        v0, v1, v2 = tri
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return np.array([u, v, w])

    def move_to_nearest_edge(vertex, polygon_edges, tri):
        # Find closest point on the polygon edges
        min_dist = np.inf
        nearest_point = vertex
        for p1, p2 in polygon_edges:
            e = p2 - p1
            t = np.dot(vertex - p1, e) / np.dot(e, e)
            t = np.clip(t, 0, 1)
            proj = p1 + t * e
            dist = np.linalg.norm(vertex - proj)
            if dist < min_dist:
                min_dist = dist
                nearest_point = proj
        bary = compute_barycentric_coordinates(nearest_point, tri)
        return nearest_point, bary

    intersected_triangles = []
    intersection_points = []

    # Find intersection points for each face
    for face in faces:
        v1, v2, v3 = vertices[face]
        intersections = []
        if is_intersecting(v1, v2):
            intersections.append(intersection_point(v1, v2))
        if is_intersecting(v2, v3):
            intersections.append(intersection_point(v2, v3))
        if is_intersecting(v3, v1):
            intersections.append(intersection_point(v3, v1))

        if len(intersections) == 2:
            intersection_points.extend(intersections)
            intersected_triangles.append(face)

    if not intersection_points:
        return vertices, faces, {}

    intersection_points = np.array(intersection_points)
    # Pair intersections into edges (assumes consistent triangle processing)
    polygon_edges = [(intersection_points[i], intersection_points[i + 1]) for i in range(0, len(intersection_points), 2)]

    new_vertices = vertices.copy()
    barycentric_coords = {}

    # Move vertices above the plane to nearest cut edge
    for face in intersected_triangles:
        tri = vertices[face]
        for i, v_idx in enumerate(face):
            v = vertices[v_idx]
            if point_plane_distance(v) > 0:  # above plane
                new_v, bary = move_to_nearest_edge(v, polygon_edges, tri)
                new_vertices[v_idx] = new_v
                barycentric_coords[v_idx] = bary

    return new_vertices, faces, barycentric_coords

