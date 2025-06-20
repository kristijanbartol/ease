from typing import List, Dict, Tuple
import trimesh
import ezdxf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A1
from reportlab.lib.units import mm
import svgwrite
import numpy as np
import cv2

from loom.eval.stretch_utils import extract_stretch_ratios
from loom.eval.const import (
    GLOBAL_IMG_SCALE,
    IMG_OFFSETS_DICT
)


def mesh_to_image(mesh, image_size=(800, 800)):
    # Project the 3D mesh to 2D
    points = np.asarray(mesh.vertices)
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    
    points[:, :2] = (points[:, :2] - min_bounds[:2]) * GLOBAL_IMG_SCALE

    # Create an empty image with alpha channel (transparent background)
    image = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)

    triangles = np.asarray(mesh.faces)
    edge_color = (42, 42, 165, 255)
    for tri in triangles:
        pts = points[tri][:, :2].astype(int)
        cv2.fillConvexPoly(image, pts, (0, 0, 0, 255))  # Black color for the mesh
        cv2.polylines(image, [pts], isClosed=True, color=edge_color, thickness=1)  # Draw edges with brown color
    
    return image[::-1]


def draw_mesh_on_canvas(canvas, mesh_image, offset, color):
    center = int(canvas.shape[0] / 2), int(canvas.shape[1] / 2)
    half_sizes = int(mesh_image.shape[0] / 2), int(mesh_image.shape[1] / 2)
    x_offset, y_offset = offset
    y1, y2 = center[1] + y_offset - half_sizes[1], center[1] + y_offset + half_sizes[1]
    x1, x2 = center[0] + x_offset - half_sizes[0], center[0] + x_offset + half_sizes[0]

    # Extract color channels if the image has an alpha channel
    if mesh_image.shape[2] == 4:
        img_rgb = mesh_image[..., :3]
        img_alpha = mesh_image[..., 3]
    else:
        img_rgb = mesh_image
        img_alpha = None

    # Apply the color to the mesh while preserving transparency
    colored_img = cv2.addWeighted(img_rgb, 0.5, np.full_like(img_rgb, color), 0.5, 0)
    
    if img_alpha is not None:
        mask = img_alpha / 255.0
        for c in range(0, 3):
            canvas[y1:y2, x1:x2, c] = canvas[y1:y2, x1:x2, c] * (1 - mask) + colored_img[..., c] * mask
        # Update the alpha channel of the canvas
        canvas[y1:y2, x1:x2, 3] = np.maximum(canvas[y1:y2, x1:x2, 3], mesh_image[..., 3])
    else:
        canvas[y1:y2, x1:x2, :3] = colored_img


class QualitativeMesh:
    def __init__(self, patch_label, param_2d_fname):
        self.mesh = trimesh.load(f'results/pattern/latest/{patch_label}/{param_2d_fname}')
        self.offset = IMG_OFFSETS_DICT[patch_label]
        self.patch_label = patch_label
        self.is_front = True if patch_label.split('_')[1] == 'front' else False
        #self.face_scales_weft, self.face_scales_warp = self._extract_stretch_colors(patch_label)
        
    def _extract_stretch_colors(self, patch_label):
        embedded_mesh = trimesh.load(f'data/embedded/{patch_label}/ref.ply')
        bary_coords_u = np.loadtxt(f'data/bary/ref_2d/{patch_label}/bary_2d_u_init.txt')
        bary_coords_v = np.loadtxt(f'data/bary/ref_2d/{patch_label}/bary_2d_v_init.txt')
        
        scales_u, scales_v = extract_stretch_ratios(
            V_2d=self.mesh.vertices,
            V_3d=embedded_mesh.vertices,
            F=embedded_mesh.faces,
            DU_bary=bary_coords_u,
            DV_bary=bary_coords_v
        )
        # NOTE: When evaluating, swap the meaning of warp/weft for sleeves
        if 'sleeve' in patch_label:
            return scales_v, scales_u
        else:
            return scales_u, scales_v


class PatchProcessor:
    def __init__(self, patches: List[QualitativeMesh], offsets: Dict[str, Tuple[float, float]]):
        """
        Initialize the processor with garment patches and their corresponding offsets.
        
        Args:
            patches: List of GarmentPatch objects containing mesh and label
            offsets: Dictionary mapping patch labels to (x, y) offset tuples
        """
        self.patches = self._preprocess_patches(patches)
        self.offsets = offsets
        self.combined_mesh = None
        
    @staticmethod
    def _preprocess_patches(patches: List[QualitativeMesh]) -> List[QualitativeMesh]:
        for patch in patches:
            patch.mesh.vertices = patch.mesh.vertices[:, :2]
        return patches
        
    def apply_offsets(self):
        """Apply the stored offsets to each mesh."""
        for patch in self.patches:
            offset = self.offsets[patch.patch_label]
            patch.mesh.vertices[:, 0] += offset[0]
            patch.mesh.vertices[:, 1] += offset[1]
            
    def combine_meshes(self):
        """
        Combine all meshes into a single structure while keeping them disconnected.
        Updates vertex and face indices accordingly.
        """
        vertices = []
        faces = []
        vertex_offset = 0
        
        for patch in self.patches:
            # Add vertices
            vertices.extend(patch.mesh.vertices)
            # Update face indices and add faces
            updated_faces = patch.mesh.faces + vertex_offset
            faces.extend(updated_faces)
            # Update vertex offset for next patch
            vertex_offset += len(patch.mesh.vertices)
        
        # Create combined mesh
        self.combined_mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces)
        )
        self.combined_mesh_3d = trimesh.Trimesh(
            vertices=np.hstack((np.array(vertices), np.zeros((np.array(vertices).shape[0], 1)))),
            faces=np.array(faces)
        )   # for exporting to PLY
        
    def export_ply(self, filepath: str):
        """Export the combined mesh as PLY file."""
        if self.combined_mesh is None:
            raise ValueError("Must call combine_meshes() before exporting")
        self.combined_mesh_3d.export(filepath)
        
    @staticmethod
    def _extract_mesh_outline(mesh):
        """
        Extract the outline of a mesh by finding edges that appear only once 
        (boundary edges).
        
        Args:
            mesh: Trimesh object
            
        Returns:
            List of point pairs representing outline edges
        """
        # Create edge to face mapping
        edge_to_face = {}
        for face_idx, face in enumerate(mesh.faces):
            # For each face, create its edges
            for i in range(len(face)):
                # Get vertices of the edge
                v1 = face[i]
                v2 = face[(i + 1) % len(face)]
                # Sort vertices to ensure consistent edge representation
                edge = tuple(sorted([v1, v2]))
                # Add to mapping
                if edge in edge_to_face:
                    edge_to_face[edge].append(face_idx)
                else:
                    edge_to_face[edge] = [face_idx]
        
        # Find boundary edges (edges that appear only once)
        boundary_edges = [edge for edge, faces in edge_to_face.items() if len(faces) == 1]
        
        # Convert edges to actual coordinates
        outline_segments = [(mesh.vertices[edge[0]], mesh.vertices[edge[1]]) 
                        for edge in boundary_edges]
        
        # Sort segments to form continuous chains
        chains = []
        current_chain = []
        used_segments = set()
        
        while outline_segments and len(used_segments) < len(outline_segments):
            if not current_chain:
                # Start new chain with first unused segment
                for i, segment in enumerate(outline_segments):
                    if i not in used_segments:
                        current_chain = list(segment)
                        used_segments.add(i)
                        break
            
            found_next = False
            for i, segment in enumerate(outline_segments):
                if i in used_segments:
                    continue
                    
                if abs(segment[0][0] - current_chain[-1][0]) < 1e-6 and \
                abs(segment[0][1] - current_chain[-1][1]) < 1e-6:
                    current_chain.append(segment[1])
                    used_segments.add(i)
                    found_next = True
                    break
                    
                if abs(segment[1][0] - current_chain[-1][0]) < 1e-6 and \
                abs(segment[1][1] - current_chain[-1][1]) < 1e-6:
                    current_chain.append(segment[0])
                    used_segments.add(i)
                    found_next = True
                    break
            
            if not found_next:
                # Chain is complete
                if len(current_chain) > 2:  # Only keep chains with more than 2 points
                    chains.append(current_chain)
                current_chain = []
        
        # Add the last chain if it exists
        if current_chain and len(current_chain) > 2:
            chains.append(current_chain)
        
        return chains

    def export_dxf_variants(self, base_filepath: str):
        """
        Export patches to DXF format with different versions and scaling factors.
        Creates both full mesh and outline-only versions.
        
        Args:
            base_filepath: Base filepath without extension (e.g., 'path/to/mesh')
        """
        # DXF versions to try
        #versions = ['R12', 'R2010', 'R2013']
        versions = ['R2010']
        # Scaling factors (1 means original, 10 means 10x larger, etc.)
        #scales = [1, 10, 100]
        scales = [100]
        
        for version in versions:
            for scale in scales:
                # Create filenames for both full mesh and outline versions
                #full_filepath = f"{base_filepath}_full_v{version}_s{scale}x.dxf"
                outline_filepath = f"{base_filepath}_outline_v{version}_s{scale}x.dxf"
                
                try:
                    # Export full mesh
                    doc_full = ezdxf.new(version)
                    msp_full = doc_full.modelspace()
                    
                    # Export outline
                    doc_outline = ezdxf.new(version)
                    msp_outline = doc_outline.modelspace()
                    
                    # Set units to millimeters for both
                    if version != 'R12':  # R12 doesn't support units
                        for doc in [doc_full, doc_outline]:
                            doc.header['$INSUNITS'] = 4  # 4 = millimeters
                            doc.header['$LUNITS'] = 2    # 2 = decimal
                    
                    # Process each patch
                    for patch in self.patches:
                        '''
                        # Full mesh export
                        for face in patch.mesh.faces:
                            points = patch.mesh.vertices[face]
                            points_2d = [(p[0] * scale, p[1] * scale) for p in points]
                            points_2d.append(points_2d[0])
                            if version == 'R12':
                                # For R12, use old-style POLYLINE
                                polyline = msp_full.add_polyline2d(points_2d)
                                polyline.close(True)
                            else:
                                msp_full.add_lwpolyline(points_2d)
                        '''
                        
                        # Outline export
                        outlines = self._extract_mesh_outline(patch.mesh)
                        for outline in outlines:
                            points_2d = [(p[0] * scale, p[1] * scale) for p in outline]
                            points_2d.append(points_2d[0])  # Close the loop
                            if version == 'R12':
                                # For R12, use old-style POLYLINE
                                polyline = msp_outline.add_polyline2d(points_2d)
                                polyline.close(True)
                            else:
                                msp_outline.add_lwpolyline(points_2d)
                    
                    # Save both versions
                    #doc_full.saveas(full_filepath)
                    doc_outline.saveas(outline_filepath)
                    #print(f"Successfully exported full mesh: {full_filepath}")
                    print(f"Successfully exported outline: {outline_filepath}")
                    
                except Exception as e:
                    print(f"Failed to export version {version} scale {scale}x: {str(e)}")
                    
    def export_pdf_variants(self, base_filepath: str):
        """
        Export patches to PDF format with different scaling factors on A1 paper.
        Creates both full mesh and outline-only versions.
        
        Args:
            base_filepath: Base filepath without extension
        """
        # A1 size in mm
        a1_width_mm = 594
        a1_height_mm = 841
        #scales = [1, 10, 100]
        scales = [100]
        
        # Get mesh bounds for all patches
        all_vertices = np.vstack([patch.mesh.vertices for patch in self.patches])
        min_x, min_y = all_vertices.min(axis=0)[:2]
        max_x, max_y = all_vertices.max(axis=0)[:2]
        mesh_width = max_x - min_x
        mesh_height = max_y - min_y
        
        for scale in scales:
            # Create filenames for both versions
            #full_filepath = f"{base_filepath}_full_s{scale}x.pdf"
            outline_filepath = f"{base_filepath}_outline_s{scale}x.pdf"
            
            try:
                # Create PDFs for both versions
                for is_outline in [True]:
                    #filepath = outline_filepath if is_outline else full_filepath
                    filepath = outline_filepath
                    c = canvas.Canvas(filepath, pagesize=A1)
                    
                    # Calculate scaling to fit on page with margins
                    margin_mm = 20  # 20mm margin
                    available_width = a1_width_mm - 2 * margin_mm
                    available_height = a1_height_mm - 2 * margin_mm
                    
                    # Calculate scale factor to fit mesh on page
                    scale_factor = min(
                        available_width / (mesh_width * scale),
                        available_height / (mesh_height * scale)
                    )
                    
                    # Transform coordinates to PDF space
                    def transform_point(p):
                        x = (p[0] - min_x) * scale * scale_factor + margin_mm
                        y = (p[1] - min_y) * scale * scale_factor + margin_mm
                        return x * mm, y * mm
                    
                    # Draw mesh
                    c.setLineWidth(0.2 * mm)  # Set line width to 0.2mm
                    
                    for patch in self.patches:
                        if is_outline:
                            # Draw only outline
                            outlines = self._extract_mesh_outline(patch.mesh)
                            for outline in outlines:
                                points = [transform_point(p) for p in outline]
                                c.path = c.beginPath()
                                c.path.moveTo(*points[0])
                                for point in points[1:]:
                                    c.path.lineTo(*point)
                                c.path.close()
                                c.drawPath(c.path)
                        else:
                            # Draw full mesh
                            for face in patch.mesh.faces:
                                points = [transform_point(patch.mesh.vertices[v]) for v in face]
                                c.path = c.beginPath()
                                c.path.moveTo(*points[0])
                                for point in points[1:]:
                                    c.path.lineTo(*point)
                                c.path.close()
                                c.drawPath(c.path)
                    
                    # Add scale information
                    c.setFont("Helvetica", 10)
                    c.drawString(margin_mm * mm, (a1_height_mm - margin_mm/2) * mm, 
                            f"Scale: {scale}x | {'Outline Only' if is_outline else 'Full Mesh'}")
                    
                    c.save()
                
                print(f"Successfully exported PDFs for scale {scale}x")
                
            except Exception as e:
                print(f"Failed to export PDF scale {scale}x: {str(e)}")

    def export_svg_variants(self, base_filepath: str):
        """
        Export patches to SVG format with different scaling factors.
        Creates both full mesh and outline-only versions.
        
        Args:
            base_filepath: Base filepath without extension
        """
        #scales = [1, 10, 100]
        scales = [100]
        
        # Get mesh bounds for all patches
        all_vertices = np.vstack([patch.mesh.vertices for patch in self.patches])
        min_x, min_y = all_vertices.min(axis=0)[:2]
        max_x, max_y = all_vertices.max(axis=0)[:2]
        mesh_width = max_x - min_x
        mesh_height = max_y - min_y
        
        for scale in scales:
            # Create filenames for both versions
            #full_filepath = f"{base_filepath}_full_s{scale}x.svg"
            outline_filepath = f"{base_filepath}_outline_s{scale}x.svg"
            
            try:
                # Create SVGs for both versions
                for is_outline in [True]:
                    #filepath = outline_filepath if is_outline else full_filepath
                    filepath = outline_filepath
                    
                    # Create SVG document with appropriate viewBox
                    margin = mesh_width * 0.05  # 5% margin
                    dwg = svgwrite.Drawing(
                        filepath,
                        viewBox=f"{(min_x-margin)*scale} {(min_y-margin)*scale} "
                            f"{(mesh_width+2*margin)*scale} {(mesh_height+2*margin)*scale}"
                    )
                    
                    # Add title
                    dwg.set_desc(title=f"Scale: {scale}x | {'Outline Only' if is_outline else 'Full Mesh'}")
                    
                    for patch in self.patches:
                        if is_outline:
                            # Draw only outline
                            outlines = self._extract_mesh_outline(patch.mesh)
                            for outline in outlines:
                                points = [(p[0]*scale, p[1]*scale) for p in outline]
                                points.append(points[0])  # Close the path
                                path = dwg.path(d=f"M {points[0][0]},{points[0][1]}")
                                for point in points[1:]:
                                    path.push(f"L {point[0]},{point[1]}")
                                path.push("Z")
                                path.stroke(color="black", width=0.5).fill(opacity=0)
                                dwg.add(path)
                        else:
                            # Draw full mesh
                            for face in patch.mesh.faces:
                                points = [(patch.mesh.vertices[v][0]*scale, 
                                        patch.mesh.vertices[v][1]*scale) for v in face]
                                points.append(points[0])  # Close the path
                                path = dwg.path(d=f"M {points[0][0]},{points[0][1]}")
                                for point in points[1:]:
                                    path.push(f"L {point[0]},{point[1]}")
                                path.push("Z")
                                path.stroke(color="black", width=0.5).fill(opacity=0)
                                dwg.add(path)
                    
                    dwg.save()
                
                print(f"Successfully exported SVGs for scale {scale}x")
                
            except Exception as e:
                print(f"Failed to export SVG scale {scale}x: {str(e)}")
