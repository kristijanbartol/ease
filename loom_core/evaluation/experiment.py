import os
import shutil
import trimesh
import numpy as np
import ezdxf
import svgwrite
from plyfile import PlyData
from collections import defaultdict

from loom_core.evaluation.param_mesh_uv import (
    add_uv_coordinates,
    ParamMeshUV
)

#from loom.eval.qualitative import qualitative_evaluation
from loom.eval.quantitative import quantitative_evaluation

from loom_core.evaluation.sim import simulate_garment_set


def load_uv_mesh_dict(experiment_name, optim_dress=False):
    # Copy latest patches to the current experiment folder (results/pattern/latest/ -> results/pattern/<experiment>/)
    pattern_2d_dir = 'results/pattern/'
    latest_dir = os.path.join(pattern_2d_dir, 'latest/')
    experiment_dir = os.path.join(pattern_2d_dir, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    for patch_label in os.listdir(latest_dir):
        shutil.copytree(os.path.join(latest_dir, patch_label), os.path.join(experiment_dir, patch_label), dirs_exist_ok=True)
    
    embedded_mesh_list_dict, param_2d_mesh_list_dict = defaultdict(list), defaultdict(list)
    param_mesh_dict = dict()
    garment_parts = ['upper'] if optim_dress else ['upper', 'lower']

    for garment_part in garment_parts:
        patches_rootdir = f'data/patches/{garment_part}'
        patch_names = os.listdir(patches_rootdir)
        for patch_name in patch_names:
            patch_path = f'{patches_rootdir}/{patch_name}/ref.ply'
            embedded_mesh = trimesh.load(patch_path)
            embedded_mesh_plydata = PlyData.read(patch_path)
            param_2d_mesh = trimesh.load(f'results/pattern/latest/{garment_part}/{patch_name}/optim_final-seams.ply')
            uv_coords = param_2d_mesh.vertices[:, :2]  

            add_uv_coordinates(embedded_mesh_plydata, uv_coords, patch_path)

            param_2d_mesh_list_dict[garment_part].append(param_2d_mesh)
            embedded_mesh_list_dict[garment_part].append(embedded_mesh)

        param_mesh_dict[garment_part] = ParamMeshUV(
            mesh_3d_list=embedded_mesh_list_dict[garment_part],
            mesh_2d_list=param_2d_mesh_list_dict[garment_part],
            garment_part=garment_part
        )
    return param_mesh_dict


class PatchProcessor:
    def __init__(self, patches):
        """
        Initialize the processor with garment patches and their corresponding offsets.
        
        Args:
            patches: List of GarmentPatch objects containing mesh and label
            offsets: Dictionary mapping patch labels to (x, y) offset tuples
        """
        self.patches = patches
        self.combined_mesh = None
        
    @staticmethod
    def _preprocess_patches(patches):
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
                        outlines = self._extract_mesh_outline(patch)
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
        all_vertices = np.vstack([patch.vertices for patch in self.patches])
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
                            outlines = self._extract_mesh_outline(patch)
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
                            for face in patch.faces:
                                points = [(patch.vertices[v][0]*scale, 
                                        patch.vertices[v][1]*scale) for v in face]
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


def qualitative_evaluation(name):
    for garment_part in ['upper', 'lower']:
        pattern_rootdir = f'results/pattern/latest/{garment_part}/'
        if os.path.exists(pattern_rootdir):
            patches = []
            for patch_name in os.listdir(pattern_rootdir):
                patch_dir = os.path.join(pattern_rootdir, patch_name)
                patch = trimesh.load(os.path.join(patch_dir, 'optim_final-seams.ply'))
                patches.append(patch)
        processor = PatchProcessor(patches)
        export_dir = f'results/qualitative/pattern/svg/{garment_part}/latest/'
        os.makedirs(export_dir, exist_ok=True)
        processor.export_svg_variants(f'{export_dir}/{name}')


#def evaluate_experiment(experiment_name, config, design_params, body_set):
def evaluate_experiment(project_dir, smpl_dir, experiment_name, design_params, body_set, is_dress, is_skirt):
    param_mesh_dict = load_uv_mesh_dict(experiment_name, is_dress)
    qualitative_evaluation(experiment_name)
    #quantitative_evaluation(experiment_name)
    simulate_garment_set(project_dir, smpl_dir, experiment_name, design_params, body_set, param_mesh_dict, is_dress, is_skirt)
