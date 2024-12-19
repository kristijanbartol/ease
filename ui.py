import sys
import os
import argparse
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QPushButton, QFrame)
from PyQt6.QtCore import Qt
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from smplx import SMPL
import json

from tailorlang.mesh_processing import (
    MeshState
)
from tailorlang.garment import DesignParameters


class Mesh3DView(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh_items = []
        self.setup_view()
        
        # Camera control states
        self.last_pos = None
        self.moving = False
        self.panning = False
        
    def setup_view(self):
        # Add grid for reference
        grid = gl.GLGridItem()
        self.addItem(grid)
        
        # Configure lighting
        self.opts['lighting'] = True
        self.opts['ambient'] = 0.5
        self.opts['diffuse'] = 0.8
        
        # Setup camera
        self.setCameraPosition(distance=2.5, elevation=30, azimuth=45)
        self.opts['fov'] = 60
        
    def mousePressEvent(self, event):
        """Handle mouse press events for rotation and panning"""
        self.last_pos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.panning = True
            else:
                self.moving = True
                
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        self.moving = False
        self.panning = False
        self.last_pos = None
        
    def mouseMoveEvent(self, event):
        """Handle mouse movement for rotation and panning"""
        if self.last_pos is None:
            self.last_pos = event.pos()
            return
            
        diff = event.pos() - self.last_pos
        self.last_pos = event.pos()
        
        if self.moving:  # Rotation
            dx = diff.x()
            dy = diff.y()
            
            # Update camera angles
            self.opts['elevation'] = min(max(self.opts['elevation'] - dy/2, -90), 90)
            self.opts['azimuth'] = (self.opts['azimuth'] + dx/2) % 360
            
            self.update()
            
        elif self.panning:  # Panning
            dx = diff.x() / 100.0
            dy = diff.y() / 100.0
            
            # Get camera vectors
            cameraPosition = np.array(self.cameraPosition())
            center = np.array(self.opts['center'])
            up = np.array(self.opts['up'])
            
            # Calculate right vector
            forward = center - cameraPosition
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            # Update camera position and center
            translation = right * dx - up * dy
            self.opts['center'] = center + translation
            self.update()
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()
        
        # Adjust zoom speed
        factor = 0.001
        
        if self.opts['distance'] * (1 - delta * factor) > 0.1:  # Prevent zoom too close
            self.opts['distance'] *= (1 - delta * factor)
            self.update()
        
    def update_meshes(self, mesh_state):
        # Clear existing mesh items
        for item in self.mesh_items:
            self.removeItem(item)
        self.mesh_items.clear()
        
        # Add canonical mesh (body)
        if mesh_state.canonical_mesh['vertices'] is not None:
            body = gl.GLMeshItem(
                vertexes=mesh_state.canonical_mesh['vertices'],
                faces=mesh_state.canonical_mesh['faces'],
                smooth=True,
                drawEdges=False,
                color=mesh_state.canonical_mesh['color']
            )
            self.addItem(body)
            self.mesh_items.append(body)
        
        # TODO: Show either active or canonical mesh
        if not np.array_equal(mesh_state.active_mesh['vertices'], 
                            mesh_state.canonical_mesh['vertices']):
            active = gl.GLMeshItem(
                vertexes=mesh_state.active_mesh['vertices'],
                faces=mesh_state.canonical_mesh['faces'],  # faces should be the same
                smooth=True,
                drawEdges=False,
                color=(0.8, 0.8, 0.8, 0.5)  # semi-transparent gray
            )
            self.addItem(active)
            self.mesh_items.append(active)
        
        if len(mesh_state.masked_patch_idxs_dict) != 0:
            for garment_label in ['upper', 'lower']:
                garment_vert_idxs = mesh_state.masked_patch_idxs_dict[garment_label]
                garment_verts = mesh_state.active_mesh['vertices'][garment_vert_idxs]
                garment = gl.GLScatterPlotItem(
                    pos=garment_verts,
                    color=(0.0, 0.5, 0.0, 1.0),
                    size=5
                )
                self.addItem(garment)
                self.mesh_items.append(garment)
            
        # TODO: Add patches to view 2 or add a switch of views in view1.
        '''
        # Add garment patches
        if hasattr(mesh_state, 'masked_patch_idxs_dict'):
            for patch_label, patch_vert_idxs in mesh_state.masked_patch_idxs_dict.items():
                if patch_vert_idxs is not None:
                    # Determine color based on garment type
                    color = (0.6, 0.8, 1.0, 1.0)  # default light blue
                    if 'upper' in patch_label:
                        color = (0.6, 0.8, 1.0, 1.0)  # light blue for upper
                    elif 'lower' in patch_label:
                        color = (0.0, 0.5, 0.0, 1.0)  # dark green for lower
                    
                    patch_verts = mesh_state.active_mesh['vertices'][patch_vert_idxs]
                    patch = gl.GLScatterPlotItem(
                        pos=patch_verts,
                        color=color,
                        size=5
                    )
                    self.addItem(patch)
                    self.mesh_items.append(patch)
        '''
        
        # Add seamlines if they exist
        # TODO: Verify this!
        if hasattr(mesh_state, 'masked_seamlines_dict'):
            for seamline_label, seamline_verts in mesh_state.masked_seamlines_dict.items():
                if seamline_verts is not None:
                    seamline = gl.GLLinePlotItem(
                        pos=seamline_verts,
                        color=(1.0, 0.0, 0.0, 1.0),  # red
                        width=2
                    )
                    self.addItem(seamline)
                    self.mesh_items.append(seamline)

class PatternView(pg.ImageView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.setImage(np.zeros((100, 100)))  # Create empty initial image
        
    def update_pattern(self, pattern_array):
        self.setImage(pattern_array)

class GarmentDesignerUI(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setup_ui()
        self.mesh_state = MeshState(
            self.args.body_set, 
            self.args.use_darts, 
            self.args.apply_remesh
        )
        self.update_skintight_view()
        
    def setup_ui(self):
        self.setWindowTitle("Garment Designer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create left and right column layouts
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        
        # Create the four view areas
        self.skintight_view = self.create_3d_view("Skintight Mesh")
        self.colored_view = self.create_3d_view("Colored Mesh")
        self.pattern_view = self.create_pattern_view("Sewing Pattern")
        self.simulation_view = self.create_3d_view("Simulation")
        
        # Add views to columns
        left_column.addWidget(self.skintight_view)
        left_column.addWidget(self.colored_view)
        right_column.addWidget(self.pattern_view)
        right_column.addWidget(self.simulation_view)
        
        # Create floating menu
        self.menu = self.create_floating_menu()
        
        # Add columns to main layout
        layout.addLayout(left_column)
        layout.addLayout(right_column)
        
    def update_parameter(self, param_name, value):
        self.mesh_state.update_parameter(param_name, value)
        # TODO: Consider not automatically updating whole garments for efficiency.
        self.update_garments()
        self.update_views()
            
    def update_garments(self):
        try:
            self.mesh_state.update_garment_meshes()
            self.update_skintight_view()
            
        except Exception as e:
            print(f"Error updating garments: {e}")
            
    def update_skintight_view(self):
        """Update the skintight mesh view with current mesh state"""
        if isinstance(self.skintight_view, QFrame):
            view = self.skintight_view.findChild(Mesh3DView)
            if view:
                view.update_meshes(self.mesh_state)
        
    def create_3d_view(self, title):
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        layout = QVBoxLayout(frame)
        
        # Add title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create 3D view widget
        view = Mesh3DView()
        layout.addWidget(view)
        
        # Add save button
        save_button = QPushButton("Save")
        save_button.setFixedWidth(80)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
        
        # Connect save button with special handling for view 1
        view_number = {
            "Skintight Mesh": 1,
            "Colored Mesh": 2,
            "Simulation": 4
        }.get(title)
        
        if view_number == 1:
            save_button.clicked.connect(lambda: self.handle_view1_save(view))
        elif view_number:
            save_button.clicked.connect(lambda: self.save_view(view_number, view))
        
        return frame
    
    def handle_view1_save(self, view):
        # Call finalize method
        self.mesh_state.finalize()
        
        self.mesh_state.optimize()
        
        # Save the view after finalization
        self.save_view(1, view)
        
    def create_pattern_view(self, title):
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        layout = QVBoxLayout(frame)
        
        # Add title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create pattern view widget
        view = PatternView()
        layout.addWidget(view)
        
        # Add save button
        save_button = QPushButton("Save")
        save_button.setFixedWidth(80)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
        
        # Connect save button
        save_button.clicked.connect(lambda: self.save_view(3, view))
        
        return frame
    
    def create_floating_menu(self):
        menu = QWidget(self)
        menu.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        layout = QVBoxLayout(menu)
        
        # Create sliders for lengths
        length_params = {
            "shirt_length": ("Shirt Length", 0.1, 0.5),
            "sleeve_length": ("Sleeve Length", 0.1, 0.5),
            "pant_length": ("Pant Length", 0.25, 0.9)
        }
        
        for param_name, (label_text, min_val, max_val) in length_params.items():
            slider_layout = QHBoxLayout()
            label = QLabel(label_text)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(min_val * 100), int(max_val * 100))
            slider.setValue(int(min_val * 100))
            slider.valueChanged.connect(
                lambda value, name=param_name: self.update_parameter(name, value/100))
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            layout.addLayout(slider_layout)
        
        # Create sliders for looseness
        looseness_params = {
            "shirt_looseness": ("Shirt Looseness", 0.8, 1.2),
            "sleeve_looseness": ("Sleeve Looseness", 0.8, 1.2),
            "pant_looseness": ("Pant Looseness", 0.8, 1.2)
        }
        
        for param_name, (label_text, min_val, max_val) in looseness_params.items():
            slider_layout = QHBoxLayout()
            label = QLabel(label_text)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(min_val * 100), int(max_val * 100))
            slider.setValue(int(min_val * 100))
            slider.valueChanged.connect(
                lambda value, name=param_name: self.update_parameter(name, value/100))
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            layout.addLayout(slider_layout)
            
        menu.move(10, 10)
        menu.show()
        return menu
    
    def update_views(self):
        # This method will be connected to your backend
        # For now it's a placeholder for the backend integration
        self.update_garments()
    
    def save_view(self, view_number, view):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = f"view{view_number}"
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"view_{timestamp}.png")
        
        # Grab the view contents
        pixmap = view.grab()
        pixmap.save(filename)

    def update_skintight_mesh(self, vertices, faces):
        """Update the skintight mesh view with new data"""
        self.skintight_view.update_mesh(vertices, faces)
    
    def update_colored_mesh(self, vertices, faces, colors):
        """Update the colored mesh view with new data"""
        self.colored_view.update_mesh(vertices, faces, colors)
    
    def update_pattern(self, pattern_array):
        """Update the pattern view with new data"""
        self.pattern_view.update_pattern(pattern_array)
    
    def update_simulation(self, vertices, faces):
        """Update the simulation view with new data"""
        self.simulation_view.update_mesh(vertices, faces)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_darts', action='store_true', dest='use_darts',
                        help='whether to use darts in the design and parameterization algorithm')
    parser.add_argument('--apply_remesh', action='store_true', dest='apply_remesh',
                        help='apply remesh to the patches (possibly improve seamline alignment and patch quality)')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply',
                        help='')
    parser.add_argument('--design', '-D', type=str, default='default')
    parser.add_argument('--body_set', type=str, default="set2")
    parser.add_argument('--project_dir', type=str, default='/home/kristijan/TailorLang/', 
                        help='an absolute path to this project')
    parser.add_argument('--smpl_dir', type=str, default="/home/kristijan/data/smpl/models/")
    parser.add_argument('--gender', type=str, default="female")
    parser.add_argument('--standard_export', action='store_true', dest='standard_export')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = GarmentDesignerUI(args)
    window.show()
    sys.exit(app.exec())
