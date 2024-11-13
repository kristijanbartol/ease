import sys
import os
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QPushButton, QFrame)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl
import pyqtgraph as pg

class GarmentParameters:
    def __init__(self):
        # Initialize with default values
        self.shirt_length = 0.3
        self.sleeve_length = 0.3
        self.pant_length = 0.3
        self.shirt_looseness = 0.8
        self.sleeve_looseness = 0.8
        self.pant_looseness = 0.8
        
    def update_parameter(self, param_name, value):
        if hasattr(self, param_name):
            setattr(self, param_name, value)
            return True
        return False

class Mesh3DView(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_view()
        
    def setup_view(self):
        # Add grid for reference
        grid = gl.GLGridItem()
        self.addItem(grid)
        
        # Configure lighting
        self.opts['lighting'] = True
        self.opts['ambient'] = 0.5
        self.opts['diffuse'] = 0.8
        
        # Setup camera
        self.setCameraPosition(distance=2.5)
        self.opts['fov'] = 60
        
    def update_mesh(self, vertices, faces, colors=None):
        # Clear existing mesh items
        for item in self.items:
            if isinstance(item, gl.GLMeshItem):
                self.removeItem(item)
        
        # Create and add new mesh
        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces, smooth=True,
                            drawEdges=False, color=colors if colors is not None else (0.7, 0.7, 0.7, 1))
        self.addItem(mesh)

class PatternView(pg.ImageView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.setImage(np.zeros((100, 100)))  # Create empty initial image
        
    def update_pattern(self, pattern_array):
        self.setImage(pattern_array)

class GarmentDesignerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.parameters = GarmentParameters()
        self.setup_ui()
        
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
        
        # Connect save button
        view_number = {
            "Skintight Mesh": 1,
            "Colored Mesh": 2,
            "Simulation": 4
        }.get(title)
        if view_number:
            save_button.clicked.connect(lambda: self.save_view(view_number, view))
        
        return frame
        
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
            "shirt_length": ("Shirt Length", 0.3, 0.9),
            "sleeve_length": ("Sleeve Length", 0.3, 0.9),
            "pant_length": ("Pant Length", 0.3, 0.9)
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
    
    def update_parameter(self, param_name, value):
        if self.parameters.update_parameter(param_name, value):
            self.update_views()
    
    def update_views(self):
        # This method will be connected to your backend
        # For now it's a placeholder for the backend integration
        pass
    
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
    app = QApplication(sys.argv)
    window = GarmentDesignerUI()
    window.show()
    sys.exit(app.exec())