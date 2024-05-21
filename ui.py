import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QComboBox
from PyQt5.QtCore import Qt
import open3d as o3d
import numpy as np

class GarmentDesignerUI(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
        # Load the body mesh and garment meshes
        self.body_mesh = o3d.io.read_triangle_mesh("body_mesh.ply")
        self.upper_garment = o3d.io.read_triangle_mesh("results/target_meshes/set2/upper_garment_init.ply")
        self.lower_garment = o3d.io.read_triangle_mesh("results/target_meshes/set2/lower_garment_init.ply")
        
        # Initial visualization
        self.visualize()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Sleeve Length Slider
        sleeve_layout = QHBoxLayout()
        sleeve_label = QLabel('Sleeve Length')
        self.sleeve_slider = QSlider(Qt.Horizontal)
        self.sleeve_slider.setMinimum(0)
        self.sleeve_slider.setMaximum(100)
        self.sleeve_slider.setValue(50)
        self.sleeve_slider.valueChanged.connect(self.update_visualization)
        sleeve_layout.addWidget(sleeve_label)
        sleeve_layout.addWidget(self.sleeve_slider)
        layout.addLayout(sleeve_layout)
        
        # Shirt Size Slider
        shirt_layout = QHBoxLayout()
        shirt_label = QLabel('Shirt Size')
        self.shirt_slider = QSlider(Qt.Horizontal)
        self.shirt_slider.setMinimum(0)
        self.shirt_slider.setMaximum(100)
        self.shirt_slider.setValue(50)
        self.shirt_slider.valueChanged.connect(self.update_visualization)
        shirt_layout.addWidget(shirt_label)
        shirt_layout.addWidget(self.shirt_slider)
        layout.addLayout(shirt_layout)
        
        # Pant Size Slider
        pant_layout = QHBoxLayout()
        pant_label = QLabel('Pant Size')
        self.pant_slider = QSlider(Qt.Horizontal)
        self.pant_slider.setMinimum(0)
        self.pant_slider.setMaximum(100)
        self.pant_slider.setValue(50)
        self.pant_slider.valueChanged.connect(self.update_visualization)
        pant_layout.addWidget(pant_label)
        pant_layout.addWidget(self.pant_slider)
        layout.addLayout(pant_layout)
        
        # Checkpoints Dropdown
        checkpoint_layout = QHBoxLayout()
        checkpoint_label = QLabel('Checkpoints')
        self.checkpoint_dropdown = QComboBox()
        self.checkpoint_dropdown.addItems(['Default', 'Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3'])
        self.checkpoint_dropdown.currentIndexChanged.connect(self.update_visualization)
        checkpoint_layout.addWidget(checkpoint_label)
        checkpoint_layout.addWidget(self.checkpoint_dropdown)
        layout.addLayout(checkpoint_layout)
        
        # Apply Button
        apply_button = QPushButton('Apply')
        apply_button.clicked.connect(self.apply_changes)
        layout.addWidget(apply_button)
        
        self.setLayout(layout)
        self.setWindowTitle('Garment Designer')
        
    def apply_changes(self):
        # Get current values
        sleeve_length = self.sleeve_slider.value()
        shirt_size = self.shirt_slider.value()
        pant_size = self.pant_slider.value()
        checkpoint = self.checkpoint_dropdown.currentText()
        
        # Apply your existing logic to update the garment meshes based on these values
        # For example:
        # self.update_sleeve_length(sleeve_length)
        # self.update_shirt_size(shirt_size)
        # self.update_pant_size(pant_size)
        # self.apply_checkpoint(checkpoint)
        
        # Refresh the visualization
        self.visualize()
    
    def visualize(self):
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Clear the existing geometries
        vis.clear_geometries()
        
        # Add body mesh and garments to the visualization
        self.body_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        self.upper_garment.paint_uniform_color([0.6, 0.8, 1.0])
        self.lower_garment.paint_uniform_color([0.8, 0.5, 0.2])
        vis.add_geometry(self.body_mesh)
        vis.add_geometry(self.upper_garment)
        vis.add_geometry(self.lower_garment)
        
        # Run the visualization
        vis.run()
        vis.destroy_window()
    
    def update_visualization(self):
        # This function can be used to preview changes in real-time
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GarmentDesignerUI()
    ex.show()
    sys.exit(app.exec_())
