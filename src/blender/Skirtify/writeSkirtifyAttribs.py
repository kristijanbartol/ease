import os
import sys
import scipy
import numpy
import torch
import csv
import bpy
import bmesh

scriptPath = os.path.dirname(bpy.data.filepath)
os.chdir(scriptPath)
sys.path.append(scriptPath)
print("skirtify: running in "+os.getcwd())

ao_bak = bpy.context.view_layer.objects.active
template_object = bpy.data.objects["(template)"]
bpy.context.view_layer.objects.active = template_object
mesh = template_object.data
bpy.ops.object.mode_set(mode='EDIT')
bmesh = bmesh.from_edit_mesh(mesh)

# Debug info
print("template object (TO):")
print("  "+str(template_object))
print("")
print("TO mesh:")
print("  "+str(mesh))
print("")
print("TO mesh attributes:")
for a in mesh.attributes:
    print("  - "+a.name)
print("")

# Write csv files
# - vertex attribs
attribs = [
    "_idx", "SkirtCutV", "PProjectW", "ProjectL",
    "ProjectR", "PProjectA", "PProjectB", "FrontEndL",
    "FrontEndR", "BackEndL", "BackEndR"
]
csvfilename = scriptPath+"/skirtifyVertAttribs.csv"
print("Writing vertex attributes to '"+csvfilename+"'")
with open(csvfilename, 'w', newline='') as csvfile:
    wr = csv.writer(
        csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL
    )
    wr.writerow(attribs)
    idx=-1
    for vert in bmesh.verts:
        idx = idx+1
        wr.writerow([
            idx, vert[bmesh.verts.layers.float.get('SkirtCutV')],
            vert[bmesh.verts.layers.float.get('PProjectW')],
            vert[bmesh.verts.layers.int.get('ProjectL')],
            vert[bmesh.verts.layers.int.get('ProjectR')],
            vert[bmesh.verts.layers.int.get('PProjectA')],
            vert[bmesh.verts.layers.int.get('PProjectB')],
            vert[bmesh.verts.layers.int.get('FrontEndL')],
            vert[bmesh.verts.layers.int.get('FrontEndR')],
            vert[bmesh.verts.layers.int.get('BackEndL')],
            vert[bmesh.verts.layers.int.get('BackEndR')]
       ])
# - face attribs
attribs = ["_idx", "SkirtCutF"]
csvfilename = scriptPath+"/skirtifyFaceAttribs.csv"
print("Writing face attributes to '"+csvfilename+"'")
with open(csvfilename, 'w', newline='') as csvfile:
    wr = csv.writer(
        csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL
    )
    wr.writerow(attribs)
    idx=-1
    for face in bmesh.faces:
        idx = idx+1
        wr.writerow([
            idx, face[bmesh.faces.layers.int.get('SkirtCutF')]
       ])

# Done!
bpy.context.view_layer.objects.active = ao_bak
print("Done!")
