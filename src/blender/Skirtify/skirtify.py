import os
import sys
import scipy
import numpy
import torch
import csv
import bpy
import bmesh

# Helper function
def ensureAttrib (mesh, aname, atype, adomain):
    existing = None
    for a in mesh.attributes:
        if a.name == aname:
            print("Found attrib: "+str(a))
            return a
    newAttrib = mesh.attributes.new(name=aname, type=atype, domain=adomain)
    print("Created missing attrib: "+str(newAttrib))
    return newAttrib
            
scriptPath = os.path.dirname(bpy.data.filepath)
os.chdir(scriptPath)
sys.path.append(scriptPath)
print("skirtify: running in "+os.getcwd())

active_object = bpy.context.view_layer.objects.active
mesh = active_object.data

# Debug info
print("active object (AO):")
print("  "+str(active_object))
print("")
print("AO mesh:")
print("  "+str(mesh))
print("")
print("AO mesh attributes (before):")
for a in mesh.attributes:
    print("  - "+a.name)
print("")

# Prepare mesh attributes
ensureAttrib(mesh, aname="SkirtCutV", atype="FLOAT", adomain="POINT")
ensureAttrib(mesh, aname="PProjectW", atype="FLOAT", adomain="POINT")
ensureAttrib(mesh, aname="ProjectL", atype="INT", adomain="POINT")
ensureAttrib(mesh, aname="ProjectR", atype="INT", adomain="POINT")
ensureAttrib(mesh, aname="PProjectA", atype="INT", adomain="POINT")
ensureAttrib(mesh, aname="PProjectB", atype="INT", adomain="POINT")
ensureAttrib(mesh, aname="FrontEndL", atype="INT", adomain="POINT")
ensureAttrib(mesh, aname="FrontEndR", atype="INT", adomain="POINT")
ensureAttrib(mesh, aname="BackEndL", atype="INT", adomain="POINT")
ensureAttrib(mesh, aname="BackEndR", atype="INT", adomain="POINT")
ensureAttrib(mesh, aname="SkirtCutF", atype="INT", adomain="FACE")
bpy.ops.object.mode_set(mode='EDIT')
bmesh = bmesh.from_edit_mesh(mesh)

# Debug info
print("AO mesh attributes (after):")
for a in mesh.attributes:
    print("  - "+a.name)
print("")

# Read csv files
# - vertex attribs
csvfilename = scriptPath+"/skirtifyVertAttribs.csv"
print("Reading vertex attributes from '"+csvfilename+"'")
with open(csvfilename, newline='') as csvfile:
    rd = csv.DictReader(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
    bmesh.verts.ensure_lookup_table()
    for row in rd:
        vert = bmesh.verts[int(row['_idx'])]
        vert[bmesh.verts.layers.float.get('SkirtCutV')] = float(row['SkirtCutV'])
        vert[bmesh.verts.layers.float.get('PProjectW')] = float(row['PProjectW'])
        vert[bmesh.verts.layers.int.get('ProjectL')] = int(row['ProjectL'])
        vert[bmesh.verts.layers.int.get('ProjectR')] = int(row['ProjectR'])
        vert[bmesh.verts.layers.int.get('PProjectA')] = int(row['PProjectA'])
        vert[bmesh.verts.layers.int.get('PProjectB')] = int(row['PProjectB'])
        vert[bmesh.verts.layers.int.get('FrontEndL')] = int(row['FrontEndL'])
        vert[bmesh.verts.layers.int.get('FrontEndR')] = int(row['FrontEndR'])
        vert[bmesh.verts.layers.int.get('BackEndL')] = int(row['BackEndL'])
        vert[bmesh.verts.layers.int.get('BackEndR')] = int(row['BackEndR'])
# - face attribs
attribs = ["_idx", "SkirtCutF"]
csvfilename = scriptPath+"/skirtifyFaceAttribs.csv"
print("Reading face attributes from '"+csvfilename+"'")
with open(csvfilename, newline='') as csvfile:
    rd = csv.DictReader(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
    bmesh.faces.ensure_lookup_table()
    for row in rd:
        face = bmesh.faces[int(row['_idx'])]
        face[bmesh.faces.layers.int.get('SkirtCutF')] = int(row['SkirtCutF'])

# Execute the Skirtify geometry nodes tool
bpy.ops.geometry.execute_node_group(name="Skirtify")

# Done!
print("Done!")
