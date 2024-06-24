import os
import sys
import scipy
import numpy
import torch
import bpy

sampleID = "06429"

scriptPath = os.path.dirname(bpy.data.filepath)
os.chdir(scriptPath+"/..")
sys.path.append(os.getcwd())
print("skirtify: running in "+os.getcwd())

from DataReader.read import DataReader as DReader
from DataReader.view import loadSMPL

reader = DReader("./Data")
info = reader.read_info(sampleID)

# We actually want the 0-pose.
info["poses"] = numpy.zeros((72, 1))

# Instantiate model
loadSMPL(sampleID, info)
