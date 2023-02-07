# API: https://docs.salome-platform.org/latest/gui/SMESH/smeshBuilder.html

import os
import sys
import salome
import SMESH
import pickle
from salome.smesh import smeshBuilder

# Directories
data_dir = os.path.realpath(os.path.dirname(__file__)+"/../Data")
utils_dir = data_dir+"/../Dataset_blender"
surfaces_in_dir = data_dir+"/3d_models/surfaces_in"
volumes_in_dir = data_dir+"/3d_models/volumes_in"
med_dir = data_dir+"/3d_models/med_meshes"
pkl_dir = data_dir+"/3d_models/pkls"

# load utils file
sys.path.insert(1, utils_dir)
from utils import *

# clear current med files
clear_directory(med_dir)

# init salome
salome.salome_init()

# smeshBuilder
assert(smeshBuilder is not None)
smesh = smeshBuilder.New()

# input surfaces
list_files = os.listdir(surfaces_in_dir)
list_files.sort()
print(str(len(list_files))+" files found in "+surfaces_in_dir)

# load groups from pickles
fix_list = pickle.load( open( pkl_dir+"/fix.pkl", "rb" ) )
top_list = pickle.load( open( pkl_dir+"/top.pkl", "rb" ) )
bottom_list = pickle.load( open( pkl_dir+"/bottom.pkl", "rb" ) )
external_list = pickle.load( open( pkl_dir+"/external.pkl", "rb" ) )
external_list = external_list+fix_list
internal_list = pickle.load( open( pkl_dir+"/internal.pkl", "rb" ) )

# Iterate through input surfaces
print("\nMeshing geometries wih Salome meca")
for i,surface_in_filename in enumerate(list_files):
    print_progress_bar(i,len(list_files))
    # set names
    name = surface_in_filename[0:-4]
    volumes_in_dir_filename = name+".unv"
    med_filename = name+".med"
    # import mesh from stl file
    mesh = smesh.CreateMeshesFromSTL(surfaces_in_dir+"/"+surface_in_filename)
    mesh.SetName(name)
    # meshing algorithm
    algo3d = mesh.Tetrahedron()
    algo3d.MaxElementVolume(100)
    # creating groups
    mesh.MakeGroupByIds("Top", SMESH.FACE, top_list)
    mesh.MakeGroupByIds("Bottom", SMESH.FACE, bottom_list)
    mesh.MakeGroupByIds("External", SMESH.FACE, external_list)
    mesh.MakeGroupByIds("Internal", SMESH.FACE, internal_list)
    mesh.MakeGroupByIds("Fix", SMESH.FACE, fix_list)
    mesh.MakeGroupByIds("Top", SMESH.NODE, top_list)
    mesh.MakeGroupByIds("Bottom", SMESH.NODE, bottom_list)
    mesh.MakeGroupByIds("External", SMESH.NODE, external_list)
    mesh.MakeGroupByIds("Internal", SMESH.NODE, internal_list)
    mesh.MakeGroupByIds("Fix", SMESH.NODE, fix_list)
    # clear and compute mesh
    mesh.Clear()
    mesh.Compute()
    # Export med file
    # mesh.ExportUNV(volumes_in_dir+"/"+volumes_in_dir_filename)
    mesh.ExportMED(med_dir+"/"+med_filename)
