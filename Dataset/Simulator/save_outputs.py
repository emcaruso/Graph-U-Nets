# # API: https://docs.salome-platform.org/latest/gui/SMESH/smeshBuilder.html
#
# import os
# import sys
# import SMESH
# import pickle
# import salome
# from salome.smesh import smeshBuilder
# from paraview.simple import *
#
#
# # Directories
# data_dir = os.path.realpath(os.path.dirname(__file__)+"/../Data")
# utils_dir = data_dir+"/../Dataset_blender"
# results_dir = data_dir+"/3d_models/med_results"
#
# # load utils file
# sys.path.insert(1, utils_dir)
# from utils import *
#
# # init salome
# salome.salome_init()
#
# # smeshBuilder
# assert(smeshBuilder is not None)
# smesh = smeshBuilder.New()
#
# # input surfaces
# list_files = os.listdir(results_dir)
# list_files.sort()
# print(str(len(list_files))+" files found in "+results_dir)
#
#
# # load utils file
# sys.path.insert(1, utils_dir)
# from utils import *
#
# # init salome
# salome.salome_init()
#
# # Iterate through input surfaces
# print("\nMeshing geometries wih Salome meca")
# for i,med_filename in enumerate(list_files):
#     print_progress_bar(i,len(list_files))
#     # set names
#     name = med_filename[0:-4]
#     mesh = smesh.CreateMeshesFromMED(results_dir+"/"+med_filename)
#
#
