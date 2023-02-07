import bpy
from bpy import context
import numpy as np
import bmesh
import math
import os
import sys
from operator import itemgetter

import sys

# load utils file
data_dir = os.path.realpath(os.path.dirname(__file__)+"/../Data")
tables_dir = data_dir+"/3d_models/table_results"
surfaces_in_dir = data_dir+"/3d_models/surfaces_in"
surfaces_out_dir = data_dir+"/3d_models/surfaces_out"
sys.path.insert(1, os.path.realpath(os.path.dirname(__file__)))
from utils import *

# clear current output surfaces
clear_directory(surfaces_out_dir)

# clear data collection
try:
    def_coll = bpy.data.collections['Deformed']
    for c in def_coll.objects:
        bpy.data.objects.remove(c)
except:
    {}

def get_verts_and_displ_from_table(table_path):
    res = []
    with open(table_path) as f:
        lines = [line.rstrip() for line in f]
    for i,line in enumerate(lines[5:-1]):
        x=float(line[91:103])
        y=float(line[104:116])
        z=float(line[117:129])
        dx=float(line[130:142])
        dy=float(line[143:155])
        dz=float(line[156:168])
        id=int(line[18:24])
        res.append([[x,y,z],[dx,dy,dz],id])

    res.sort(key=lambda x:x[2])

    k=0
    for j in res:
        assert j[2]==k+1
        k=j[2]

    return res


def check_vertices(mesh,res):
    assert(len(mesh.vertices) == len(res))
    for i in range(len(mesh.vertices)):
        assert(i+1==res[i][2])
        if (abs(mesh.vertices[i].co[0]-res[i][0][0])>0.0001 or
            abs(mesh.vertices[i].co[1]-res[i][0][1])>0.0001 or
            abs(mesh.vertices[i].co[2]-res[i][0][2])>0.0001):
            print(mesh.vertices[i].co,res[i][0])
            print("FAIL: Vertices don't match")
            exit(1)
            # assert(abs(mesh.vertices[i].co[0]-res[i][0][0])<0.000001)
            # assert(abs(mesh.vertices[i].co[1]-res[i][0][1])<0.000001)
            # assert(abs(mesh.vertices[i].co[2]-res[i][0][2])<0.000001)

def apply_displacements(mesh,res):
    assert(len(mesh.vertices) == len(res))
    for i in range(len(mesh.vertices)):
        assert(i+1==res[i][2])
        mesh.vertices[i].co[0]+=res[i][1][0]
        mesh.vertices[i].co[1]+=res[i][1][1]
        mesh.vertices[i].co[2]+=res[i][1][2]


# GENERATE OUTPUT SURFACES
# Iterate through surfaces
list_files = os.listdir(surfaces_in_dir)
list_files.sort()
for i,surface_in_filename in enumerate(list_files):
    print_progress_bar(i,len(list_files))
    print(" ")
    obj_name = surface_in_filename[0:-4]

    # import
    path_in = surfaces_in_dir+"/"+surface_in_filename
    bpy.ops.import_mesh.stl(filepath=path_in)
    obj = bpy.context.view_layer.objects.active
    obj_mesh = obj.data


    res=get_verts_and_displ_from_table(tables_dir+"/"+obj_name+".txt")
    check_vertices(obj_mesh,res)
    apply_displacements(obj_mesh,res)

    # export stl
    obj.select_set(True)
    path_out = os.path.join(surfaces_out_dir, obj_name + '.stl')
    bpy.ops.export_mesh.stl(filepath=path_out,use_selection=True)
    obj.select_set(False)
