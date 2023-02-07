import bpy
import os
import json
import shutil
import numpy as np
import math
import mathutils
import sys
from shutil import copy
import random
import bmesh
import pickle

# directories
datablend_dir = os.path.dirname(bpy.data.filepath)
data_dir = os.path.realpath(datablend_dir+"/../Data")
models3d_dir = data_dir+"/3d_models/surfaces_in/"
pkl_dir = data_dir+"/3d_models/pkls"

# load utils file
sys.path.insert(1, datablend_dir)
from utils import *

############################
## parameters
############################

n_objects = 2
line_num = 10
random.seed(10)
tex1 = bpy.data.textures.new("displacement1", 'CLOUDS')
tex2 = bpy.data.textures.new("displacement2", 'DISTORTED_NOISE')

###########################
## deforming shapes
###########################


bpy.ops.object.mode_set(mode='OBJECT')

ref = bpy.data.objects['reference']

assert(ref is not None)

scene = bpy.context.scene
def_coll = 0
try:
    def_coll = bpy.data.collections['Deformed']
    for c in def_coll.objects:
        bpy.data.objects.remove(c)
except:
    def_coll = bpy.data.collections.new("Deformed")
    scene.collection.children.link(def_coll)

rows=math.sqrt(n_objects)
offs_x = 1

ref.select_set(False)
# create deformations
print("\nCreating deformed shapes in Blender")
for i in range(n_objects):
    print_progress_bar(i,n_objects)

    new_obj = ref.copy()
    new_obj.data = ref.data.copy()
    new_obj.animation_data_clear()
    bpy.context.collection.objects.link(new_obj)
    obj_name = "copy_"+f"{i:03d}"
    new_obj.name = obj_name
    axes = ["X","Y","Z"]

    #deform twist
    axis_id= random.randint(0,1)
    angle = random.uniform(-0.18,0.18)
    #angle = random.random()
    mod_twist = new_obj.modifiers.new("Simple Deform","SIMPLE_DEFORM")
    mod_twist.deform_method = "TWIST"
    mod_twist.deform_axis = axes[axis_id]
    mod_twist.angle=angle

    #deform bend
    axis_id= random.randint(0,1)
    angle = random.uniform(-0.18,0.18)
    #angle = random.random()
    mod_bend = new_obj.modifiers.new("Simple Deform","SIMPLE_DEFORM")
    mod_bend.deform_method = "BEND"
    mod_bend.deform_axis = axes[axis_id]
    mod_bend.angle=angle

    #deform taper
    axis_id= random.randint(0,1)
    angle = random.uniform(-0.18,0.18)
    #angle = random.random()
    mod_taper = new_obj.modifiers.new("Simple Deform","SIMPLE_DEFORM")
    mod_taper.deform_method = "TAPER"
    mod_taper.deform_axis = axes[axis_id]
    mod_taper.angle=angle

    #deform taper_z
    axis_id= random.randint(0,1)
    angle = random.uniform(-0.18,0.18)
    #angle = random.random()
    mod_taper_z = new_obj.modifiers.new("Simple Deform","SIMPLE_DEFORM")
    mod_taper_z.deform_method = "TAPER"
    mod_taper_z.deform_axis = "Z"
    mod_taper_z.angle=angle

  #deform displacements 1
   mod_displ1 = new_obj.modifiers.new("Displace","DISPLACE")
   mod_displ1.texture = tex1
   mod_displ1.texture_coords = "GLOBAL"
   mod_displ1.strength = random.uniform(0,0.001)
   mod_displ1.mid_level = 0

  #deform displacements 2
   mod_displ2 = new_obj.modifiers.new("Displace","DISPLACE")
   mod_displ2.texture = tex2
   mod_displ2.texture_coords = "GLOBAL"
   mod_displ2.strength = random.uniform(0,0.001)
   mod_displ2.mid_level = 0

    #deform size
    offs = random.uniform(-0.05,0.05)
    new_obj.scale=new_obj.scale*(1+offs)

    #deform size axis
    axis = random.randint(0,1)
    offs = random.uniform(-0.05,0.05)
    new_obj.scale[axis]=new_obj.scale[axis]*(1+offs)

    # apply modifiers
    bpy.context.view_layer.objects.active = new_obj
    for modifier in new_obj.modifiers:
        bpy.ops.object.modifier_apply(
            modifier=modifier.name
        )

    # add object to collection
    def_coll.objects.link(new_obj) #link it with collection
    bpy.context.scene.collection.objects.unlink(new_obj) #unlink it from master collection

    # export stl
    new_obj.select_set(True)
    stl_path = os.path.join(models3d_dir, obj_name + '.stl')
    bpy.ops.export_mesh.stl(filepath=stl_path,use_selection=True)
    new_obj.select_set(False)

    # translate object
    new_obj.matrix_world.translation[0]= offs_x + (int)(i/line_num)
    new_obj.matrix_world.translation[1]= (int)(i%line_num)

###########################
## Saving groups
###########################

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='FACE')
bpy.ops.mesh.select_all(action='DESELECT')
bm = bmesh.new()   # create an empty BMesh
bm.from_mesh(ref.data)
bm.faces.ensure_lookup_table()
l = bm.faces.layers.face_map.active
groups = [[] for i in range(len(ref.face_maps))]

print("\Loading groups")
for i,f in enumerate(bm.faces):
    print_progress_bar(i,len(bm.faces))
    groups[f[l]].append(i)

print("\nSaving pickles")
for i,g in enumerate(groups):
    print_progress_bar(i,len(groups))
    # save group
    name = ref.face_maps[i].name
    # open a file, where you ant to store the data
    file = open(pkl_dir+"/"+name+".pkl", 'wb')
    pickle.dump(g, file)

bm.free()
