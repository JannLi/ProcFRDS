import sys
sys.path.append('../')

from generation.objects import gen_slot
import bpy
import os
import random
import numpy as np
import pandas as pd
import math
import json
from generation.materials import add_material_from_file, change_material_random, add_puddle2
from generation.objects import uv_editing, duplicate_link_obj, add_paved_path_generation
from generation.rendering import render_img_mask
from generation.scenes import SceneModifier

def add_layout_info(layout, obj, name, slot_angle=np.pi/2, slot_type='vertical'):
    obj_info = dict()
    obj_info['location'] = list(obj.location)
    obj_info['rotation'] = list(obj.rotation_euler)
    obj_info['size'] = list(obj.dimensions)
    if name == 'slots':
        obj_info['slot_angle'] = slot_angle
        if slot_type == 'parallel':
            obj_info['slot_angle'] = np.pi
        if slot_type == 'slant':
            obj_info['size'] = [2.5, 5.5, 0]
    obj_list = layout.setdefault(name, [])
    obj_list.append(obj_info)
    return layout

def export_layout(layout, slot_angle, slot_type):
    for obj in bpy.data.objects:
        # if obj.name.startswith('spare_ground'):
        #     self.add_layout_info(obj, 'ground')
        if obj.name.startswith('spare_road'):
            add_layout_info(layout, obj, 'roads')
        elif 'pillar' in obj.name:
            add_layout_info(layout, obj, 'pillars')
        # elif 'hedge' in obj.name:
        #     self.add_layout_info(obj, 'hedges')
        elif 'slot' in obj.name:
            add_layout_info(layout, obj, 'slots', slot_angle, slot_type)
        # elif 'curb' in obj.name:
        #     self.add_layout_info(obj, 'curbs')
        elif 'road' in obj.name:
            add_layout_info(layout, obj, 'lanes')
    return layout

if __name__ == '__main__':
    blend_file = sys.argv[1]
    slot_type = sys.argv[2]
    slot_angle = sys.argv[3]
    out_path = sys.argv[4]
    slot_angle = math.radians(float(slot_angle))

    bpy.ops.wm.open_mainfile(filepath=sys.argv[1])
    layout = dict()
    layout = export_layout(layout, slot_angle, slot_type)
    with open(out_path, 'w') as f:
        json.dump(layout, f, indent=4)
