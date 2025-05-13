import sys
sys.path.append('../')

from generation.objects import gen_slot
import bpy
import os
import random
import numpy as np
import pandas as pd
from generation.materials import add_material_from_file, change_material_random, add_puddle2
from generation.objects import uv_editing, duplicate_link_obj, add_paved_path_generation
from generation.rendering import render_img_mask
from generation.scenes import SceneModifier

# bpy.ops.wm.open_mainfile(filepath='./test.blend')
# add_paved_path_generation(os.path.join('/home/sczone/disk1/share/3d/blender_slots/elements/path/Paved_Path_Generator1.3.blend'), 'STARS PATH', 5, 30, (0, 0))
# bpy.ops.wm.save_mainfile(filepath='./test.blend')

# line_type = random.choice(['open', 'closed', 'half_closed'])
# slot_mat_suffix = '/home/sczone/disk1/share/3d/blender_slots/elements/materials/slots/indoor/'
# slot, marking_pts = gen_slot(np.pi/3, 2.501, 5.501, 'closed', 0.2, 'slots')
# if len(slot_mat_suffix) > 0:
#     change_material_random(slot, slot_mat_suffix)
#     uv_editing(slot, (2, 2))

# locations, rotations = [], []
# for i in range(30):
#     locations += [[-50+i*3, -7, 0], [-50+i*3, 7, 0]]
#     rotations += [[0, 0, 0], [0, 0, 0]]
# duplicate_link_obj(slot, locations, rotations)
# bpy.data.objects.remove(slot)

# bpy.ops.wm.save_mainfile(filepath='./test.blend')


# bpy.ops.wm.open_mainfile(filepath='./a.blend')
# names = ['超级blender.001', '超级blender.002', '超级blender.003', '超级blender.004', '超级blender.005', '超级blender.006', '超级blender.007', '超级blender.008', '超级blender.009', '超级blender.010', '超级blender.011', '超级blender.012', '超级blender.013', '超级blender.014', '超级blender.015', '超级blender.016', '超级blender.017', '超级blender.018', '超级blender.019', '超级blender.020', '超级blender.021', '超级blender.022', '超级blender.023', '超级blender.024', '超级blender.025', '超级blender.026', '超级blender.027']
# obj = bpy.data.objects['Plane']
# for i in range(len(names)):
#     name = names[i]
#     add_material_from_file('/home/sczone/disk1/share/3d/blender_slots/assets/materials/ground/ground.blend', names[i], obj)
# uv_editing(obj, (2, 2))
# bpy.ops.wm.save_mainfile(filepath='./test{}.blend'.format(str(i)))  

# bpy.ops.wm.open_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/elements/grounds/spare_ground.blend')
# obj = bpy.data.objects['spare_ground']
# add_material_from_file('/home/sczone/disk1/share/3d/blender_slots/elements/materials/grounds/indoor/green_rubber1.blend', 'green_rubber1', obj)
# uv_editing(obj, (2, 2))
# bpy.ops.wm.save_mainfile(filepath='./test.blend')


# csv_path = '/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5/ego_position_parking.csv'
# npy_path = '/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/e2e/ego.npy'
# poses = np.load(npy_path)
# df = pd.read_csv(csv_path, encoding='utf-8')
# # df_loc = df.drop('time', axis=1)
# df_loc = df.drop_duplicates()
# output = []
# for pose in poses:
#     x = pose[3]
#     res = df_loc[df_loc['x']==x].iloc[0]
#     print(res)
#     output.append([0, 0, float(res['head']), float(res['x']), float(res['y']), 0, float(res['time'])])
# print(len(output))
# print(output[0])
# print(poses[0])
# np.save('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/ego.npy', np.array(output))

# csv_path = '../../code/results/3_3/ego_position_parking.csv'

# ego_poses = []
# df = pd.read_csv(csv_path, encoding='utf-8')
# df_loc = df.drop('time', axis=1)
# df_loc = df_loc.drop_duplicates()
# for _, row in df_loc.iterrows():
#     pose = np.array([0, 0, float(row['head']), float(row['x']), float(row['y']), 0])
#     ego_poses.append(pose)
# if len(ego_poses) > 200:
#     pose_index = random.sample(range(len(ego_poses)), 200)
#     pose_index.sort()
#     print(pose_index)
#     ego_poses = [ego_poses[i] for i in pose_index]
# #ego_poses = [item+np.array([0, 0, 0, -31.73, -25.64, 0]) for item in ego_poses]

# print(ego_poses[:3])

# import addon_utils
# import random

# addon_utils.enable('AgedFX', default_set=True)

# filepath = '/home/sczone/disk1/share/3d/blender_slots/code/results/1.blend'
# output = '/home/sczone/disk1/share/3d/blender_slots/code/results/test.blend'

# bpy.ops.wm.open_mainfile(filepath=filepath)

# obj = bpy.data.objects['ground']
# obj.select_set(True)
# bpy.ops.object.make_dust_selected_operator()
# bpy.context.scene.aged_fx.dust_amount = random.random()
# bpy.context.scene.aged_fx.dust_top = random.random()
# bpy.context.scene.aged_fx.dust_side = random.random()
# bpy.context.scene.aged_fx.dust_scale = random.uniform(0, 20)
# obj.select_set(False)
# # bpy.context.view_layer.update()


# obj = bpy.data.objects['road1']
# obj.select_set(True)
# bpy.ops.object.make_deformations_selected_operator()
# bpy.context.scene.aged_fx.bumps = random.random()
# bpy.context.scene.aged_fx.scratches = random.random()
# bpy.context.scene.aged_fx.deformations_amount = random.random()
# bpy.context.scene.aged_fx.deformations_scale = random.uniform(0, 10)
# obj.select_set(False)
# # bpy.context.view_layer.update()


# bpy.ops.wm.open_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/test.blend')
# ground = bpy.data.objects['spare_ground']
# add_material_from_file('/home/sczone/disk1/share/3d/blender_slots/elements/materials/grounds/indoor/green_rubber1.blend', 'green_rubber1', ground)
# ground_mat_name = ground.data.materials.keys()[0]
# ground_mat = bpy.data.materials[ground_mat_name]

# add_puddle((9.65, -4.85, 0), 100, ground_mat)

# bpy.ops.wm.save_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/test3.blend')

# bpy.ops.wm.open_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/out_puddle2.blend')
# render_img_mask('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/e2e3/front_120', '/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/e2e3/egos3.npy')


# modifier = SceneModifier('/home/sczone/disk1/share/3d/blender_slots/elements', '/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/out_puddle2.blend', np.pi/2, lock=True, obstacle=True)
# modifier.modify_static_objs()
# modifier.add_dynamic_objs_on_roads()
# occupied_slots = modifier.add_dynamic_objs_in_slots()
# modifier.save_output_scene()
# ground = bpy.data.objects['spare_ground']
# add_material_from_file('/home/sczone/disk1/share/3d/blender_slots/elements/materials/grounds/indoor/concrete1.blend', 'concrete1', ground)

# mat = bpy.data.materials['concrete1']
# add_puddle2(mat, 0.5)


# bpy.ops.wm.save_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/5_1/out_puddle.blend')


bpy.ops.wm.open_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/results/2/out.blend')
render_img_mask('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/results/2/front_120', '/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/predpaths/roadside_vertical.npy')

