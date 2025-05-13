import sys
sys.path.append('../')
import bpy
import os
import time
import numpy as np
import open3d as o3d
from blender_utils.blr import get_children_in_scene
from generation.objects import add_object_from_file

def change_point_density(obj, level, render_level):
    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=True, obdata=True, material=False, animation=False)
    subdivide = obj.modifiers.new(name='subdivide', type='SUBSURF')
    subdivide.subdivision_type = 'SIMPLE'
    subdivide.levels = level
    subdivide.render_levels = render_level
    bpy.ops.object.modifier_apply(modifier=subdivide.name)

def change_point_color(obj, rgba):
    parts = get_children_in_scene(obj, [])
    for part in parts:
        if len(part.data.color_attributes) > 0:
            for att in part.data.color_attributes:
                part.data.color_attributes.remove(att)
        bpy.context.view_layer.objects.active = part
        bpy.ops.geometry.color_attribute_add(name='col', color=rgba)
        # bpy.ops.geometry.color_attribute_add(name='col', domain='CORNER', data_type='FLOAT_COLOR', color=rgba, color_space='sRGB')
        # color_attr = part.data.color_attributes['col']
        # for i in range(len(part.data.loops)):
        #     color_attr.data[i].color = rgba

def recusive_objs_iter(collection):
    for obj in collection.objects:
        yield obj
    for child_col in collection.children:
        for obj in recusive_objs_iter(child_col):
            yield obj

def modify_color_collection(collect_name, color_dict, level=[]):
    collect = bpy.data.collections.get(collect_name)
    if collect:
        for obj in recusive_objs_iter(collect):
            if len(level) == 2:
                change_point_density(obj, level[0], level[1])
            change_point_color(obj, color_dict[collect_name])

def export_moving_obj(obj_files, obj_poses, colors, ply_file, levels):
    bpy.ops.wm.open_mainfile(filepath='/home/sczone/lijian/3d_model/temp.blend')
    for i in range(len(obj_files)):
        obj_name = add_object_from_file(obj_files[i], obj_poses[i][3:], obj_poses[i][:3])
        obj = bpy.data.objects[obj_name]
        if len(levels[i]) == 2:
            change_point_density(obj, levels[i][0], levels[i][1])
        change_point_color(obj, colors[i])
    bpy.ops.wm.ply_export(filepath=ply_file)
        
def modify_color_people(pose_file, output_path, color_dict):
    people_files = []
    people_poses = []
    colors = []
    levels = []
    people_model = '/home/sczone/disk1/share/3d/blender_slots/elements/people/business_019.blend'
    poses = np.load(pose_file)
    for i in range(len(poses)):
        pose = poses[i]
        if len(pose) >= 13:
            people_pose = list(pose[-6:]-pose[:6])
            people_poses.append(people_pose)
            people_files.append(people_model)
            colors.append(color_dict['people'])
            ply_path = os.path.join(output_path, str(i)+'.ply')
            levels.append([])
            export_moving_obj(people_files, people_poses, colors, ply_path, levels)

def ply2pcd(ply_path, pcd_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points)
    new_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(pcd_path, new_pcd, write_ascii=False)
    os.remove(ply_path)

def main(args):
    colors = dict()
    colors['cars'] = (0, 1, 0, 1)
    colors['ground'] = (1, 0, 0, 1)
    colors['people'] = (0, 0, 1, 1)
    colors['fence'] = (1, 1, 0, 1)
    colors['two_wheels'] = (0, 1, 1, 1)
    colors['walls'] = (1, 0, 1, 1)
    colors['pillars'] = (0.25, 0.5, 0.5, 1)
    colors['limiters'] = (0.5, 0.25, 0.5, 1)
    colors['locks'] = (0.5, 0.5, 0.25, 1)
    colors['speed_bumps'] = (0.5, 0.25, 0.25, 1)
    colors['corns'] = (0.25, 0.5, 0.25, 1)
    colors['shopping_carts'] = (0.25, 0.25, 0.5, 1)
    colors['ceiling'] = (0.25, 0.25, 0.25, 1)
    colors['curbs'] = (0.25, 0.75, 0.75, 1)
    colors['hedges'] = (0.75, 0.25, 0.75, 1)
    colors['others'] = (0, 0, 0, 1)
    bpy.ops.wm.open_mainfile(filepath=args.input_blend_path)

    for obj in bpy.data.objects:
        if obj.name != 'Camera' and obj.name != 'Light' and not obj.name.startswith('Dome') and not obj.name.startswith('DOME'):
            change_point_color(obj, colors['others'])

    roads = bpy.data.collections.get('roads')
    signs = bpy.data.collections.get('ground_signs')
    slots = bpy.data.collections.get('slots')
    hdri = bpy.data.collections.get('Hdri Maker Tools')
    forbidden = bpy.data.collections.get('forbiddens')
    if roads:
        bpy.data.collections.remove(roads)
    if signs:
        bpy.data.collections.remove(signs)
    if slots:
        bpy.data.collections.remove(slots)
    if forbidden:
        bpy.data.collections.remove(forbidden)
    if hdri:
        bpy.data.collections.remove(hdri)
    
    ground = bpy.data.objects.get('spare_ground')
    if ground:
        change_point_density(ground, 5, 10)
        change_point_color(ground, colors['ground'])

    modify_color_collection('pillars', colors, [5, 10])
    modify_color_collection('walls', colors, [5, 10])
    modify_color_collection('people', colors)
    modify_color_collection('locks', colors)
    modify_color_collection('two_wheels', colors)
    modify_color_collection('limiters', colors)
    modify_color_collection('speed_bumps', colors)
    modify_color_collection('curbs', colors)
    modify_color_collection('hedges', colors)
    
    cars = bpy.data.collections.get('cars')
    if cars:
        for obj in recusive_objs_iter(cars):
            if obj.name.endswith('clean'):
                change_point_color(obj, colors['cars'])
 
    obstacles = bpy.data.collections.get('obstacles')
    if obstacles:
        for obj in recusive_objs_iter(obstacles):
            if obj.name.startswith('shopping_cart'):
                change_point_color(obj, colors['shopping_carts'])
            elif obj.name.startswith('corn'):
                change_point_color(obj, colors['corns'])
            else:
                change_point_color(obj, colors['fence'])

    ceiling = bpy.data.objects.get('ceiling')
    if ceiling:
        change_point_density(ceiling, 3, 6)
        change_point_color(ceiling, colors['ceiling'])

    if not os.path.exists(args.output_ply_path):
        os.mkdir(args.output_ply_path)
    bpy.ops.wm.ply_export(filepath=args.output_ply_path+'/scene.ply')

    modify_color_people(args.pose_file, args.output_ply_path, colors)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='export point cloud')
    parser.add_argument('--input_blend_path', help='input model path')
    parser.add_argument('--pose_file', help='ego pose file', default='')
    parser.add_argument('--output_ply_path', help='output model path')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)