import sys
sys.path.append('../')
import bpy
import time
import os
import numpy as np
from generation.materials import new_material, copy_material
from generation.rendering import render_img_mask
from generation.objects import duplicate_link_obj


def set_marking_pts(count, color):
    material = new_material('pt', color)
    # bpy.ops.object.metaball_add(type='BALL')
    # ball = bpy.data.objects['Mball']
    # ball.dimensions = (0.1, 0.1, 0.1)
    bpy.ops.mesh.primitive_cylinder_add()
    ball = bpy.data.objects['Cylinder']
    ball.dimensions = (0.1, 0.1, 0.0025)
    ball.location[-1] = 0.00125
    copy_material(material, ball)
    dist_x = 20/count
    dist_y = 20/count
    locations = []
    rotations = []
    for i in range(count+1):
        for j in range(count+1):
            locations.append([-10+dist_x*i, -10+dist_y*j, 0.00125])
            rotations.append([0, 0, 0])
    locations.append([0, 0, 0])
    rotations.append([0, 0, 0])
    duplicate_link_obj(ball, locations, rotations)

def render_marking_pts(out_path):
    for name in bpy.data.objects.keys():
        obj = bpy.data.objects[name]
        # if not name.startswith('Cylinder'):    
        #     obj.hide_render = True
        # else:
        if name.startswith('Cylinder'):
            if abs(obj.location[0]) > 5 or abs(obj.location[0]) > 5:
                obj.dimensions[0] = 0.2
                obj.dimensions[1] = 0.2
        # if name.startswith('spare_ground') or name.startswith('ceiling'):
        #     obj.hide_render = False
    ego_poses = []
    dists = [0, 0.4, 0.8, 1.2]
    for i in dists:
        for j in dists:
            ego_poses.append([0, 0, 0, -i, j, 0])
    np.save('ego.npy', np.array(ego_poses))
    render_img_mask(out_path, 'ego.npy')


def main(args):
    color = (1, 0, 0, 1)
    for count in range(15, 16):
        bpy.ops.wm.open_mainfile(filepath=args.blend_path)
        set_marking_pts(count, color)
        # bpy.ops.wm.save_mainfile(filepath='test.blend')
        render_marking_pts(os.path.join(args.out_img_path, str(count)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='render marking pts')
    parser.add_argument('--blend_path', help='blend path')
    parser.add_argument('--out_img_path', help='rendered images')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)