import sys
sys.path.append('../')
import os
import bpy
import time
import json
import random
import cv2
import math
import numpy as np
from generation.rendering import render_img_mask
from generation.background import hdri_maker_bg_dome
from generation.materials import add_puddle2
from generation.objects import add_object_from_file

def set_background(hdri_path):
    hdri_name = random.choice([os.path.basename(name) for name in os.listdir(hdri_path)])
    # hdri_name = 'City_006'
    print(hdri_name)
    hdri_maker_bg_dome(hdri_path, hdri_name.split('.')[0], 15)
    bpy.data.objects['Dome Handler'].constraints['Limit Location'].enabled = False
    bpy.data.objects['Dome Handler'].location[-1] = -0.00001

def lane_marker_noise(noise_value, trans_value):
    bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group.017"].inputs[1].default_value = noise_value
    bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group.012"].inputs[1].default_value = noise_value
    bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group.013"].inputs[1].default_value = noise_value

    # 透明度
    bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Math.019"].inputs[1].default_value = trans_value

def set_puddle(mat_name, value):
    add_puddle2(bpy.data.materials[mat_name], value)

def generate_bev_mask(bev_img):
# 确保图像是二值化的
    road_binary = cv2.inRange(bev_img, lowerb=np.array([155, 155, 155]), upperb=np.array([255, 255, 255]))
    centerline_binary = cv2.inRange(bev_img, lowerb=np.array([150, 0, 0]), upperb=np.array([255, 100, 100]))
    centerline_binary = 255-centerline_binary
    dist = cv2.distanceTransform(src=centerline_binary, distanceType=cv2.DIST_L2, maskSize=3)
    dist1 = cv2.convertScaleAbs(dist)
    # dist2 = cv2.normalize(dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    grad_x = cv2.Sobel(dist1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dist1, cv2.CV_64F, 0, 1, ksize=3)
    return road_binary, grad_x, grad_y

def modify_lines(line1, line2, middle_line):
    if line1 == 'd':
        bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group.021"].inputs[2].default_value = 3
    elif line1 == 's':
        bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group.021"].inputs[2].default_value = 0
    if line2 == 'd':
        bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group.020"].inputs[2].default_value = 3
    elif line2 == 's':
        bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group.020"].inputs[2].default_value = 0
    if middle_line == 'dd':
        bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group"].inputs[2].default_value = 3
    elif middle_line == 'ss':
        bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group"].inputs[2].default_value = 0
    elif middle_line == 'ds':
        bpy.data.materials["Next_Asphalt_Realistic"].node_tree.nodes["Group"].inputs[2].default_value = 3
        bpy.data.node_groups["NodeGroup.013"].nodes["Map Range"].inputs[4].default_value = 100000

def cal_ego_pose(pic_loc, grad_x, grad_y):
    x = (pic_loc[0]-1920/2)/48*10
    y = -(pic_loc[1]-1080/2)/48*10
    world_loc = (x, y)
    dx = grad_x[pic_loc[1], pic_loc[0]]
    dy = grad_y[pic_loc[1], pic_loc[0]]
    world_rot = math.atan2(dx, dy)
    return world_loc, world_rot

def set_ego_poses(bev_img, ego_pose_path, ego_num):
    road_binary, grad_x, grad_y = generate_bev_mask(bev_img)
    road_pixels = np.column_stack(np.where(road_binary==255))
    selected_indices = np.random.choice(len(road_pixels), size=ego_num, replace=False)
    selected_pixels = road_pixels[selected_indices]
    ego_poses = []
    for pixel in selected_pixels:
        pixel = pixel[::-1]
        world_loc, world_rot = cal_ego_pose(pixel, grad_x, grad_y)
        if world_loc:
            ego_poses.append([0, 0, world_rot, world_loc[0], world_loc[1], 0])
    np.save(ego_pose_path, np.array(ego_poses))

def main(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    bpy.ops.wm.open_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/lane_marker/blender-city-nextstreet1_2.blend')
    lane_marker_noise(0, 2)
    line_type = args.line_type
    assert len(line_type) == 4
    modify_lines(line_type[0], line_type[1], line_type[2:])
    set_background(args.hdri_path)
    bev_img = cv2.imread('/home/sczone/disk1/share/3d/blender_slots/lane_marker/untitled1.png')
    if not os.path.exists(args.ego_pose_path):
        set_ego_poses(bev_img, args.ego_pose_path, 200)

    render_img_mask(os.path.join(args.out_path, 'images'), args.ego_pose_path, 'front_cone')
    bpy.ops.wm.open_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/lane_marker/traffic_light.blend')
    render_img_mask(os.path.join(args.out_path, 'masks1'), args.ego_pose_path, 'front')
    bpy.ops.wm.open_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/lane_marker/crosswalk.blend')
    render_img_mask(os.path.join(args.out_path, 'masks2'), args.ego_pose_path, 'front')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='random PROCFRPS')
    parser.add_argument('--hdri_path', help='hdri file', default='/home/sczone/disk1/share/3d/blender_slots/elements/hdri/roadside')
    parser.add_argument('--ego_pose_path', help='ego pose file')
    parser.add_argument('--line_type', default='dsss')
    parser.add_argument('--out_path', help='output path')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)
