import sys
sys.path.append('../')
import os
import bpy
import time
import cv2
import numpy as np
from generation.rendering import set_blender_cam
from blender_utils.blr import add_whole_object
from blender_utils.slam_utils.matrix_utils import pose_to_4x4
from blender_utils.slam_utils.cam_utils import DistiortTool
from blender_utils.blr import render_with_cycles2
from generation.background import world_bg_img

def add_light(location, rotation, light_type):
    light_data = bpy.data.lights.new(name='lightup', type=light_type)
    light_data.energy = 1.5
    light_object = bpy.data.objects.new(name='light', object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation

def render_sample_img(ego_pose, ori_img):
    h,w = ori_img.shape[:2]
    intrinsic = np.array([[506.61830619, 0, 962.24975743],
                         [0, 507.11285311, 771.52819392],
                         [0, 0, 1]])
    extrinsic = np.array([[-0.00334608, 0.30846554, -0.95122963, -1.02354848],
                         [ 0.99991868, -0.01067376, -0.00697865, -0.0041677],
                         [-0.01230587, -0.95117563, -0.30840474, 0.88502121],
                         [0, 0, 0, 1]])
    cam_mat = pose_to_4x4(ego_pose)
    # extrinsic = np.linalg.inv(extrinsic)
    extrinsic = np.dot(cam_mat, extrinsic)
    set_blender_cam(intrinsic, extrinsic, (w, h))
    combined_img = render_with_cycles2(ori_img, rendered_img_shadow_path='./test.png')
    bpy.ops.wm.save_mainfile(filepath='/home/sczone/disk1/share/temp1.blend')
    return combined_img

def main(args):
    bpy.ops.wm.open_mainfile(filepath=args.empty_blend)
    data = np.load(args.pose_file)
    car_pose = data[0]
    ego_pose = data[1]
    print(data)
    car_name = os.path.basename(args.car_file).split('.')[0]
    car_name = add_whole_object(args.car_file, car_name, pose_to_4x4(car_pose))
    hdri_path = '/home/sczone/disk1/share/3d/blender_slots/elements/hdri/roadside/cituyCity_006.hdr'
    world_bg_img([args.ori_img, hdri_path])
    ori_img = cv2.imread(args.ori_img)
    combined_img = render_sample_img(ego_pose, ori_img)
    cv2.imwrite(args.out_path, combined_img)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='random PROCFRPS')
    parser.add_argument('--empty_blend')
    parser.add_argument('--car_file')
    parser.add_argument('--pose_file')
    parser.add_argument('--ori_img')
    parser.add_argument('--out_path', help='output path')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)