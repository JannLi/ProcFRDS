import numpy as np
import bpy
import os
import random
import cv2
import json
import shutil
from blender_utils.slam_utils.matrix_utils import pose_to_4x4, cam_mat_to_blender_mat
from blender_utils.blr import render_with_fisheye, render_with_cycles2, full_render_with_fisheye
from blender_utils.slam_utils.cam_utils import DistiortTool
from utils.parking_utils import get_json_param
from generation.materials import change_avmseg_material
from generation.objects import add_object_from_file, add_cone

def cal_marking_pts(slot, slot_angle, line_width=0.21):
    center = np.array(slot.location[:2])
    angle = slot.rotation_euler[2]
    h = slot.dimensions[1]-2*line_width
    w = slot.dimensions[0]-2*(line_width/np.tan(slot_angle)+line_width/np.sin(slot_angle))
    angle1 = np.array([np.cos(angle), -np.sin(angle)])
    angle2 = np.array([np.sin(angle), np.cos(angle)])
    delta = h/np.tan(slot_angle)*2
    pt1 = np.array([-w+delta, h])/2
    pt2 = np.array([-w, -h])/2
    pt3 = np.array([w-delta, -h])/2
    pt4 = np.array([w, h])/2
    pt1 = center + np.array([np.dot(pt1, angle1), np.dot(pt1, angle2)])
    pt2 = center + np.array([np.dot(pt2, angle1), np.dot(pt2, angle2)])
    pt3 = center + np.array([np.dot(pt3, angle1), np.dot(pt3, angle2)])
    pt4 = center + np.array([np.dot(pt4, angle1), np.dot(pt4, angle2)])
    return [pt1.tolist(), pt4.tolist(), pt3.tolist(), pt2.tolist()]

def dump_marking_pts(out_json_path, occupied_slots, slot_angle):
    car_slots, lock_slots, obstacle_slots, forbidden_slots = occupied_slots
    other_obstacle_slots, corn_slots, no_parking_slots = obstacle_slots
    slots_info = dict()
    for slot in bpy.data.collections['slots'].objects:
        slot.location[2] = 0
        slots_info[slot.name] = dict()
        slots_info[slot.name]['pts'] = cal_marking_pts(slot, slot_angle)
        slots_info[slot.name]['locked'] = 0
        slots_info[slot.name]['occupied'] = 0
        slots_info[slot.name]['forbidden'] = 0
        if slot.name in corn_slots:
            slots_info[slot.name]['locked'] = 1
        elif slot.name in lock_slots:
            slots_info[slot.name]['locked'] = 2
        elif slot.name in no_parking_slots:
            slots_info[slot.name]['locked'] = 3
        if slot.name in car_slots:
            slots_info[slot.name]['occupied'] = 1
        elif slot.name in other_obstacle_slots:
            slots_info[slot.name]['occupied'] = 2
        if slot.name in forbidden_slots:
            slots_info[slot.name]['forbidden'] = 1
    with open(out_json_path, 'w') as f:
        f.write(json.dumps(slots_info))

def catch_up_rendering(res_path):
    names = os.listdir(res_path)
    fish_count = 0
    seg_count = 0
    if len(names) <= 4:
        return fish_count, seg_count
    fisheye_names = [name for name in names if name.startswith('fisheye')]
    seg_names = [name for name in names if name.startswith('seg')]
    if len(seg_names) > 0:
        seg_names = sorted(seg_names, key=lambda s: int(''.join(filter(str.isdigit, s))))
        fish_count = int(len(fisheye_names)/4-1)
        seg_count = int(seg_names[-1].split('_')[1])
    elif len(fisheye_names) > 0:
        fisheye_names = sorted(fisheye_names, key=lambda s: int(''.join(filter(str.isdigit, s))))
        fish_count = int(fisheye_names[-1].split('_')[1])
        seg_count = 0
    return fish_count, seg_count

def set_cam_params(cam_param_index, distort_tool, img_size=(1920, 1536)):
    w_fisheye, h_fisheye = img_size
    # json_path = '/home/sczone/disk1/share/3d/24'
    json_path = '/home/sczone/disk1/share/3d/AH4EM'
    intrinsic, dist_coeff, extrinsic, ego_pos = get_json_param(json_path, cam_param_index)
    fov = np.pi
    if cam_param_index == 'front120':
        fov = 2*np.pi/3
    else:
        distort_tool.update_cam_info(distort_coeff=dist_coeff, distort_K=intrinsic)
        distort_tool.update_fisheye_blr_info(img_size=np.array([w_fisheye, h_fisheye]), fov=fov)
        distort_tool.update_map_fisheye2distort()
    return intrinsic, dist_coeff, extrinsic, distort_tool

def set_camera(sensor_width, focal_length):
    cam = bpy.data.objects['Camera']
    cam.location.x = 0
    cam.location.y = 0
    cam.location.z = 0
    cam.rotation_euler = [0, 0, 0]
    cam.data.sensor_width = sensor_width
    cam.data.lens = focal_length

def cam_intr_2_lens(img_size, camera_intrinsic, sensor_width):
    focal_length = (sensor_width / img_size[0]) * camera_intrinsic[0][0]
    return focal_length

def set_blender_cam(camera_intrinsic, camera_extrinsic, img_size):
    sensor_width = 32
    camera_intrinsic[0][-1] = img_size[0]/2
    camera_intrinsic[1][-1] = img_size[1]/2
    focal_length = cam_intr_2_lens(img_size, camera_intrinsic, sensor_width)
    set_camera(sensor_width, focal_length)
    # camera_extrinsic = np.linalg.inv(camera_extrinsic)
    cam = bpy.data.objects['Camera']
    cam.matrix_world = cam_mat_to_blender_mat(camera_extrinsic).T

def set_cam_render(cam_params, cam_pose, distort_tool, combined_img_path, fisheye_size=(1536, 1920), cam_model='opencv_fisheye', target='norm', ori_img=''):
    h_fisheye, w_fisheye = fisheye_size
    intrinsic, dist_coeff, extrinsic = cam_params
    cam_mat = pose_to_4x4(cam_pose)
    # extrinsic = np.linalg.inv(extrinsic)
    extrinsic = np.dot(cam_mat, extrinsic)
    set_blender_cam(intrinsic, extrinsic, (w_fisheye, h_fisheye))

    distort_tool.update_cam_info(distort_coeff=dist_coeff, distort_K=intrinsic, cam_extri=extrinsic, cam_model=cam_model)
    ori_img = np.zeros((h_fisheye, w_fisheye, 3), dtype=np.uint8)
    fisheye_info = dict()
    mask1 = os.path.join(os.path.dirname(combined_img_path), '2.png')
    mask2 = os.path.join(os.path.dirname(combined_img_path), '3.png')
    if cam_model == 'opencv_fisheye':
        # combined_img, _ = render_with_fisheye(ori_img, fisheye_info, distort_tool, rendered_img_shadow_path=mask1, rendered_img_no_shadow_path=mask2, target=target, mask_only=False)
        combined_img = full_render_with_fisheye(ori_img, distort_tool, rendered_img_shadow_path=mask1, target=target)
    else:
        combined_img = render_with_cycles2(ori_img, rendered_img_shadow_path=mask1)
        map1 = np.load('/home/sczone/disk1/share/3d/24/front_120_map1.npy')
        map2 = np.load('/home/sczone/disk1/share/3d/24/front_120_map2.npy')
        combined_img = cv2.remap(combined_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        combined_img = combined_img[1620:-1620, 2880:-2880]
    cv2.imwrite(combined_img_path, combined_img)

def render_img_mask(out_img_path, out_pose_path, img_type='fisheye'):
    camera = bpy.data.objects.get('Camera')
    if camera:
        bpy.context.view_layer.objects.active = camera
        camera.select_set(True)

    cam_params = []
    for i in range(4):
        distort_tool = DistiortTool()
        intrinsic, dist_coeff, extrinsic, distort_tool = set_cam_params(i, distort_tool)
        cam_params.append([intrinsic, dist_coeff, extrinsic, distort_tool])
    distort_tool = DistiortTool()
    intrinsic, dist_coeff, extrinsic, distort_tool = set_cam_params('front120', distort_tool, img_size=(9600, 5400))
    cam_params.append([intrinsic, dist_coeff, extrinsic, distort_tool])    
    
    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)
    fish_count, seg_count = catch_up_rendering(out_img_path)

    cam_index = ['rear', 'front', 'left', 'right']
    ego_poses = np.load(out_pose_path)

    bpy.context.scene.camera = bpy.data.objects['Camera']

    if len(ego_poses[0]) >= 13:
        people_name = add_object_from_file('/home/sczone/disk1/share/3d/blender_slots/elements/people/business_019.blend', (0, 0, 0), (0, 0, 0), 'people')

    if fish_count >= 0:
        for j in range(fish_count, len(ego_poses)):
            if len(ego_poses[j]) >= 13:
                people_pose = list(ego_poses[j][-6:]-ego_poses[j][:6])
                people = bpy.data.objects[people_name]
                people.location = people_pose[3:]
                people.rotation_euler = people_pose[:3]
            cam_pose = ego_poses[j][:6]
            if j > 0 and np.equal(cam_pose, ego_poses[j-1][:6]).all():
                for i in range(4):
                    pre_frame = os.path.join(out_img_path, 'fisheye_'+str(j-1)+'_'+cam_index[i]+'.png')
                    cur_frame = os.path.join(out_img_path, 'fisheye_'+str(j)+'_'+cam_index[i]+'.png')
                    shutil.copy2(pre_frame, cur_frame)
                pre_front = os.path.join(out_img_path, 'front_120_'+str(j-1)+'.png')
                cur_front = os.path.join(out_img_path, 'front_120_'+str(j)+'.png')
                shutil.copy2(pre_front, cur_front)
                continue
            if img_type == 'fisheye':
                for i in range(4):
                    combined_img_path = os.path.join(out_img_path, 'fisheye_'+str(j)+'_'+cam_index[i]+'.png')
                    set_cam_render(cam_params[i][:3], cam_pose, cam_params[i][3], combined_img_path)
            elif img_type == 'front':
                combined_img_path = os.path.join(out_img_path, 'front_120_'+str(j)+'.png')
                set_cam_render(cam_params[4][:3], cam_pose, cam_params[4][3], combined_img_path, fisheye_size=(5400, 9600), cam_model='pinhole')
            elif img_type == 'front_cone':
                cone_dires = add_cone(cam_pose)
                with open(os.path.join(out_img_path, 'front_120_'+str(j)+'.txt'), 'w') as f:
                    f.write(','.join(cone_dires))
                combined_img_path = os.path.join(out_img_path, 'front_120_'+str(j)+'.png')
                set_cam_render(cam_params[4][:3], cam_pose, cam_params[4][3], combined_img_path, fisheye_size=(5400, 9600), cam_model='pinhole')

    # change_avmseg_material()

    # for j in range(seg_count, len(ego_poses)):
    #     cam_pose = ego_poses[j]
    #     for i in range(4):    
    #         seg_mask_path = os.path.join(out_img_path, 'seg_'+str(j)+'_'+cam_index[i]+'.png')
    #         set_cam_render(cam_params[i][:3], cam_pose, cam_params[i][3], seg_mask_path, target='seg')