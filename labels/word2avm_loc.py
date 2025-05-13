import sys
sys.path.append('../')

import numpy as np
import json
import os
import cv2
from blender_utils.slam_utils.matrix_utils import pose_to_4x4, mat_to_pose, cam_mat_to_blender_mat
from utils.parking_utils import get_json_param
from scipy.spatial.transform import Rotation as R
from blender_utils.slam_utils.cam_utils import points_3d_to_distort_2d
from joblib import dump, load
from scipy.interpolate import griddata

def trans_loc(R, T, sight_dist, pt, cam_index, trans):
    new_pt = np.dot(R, pt-T)
    if cam_index in [0, 1]:
        delta_up = abs(trans[0])
        delta_left = abs(trans[1])
    elif cam_index in [2, 3]:
        delta_up = abs[trans[1]]
        delta_left = abs[trans[0]]
    if new_pt[0] >= -sight_dist-delta_left and new_pt[0] <= sight_dist-delta_left and new_pt[1] <= sight_dist-delta_up and new_pt[1] >= 0:
        return True
    else:
        return False

def slots_in_sight(marking_pts, cam_pos, cam_index, trans):
    x, y, yaw = cam_pos[-3], cam_pos[-2], cam_pos[2]
    sight_dist = 10
    # yaw_radians = np.radians(yaw)
    R = np.array([[np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
    T = np.array([x, y])
    marking_pts = np.array(marking_pts)
    slot_index = []
    for i in range(len(marking_pts)):
        slot = marking_pts[i]
        pt_in_sight = list(map(lambda pt:trans_loc(R, T, sight_dist, pt, cam_index, trans), slot))
        if sum(pt_in_sight) >= 2:
            slot_index.append(i)
    return slot_index

def world2fisheye_location(world_locs, intrinsic, extrinsic, dist_coeff, mask):
    w_fisheye, h_fisheye = 1920, 1536
    pixel_locs = []
    for slot in world_locs:
        pixel_slot = []
        for loc in slot:
            if len(loc) == 2:
                loc = [[loc[0], loc[1], 0]]
            pt_2d = points_3d_to_distort_2d(np.array(loc), intrinsic, dist_coeff, extrinsic, cam_model='opencv_fisheye')
            pt_2d = [int(item) for item in pt_2d[0][0]]
            if pt_2d[0] < 0 or pt_2d[0] > h_fisheye-1 or pt_2d[1] < 0 or pt_2d[1] > w_fisheye-1:
                pixel_slot.append(None)
            else:
                if mask[pt_2d[0]][pt_2d[1]] == 0:
                    pixel_slot.append(pt_2d)
                else:
                    pixel_slot.append(None)
        pixel_locs.append(pixel_slot)
    return pixel_locs

def world2ego_trans(car_mat, world_loc):
    # R = car_mat[:3, :3]
    # T = car_mat[:3, 3]
    # loc_ego = np.dot(np.linalg.inv(R), (world_loc-T).T)
    # loc_ego = np.delete(loc_ego, 2, 0)
    loc_ego = np.dot(np.linalg.inv(car_mat), world_loc.T)
    loc_ego = np.delete(loc_ego, 2, 0)
    loc_ego = np.delete(loc_ego, 2, 0)
    return loc_ego.T

def ego2avm_trans(ego_locs, wheel_arc):
    W, H = 1088, 1216
    center_pix = np.array([[[W/2, H/2]]], dtype=np.float32)
    ego_locs[:,:, 0] -= wheel_arc
    ego_locs /= 0.02
    ego_locs = -ego_locs[:, :, ::-1]
    slots_pts = ego_locs + center_pix
    return slots_pts

def avm_edge_judge(slot):
    out = 0
    W, H = 1088, 1216
    for pt in slot:
        if pt[0] < 0 or pt[0] > W-1 or pt[1] < 0 or pt[1] > H-1:
            out += 1
    if out >= 2:
        return True
    return False

def world2avm_trans(world_locs, ego_pose, wheel_arc):
    avm_locs = []
    ego_slots = []
    cam_pos = pose_to_4x4(ego_pose)
    for i in range(len(world_locs)):
        slot = world_locs[i]
        slot = np.array([pt+[0, 1] for pt in slot])
        ego_slot = world2ego_trans(cam_pos, slot)
        ego_slots.append(ego_slot)
    avm_slots = ego2avm_trans(np.array(ego_slots), wheel_arc).tolist()

    for j in range(len(avm_slots)):
        if not avm_edge_judge(avm_slots[j]):
            avm_locs.append(avm_slots[j])           
    return avm_locs

def get_marking_pts(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    marking_pts = []
    for key in data:
        slot = data[key]
        marking_pts.append(slot['pts'])
    return marking_pts

def main(args):
    h_path = args.h_path
    intrinsic, dist_coeff, extrinsic, ego_pos = get_json_param(h_path, 0)
    x0 = ego_pos[-3]
    intrinsic, dist_coeff, extrinsic, ego_pos = get_json_param(h_path, 1)
    x1 = ego_pos[-3]
    wheel_arc = (x0+x1)/2
    points = np.load('./points.npy')
    values = np.load('./values.npy')

    ego_pos = np.load(args.pos_file)
    marking_pts = get_marking_pts(args.pts_file)

    for cam_pos in ego_pos:
        avm_slots = world2avm_trans(marking_pts, cam_pos, wheel_arc)
        img = cv2.imread()
        for slot in avm_slots:
            new_slot = griddata(points, values, np.array(slot), method='linear')
            for pt in new_slot:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), 3)

if __name__ == '__main__':
    h_path = '../../../24'
    intrinsic, dist_coeff, extrinsic, ego_pos = get_json_param(h_path, 0)
    x0 = ego_pos[-3]
    intrinsic, dist_coeff, extrinsic, ego_pos = get_json_param(h_path, 1)
    x1 = ego_pos[-3]
    wheel_arc = (x0+x1)/2
    print(wheel_arc)

    # model = load('./test.pkl')
    # scaler = load('./test_scaler.pkl')
    points = np.load('./points.npy')
    values = np.load('./values.npy')

    # cam_pos = np.array([0, 0, -2.2567, 6.2874, -7.6693, 0])
    cam_pos = np.array([0, 0, -np.pi/2, 7.7536, 1.2347, 0])
    marking_pts = get_marking_pts('/home/sczone/disk1/share/3d/blender_slots/code/results/3.json')
    avm_slots = world2avm_trans(marking_pts, cam_pos, wheel_arc)

    # img = cv2.imread('/home/sczone/disk1/share/3d/blender_slots/code/results/3_2/avm/avm_2023-11-18-02-02-00_233122000000.jpg')
    img = cv2.imread('test.png')
    for slot in avm_slots:
        slot = griddata(points, values, np.array(slot), method='linear')
        # slot = model.predict(scaler.transform(np.array(slot)))
        for pt in slot:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), 3)
    cv2.imwrite('test2.png', img)


    # print(marking_pts[22])
    # print(len(marking_pts))
    # cam_index = ['rear', 'front', 'left', 'right']
    # h_path = '../../24'
    # for i in range(1, 2):
    #     intrinsic, dist_coeff, extrinsic, ego_pos = get_json_param(h_path, i)
    #     cam_mat = pose_to_4x4(cam_pos)
    #     extrinsic = np.dot(cam_mat, extrinsic)
    #     extrinsic = np.linalg.inv(extrinsic)
    #     mask = cv2.imread('/home/sczone/disk1/share/3d/24/mask_'+cam_index[i]+'.png')
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    #     distort_locs = world2fisheye_location(marking_pts, intrinsic, extrinsic, dist_coeff, mask)
    #     print(distort_locs)
    #     print(len(distort_locs))




