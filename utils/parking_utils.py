import sys
sys.path.append('../')
import numpy as np
import bpy
import json
import cv2
import torch
from utils.json2json import get_json_parameters
from mathutils import Vector
from blender_utils.slam_utils.matrix_utils import pose_to_4x4
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata
from pyquaternion import Quaternion

def cal_angle(vector1, vector2):
    cos = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    angle = np.arccos(cos)
    sin = np.cross(vector1, vector2)
    if sin > 0:
        return -angle
    return angle

def get_target_loc(slot_pts, left_ratio, upper_ratio):
    pt_tmp1 = upper_ratio*np.array(slot_pts[1])+(1-upper_ratio)*np.array(slot_pts[2])
    pt_tmp2 = upper_ratio*np.array(slot_pts[0])+(1-upper_ratio)*np.array(slot_pts[3])
    pt = pt_tmp1*(1-left_ratio)+pt_tmp2*left_ratio
    return [int(pt[0]), int(pt[1])]

def cal_pixel_loc(intri, extri, xw, yw, zw):
    extri = extri[:3]
    pixel_loc = intri.dot(extri.dot(np.array([xw, yw, zw, 1]).T))
    norm_loc = pixel_loc/pixel_loc[-1]
    u, v = norm_loc[:2]
    return u, v

def cal_world_loc(intri, extri, u, v, z):
    K = np.array(intri)
    M = np.array(extri)
    p = K.dot(M[:3, :])
    matrix = np.zeros((4, 4))
    matrix[:3, :] = p
    matrix[-1, -1] = 1
    matrix = np.linalg.inv(matrix)
    z_c = (z-matrix[2][3]) / (matrix[2][0]*u+matrix[2][1]*v+matrix[2][2])
    x = matrix[0][0]*u*z_c + matrix[0][1]*v*z_c + matrix[0][2]*z_c + matrix[0][3]
    y = matrix[1][0]*u*z_c + matrix[1][1]*v*z_c + matrix[1][2]*z_c + matrix[1][3]
    return x, y

def cal_world_rot(p1, p2, intri, extri, model_name, z):
    x1, y1 = cal_world_loc(intri, extri, p1[0], p1[1], z)
    x2, y2 = cal_world_loc(intri, extri, p2[0], p2[1], z)
    vector_points = np.array([x1-x2, y1-y2])
    obj = bpy.data.objects[model_name]
    box_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    vetor_corners = np.array([box_corners[1][0]-box_corners[2][0], box_corners[1][1]-box_corners[2][1]])
    alpha = cal_angle(vector_points, vetor_corners)
    return alpha

def homo_trans_points(H, points):
    new_points = []
    for p in points:
        x = (H[0][0]*p[0]+H[0][1]*p[1]+H[0][2])/(H[2][0]*p[0]+H[2][1]*p[1]+H[2][2])
        y = (H[1][0]*p[0]+H[1][1]*p[1]+H[1][2])/(H[2][0]*p[0]+H[2][1]*p[1]+H[2][2])
        new_points.append([int(x), int(y)])
    return new_points

def get_json_param(json_path, cam_index):
    with open(json_path+'/camera_'+str(cam_index)+'.json', 'r') as f:
        data = json.load(f)
    params = get_json_parameters(data)
    intrinsic = np.array(params['intrinsic'])
    dist_coeff = np.array(params['distort'])
    ego_pos = np.array(params['sensor2ego_rotation']+params['sensor2ego_translation'])
    extrinsic = pose_to_4x4(ego_pos)
    return intrinsic, dist_coeff, extrinsic, ego_pos

def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

def quaternion2euler(quat):
    r = R.from_quat(quat)
    euler = r.as_euler('xyz', degrees=True)
    return euler

def word2image_pts(pts, intrinsic, extrinsic, dist_coeff):
    rotation = extrinsic[:3, :3]
    revc, _ = cv2.Rodrigues(rotation)
    translation = extrinsic[:3, 3]
    if len(dist_coeff) == 4:
        img_pts, _ = cv2.fisheye.projectPoints(pts, revc, np.array(translation), np.array(intrinsic), np.array(dist_coeff))
    else:
        img_pts, _ = cv2.projectPoints(pts, revc, np.array(translation), np.array(intrinsic), np.array(dist_coeff))
    return img_pts

# def get_target_slot(pred_path, layout):
#     path_data = np.load(pred_path)
#     last_loc = path_data[-1][3:5]
#     slots = layout['slots']
#     target_loc = []
#     for slot in slots:
#         slot_loc = slot['location'][:2]
#         if np.linalg.norm(np.array(last_loc)-np.array(slot_loc)) <= 1:
#             target_loc = slot_loc
#             break
#     return target_loc

def get_target_slot(pred_path):
    path_data = np.load(pred_path)
    last_loc = path_data[-1][3:5]
    slots = bpy.data.collections.get('slots')
    if not slots:
        return ''
    target_slot = ''
    mini_dist = 100
    for slot in slots.objects:
        slot_loc = slot.location[:2]
        dist = np.linalg.norm(np.array(last_loc)-np.array(slot_loc))
        if dist < mini_dist:
            mini_dist = dist
            target_slot = slot.name
    return target_slot    

def DistortPointsBatch_panorama(points2d_undistorted, intrinsics, distort):
    """
    distort points
    Args:
        points2d_undistorted torch.Tensor(): B, W*H , 2
        intrinsics torch.Tensor(): B, 3, 3
        distort torch.Tensor():  B, 5

    Returns:

    """
    # points2d_undistorted shape: N * 2
    px = points2d_undistorted[:, :, 0]
    py = points2d_undistorted[:, :, 1]

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    ux = intrinsics[:, 0, 2]
    uy = intrinsics[:, 1, 2]
    distort = distort.T

    if distort.shape[0] == 8:
        k1, k2, p1, p2, k3, k4, k5, k6 = distort
    elif distort.shape[0] == 5:
        k1, k2, p1, p2, k3 = distort
        k4, k5, k6 = torch.zeros_like(fx), torch.zeros_like(fx), torch.zeros_like(fx)
    else:
        print('distort error')

    fx = fx.repeat(px.shape[-1], 1).T
    fy = fy.repeat(px.shape[-1], 1).T
    ux = ux.repeat(px.shape[-1], 1).T
    uy = uy.repeat(px.shape[-1], 1).T

    k1 = k1.repeat(px.shape[-1], 1).T
    k2 = k2.repeat(px.shape[-1], 1).T
    p1 = p1.repeat(px.shape[-1], 1).T
    p2 = p2.repeat(px.shape[-1], 1).T
    k3 = k3.repeat(px.shape[-1], 1).T
    k4 = k4.repeat(px.shape[-1], 1).T
    k5 = k5.repeat(px.shape[-1], 1).T
    k6 = k6.repeat(px.shape[-1], 1).T

    xCorrected = (px - ux) / fx
    yCorrected = (py - uy) / fy

    r2 = xCorrected * xCorrected + yCorrected * yCorrected

    deltaRa = 1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    deltaRb = 1 / (1. + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2)
    deltaTx = 2. * p1 * xCorrected * yCorrected + p2 * (r2 + 2. * xCorrected * xCorrected)
    deltaTy = p1 * (r2 + 2. * yCorrected * yCorrected) + 2. * p2 * xCorrected * yCorrected

    xDistortion = xCorrected * deltaRa * deltaRb + deltaTx
    yDistortion = yCorrected * deltaRa * deltaRb + deltaTy

    xDistortion = xDistortion * fx + ux
    yDistortion = yDistortion * fy + uy

    points2d_distort = torch.stack([xDistortion, yDistortion], dim=2)
    return points2d_distort

def add_distortion(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    
    # 创建映射表
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    y_coords, x_coords = np.indices((h, w), dtype=np.float32)

    x_normalized = (x_coords - cx) / fx
    y_normalized = (y_coords - cy) / fy
    
    # 计算 r^2
    r2 = x_normalized**2 + y_normalized**2
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r6 * r2
    r10 = r8 * r2
    r12 = r10 * r2
            
    # 应用畸变公式
    radial_distortion = 1 + dist_coeffs[0]*r2 + dist_coeffs[1]*r4 + dist_coeffs[4]*r6 + dist_coeffs[5]*r8 + dist_coeffs[6]*r10 + dist_coeffs[7]*r12
    x_distorted = x_normalized * radial_distortion + 2*dist_coeffs[2]*x_normalized*y_normalized + dist_coeffs[3]*(r2 + 2*x_normalized**2)
    y_distorted = y_normalized * radial_distortion + dist_coeffs[2]*(r2 + 2*y_normalized**2) + 2*dist_coeffs[3]*x_normalized*y_normalized
    
    # 反归一化坐标
    map_x = x_distorted *fx + cx
    map_y = y_distorted * fy + cy

    # 使用 remap 进行重映射
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return distorted_image

if __name__ == '__main__':
    json_path = '/home/sczone/disk1/share/3d/24'
    intrinsic, dist_coeff, extrinsic, ego_pos = get_json_param(json_path, 'front120')
    # poses = np.load('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/7/poses.npy')
    # cam2ego = np.eye(4)
    # cam2ego[:3, :3] = Quaternion(ego_pos[:4]).rotation_matrix
    # cam2ego[:3, 3] = ego_pos[4:]
    # ego2cam = np.linalg.inv(cam2ego)

    # e2g = np.eye(4)
    # e2g[:3, :3] = R.from_rotvec(poses[0][:3]).as_matrix()
    # e2g[:3, 3] = poses[0][3:]
    # g2e = np.linalg.inv(e2g)
    
    # pts = np.array([[-9.42, 6.5, 0, 1]], dtype=np.float32)
    # pts = pts.reshape(pts.shape[::-1])
    # pts = g2e.dot(pts)[:3, :]
    # pts = pts.transpose(1, 0)[None, :]
    # print(word2image_pts(pts, intrinsic, ego2cam, dist_coeff))

    img = cv2.imread('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/results/3/results/front_120_17_1.png')
    h, w = img.shape[:2]
    x, y = np.arange(w), np.arange(h)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    coords = torch.tensor(coords[np.newaxis, :, :])
    # coords = torch.tensor([[[1997, 1229]]])
    # intrinsic, _ = cv2.getOptimalNewCameraMatrix(intrinsic, dist_coeff, (w, h), 0, (w, h))
    intrinsic = torch.tensor(intrinsic[np.newaxis, :, :])
    dist_coeff = torch.tensor(dist_coeff[np.newaxis, :])
    new_coords = DistortPointsBatch_panorama(coords, intrinsic, dist_coeff[:5])

    new_coords = np.array(new_coords[0])
    new_coords = new_coords.reshape(h, w, 2)
    map1 = np.zeros((h, w))
    map2 = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            map1[int(new_coords[i][j][1])][int(new_coords[i][j][0])] = i
            map2[int(new_coords[i][j][1])][int(new_coords[i][j][0])] = j
    map1 = np.array(map1, dtype=np.float32)
    map2 = np.array(map2, dtype=np.float32)
    np.save('/home/sczone/disk1/share/3d/24/front_120_map1.npy', map2)
    np.save('/home/sczone/disk1/share/3d/24/front_120_map2.npy', map1)
    
    # # map1 = np.array(new_coords[0][:, 0], dtype=np.float32)
    # # map1 = map1.reshape(h, w)
    # # map2 = np.array(new_coords[0][:, 1], dtype=np.float32)
    # # map2 = map2.reshape(h, w)
    # img2 = cv2.remap(img, map2, map1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # cv2.imwrite('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/results/3/results/front_120_17_distort1.png', img2)
    # img2 = img2[1620:-1620, 2880:-2880]
    # cv2.imwrite('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/results/3/results/front_120_17_distort2.png', img2)

    # img2 = cv2.imread('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/results/3/results/front_120_17_1_distort1.png')
    # map12, map22 = cv2.initUndistortRectifyMap(intrinsic, dist_coeff, np.eye(3, 3), intrinsic, (9600, 5400), cv2.CV_16SC2)
    # img3 = cv2.remap(img2, map12, map22, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # img3 = img3[1620:-1620, 2880:-2880]
    # cv2.imwrite('/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/results/3/results/front_120_17_distort4.png', img3)

