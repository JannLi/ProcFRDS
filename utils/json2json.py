import sys
sys.path.append('../')
import argparse
import os
import json
import math

import numpy as np
from pyquaternion import Quaternion
from blender_utils.slam_utils.matrix_utils import pose_to_4x4


custom_dict = {}


def get_rotate_yaw(angle):
    return np.array(
        [
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=float
    )


def get_rotate_pitch(angle):
    return np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ],
        dtype=float
    )


def get_rotate_roll(angle):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ],
        dtype=float
    )


def get_rotate(yaw, pitch, roll):
    rz = get_rotate_yaw(yaw)
    ry = get_rotate_pitch(pitch)
    rx = get_rotate_roll(roll)

    return np.dot(rz, np.dot(ry, rx))



def get_intrinsics(calibration):
    """
    通过标定文件得到 内参矩阵 3 x 3
    @param calibration: 单个相机标定参数
    @return: 单个相机的cam2img的相机内参和畸变系数
    """
    intrinsics = np.array([calibration['fx'], 0, calibration['cx'], 0, calibration['fy'], calibration['cy'], 0, 0, 1])
    intrinsics = np.reshape(intrinsics, (3, 3))

    return intrinsics


def get_extrinsics(calibration):
    """
    将坐标点由世界坐标系转到相机坐标
    @param calibration: 标定文件，csv 包含相机外参，内参
    @return:
    """

    # 先平移后旋转， 旋转过程，先绕z轴转，y轴-> x轴
    # 平移变量除了原始的x值变换，还需要由后轴到前轴的变话，wheelbase=2.8
    c = math.cos(calibration['yaw'])
    s = math.sin(calibration['yaw'])
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)

    c = math.cos(calibration['pitch'])
    s = math.sin(calibration['pitch'])
    Ry = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)

    c = math.cos(calibration['roll'])
    s = math.sin(calibration['roll'])
    Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)

    rotation = Rx.dot(Ry).dot(Rz)

    trans = np.array([[- calibration['camPosX'], - calibration['camPosY'], - calibration['camPosZ']]])

    # 车体坐标系xyz方向为：x-> left， y->front z->top， 转到相机坐标系 x->right, y->down, z->front
    # 转换坐标系方向，cr为此变换矩阵
    cr = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    ego2sensor_rotation = cr.dot(rotation)
    ego2sensor_trans = ego2sensor_rotation.dot(trans.T)

    r = np.eye(4)
    r[:3, :3] = ego2sensor_rotation
    r[:3, 3] = ego2sensor_trans.squeeze(1)
    rr = np.linalg.inv(r)
    sensor2egorotation = rr[:3, :3]
    sensor2egotranslation = rr[:3, 3]

    return sensor2egorotation, sensor2egotranslation


def convertRMatrix2EulerAngle(R):
    pitch = -math.asin(R[0, 2])
    yaw = math.atan2(R[0, 1], R[0, 0])
    roll = math.atan2(R[1, 2], R[2, 2])

    return roll, pitch, yaw


def convertEulerAngle2RMatrix(roll, pitch, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)

    c = math.cos(pitch)
    s = math.sin(pitch)
    Ry = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)

    c = math.cos(roll)
    s = math.sin(roll)
    Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)

    return Rx, Ry, Rz


def get_position(contents):
    # local2camera position
    posx_local2cam = contents["camera_x"]
    posy_local2cam = contents["camera_y"]
    posz_local2cam = contents["camera_z"]

    # ego2local position
    posx_veh2local = contents["vcs"]["translation"][0]
    posy_veh2local = contents["vcs"]["translation"][1]
    posz_veh2local = contents["vcs"]["translation"][2]

    x = posx_local2cam + posx_veh2local  # + 2.95
    y = posy_local2cam + posy_veh2local
    z = posz_local2cam + posz_veh2local  # - 1

    return x, y, z


def get_yaw_pitch_roll(contents):
    # local2camera yaw-pitch-roll
    yaw_local2cam = contents["yaw"]
    pitch_local2cam = contents["pitch"]
    roll_local2cam = contents["roll"]

    # ego2local yaw-pitch-roll
    yaw_veh2local = contents["vcs"]["rotation"][2]
    pitch_veh2local = contents["vcs"]["rotation"][1]
    roll_veh2local = contents["vcs"]["rotation"][0]

    RX_VehToLocal, RY_VehToLocal, RZ_VehToLocal = convertEulerAngle2RMatrix(roll_veh2local, pitch_veh2local, yaw_veh2local)
    RX_LocalToCam, RY_LocalToCam, RZ_LocalToCam = convertEulerAngle2RMatrix(roll_local2cam, pitch_local2cam, yaw_local2cam)

    R_VehToLocal = RZ_VehToLocal.dot(RY_VehToLocal).dot(RX_VehToLocal)
    R_LocalToCam = RZ_LocalToCam.dot(RY_LocalToCam).dot(RX_LocalToCam)

    R_VehToCam = R_LocalToCam.dot(R_VehToLocal)

    roll, pitch, yaw = convertRMatrix2EulerAngle(R_VehToCam)

    return roll, pitch, yaw


def get_json_parameters(contents):
    json_dict = {}
    x, y, z = get_position(contents)
    roll, pitch, yaw = get_yaw_pitch_roll(contents)   

    json_dict = {}
    json_dict["camPosX"] = x
    json_dict["camPosY"] = y
    json_dict["camPosZ"] = z  # + contents["image_height"] / 2
    json_dict["yaw"] = yaw
    json_dict["pitch"] = pitch
    json_dict["roll"] = roll

    json_dict["fx"] = contents["focal_u"]
    json_dict["fy"] = contents["focal_v"]
    json_dict["cx"] = contents["center_u"]
    json_dict["cy"] = contents["center_v"]

    json_dict["height"] = contents["image_height"]
    json_dict["width"] = contents["image_width"]

    json_dict["k1"] = contents["distort"][0]
    json_dict["k2"] = contents["distort"][1]
    json_dict["k3"] = contents["distort"][2]
    json_dict["k4"] = contents["distort"][3]

    json_dict["distort"] = contents["distort"]

    json_dict["sensorFOV"] = contents["fov"]

    intrinsic = get_intrinsics(json_dict)
    rotation, trans = get_extrinsics(json_dict)

    json_dict["intrinsic"] = intrinsic.tolist()
    json_dict["sensor2ego_rotation"] = list(Quaternion(matrix=rotation))
    json_dict["sensor2ego_translation"] = trans.tolist()

    return json_dict

if __name__ == '__main__':
    with open('/home/sczone/disk1/share/3d/24/camera_0.json', 'r') as f:
        data = json.load(f)
    params = get_json_parameters(data)
    intrinsic = np.array(params['intrinsic'])
    dist_coeff = np.array(params['distort'])
    extrinsic = pose_to_4x4(np.array(params['sensor2ego_rotation']+params['sensor2ego_translation']))
    print(extrinsic)
    print(intrinsic)
    print(data['vcs']['rotation'])
    print(data['vcs']['translation'])