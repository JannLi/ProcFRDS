import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
# apollo pose to matrix
def pose_to_matrix(pose: np.array, axis='xyz'):
    if np.ndim(pose) == 1:
        if pose.shape[0] == 6:
            euler = pose[:3]
            translation = pose[3:]
            r = R.from_euler(axis, np.array(euler), degrees=False)
            mat = r.as_matrix()
            mat = np.hstack([mat, translation.reshape((3, 1))])
        elif pose.shape[0] == 7:
            quaternion = pose[:4]
            translation = pose[4:]
            # r = R.from_quat(np.array(quaternion))
            # mat = r.as_matrix()
            mat = Quaternion(quaternion).rotation_matrix
            mat = np.hstack([mat, translation.reshape((3, 1))])
        elif pose.shape[0] == 3:
            mat = pose.reshape((3, 4))
    return mat

def mat_to_pose(mat: np.array):
    if mat.shape == (3, 4) or mat.shape == (4, 4):
        rot = R.from_matrix(mat[:3, :3]).as_euler('xyz')
        location = mat[:3, 3]
    elif mat.shape == (3, 3):
        rot = R.from_matrix(mat).as_euler('xyz')
        location = np.zeros((3,))
    return np.hstack([rot, location])

def rot_vec_to_4x4(rot_vec, axis='xyz'):
    if rot_vec.shape[0] == 3:
        r = R.from_euler(axis, np.array(rot_vec), degrees=False)
        mat = r.as_matrix()
        mat = rot_mat_to_4x4(mat)
    elif rot_vec.shape[1] == 4:
        mat = Quaternion(rot_vec).rotation_matrix
        mat = rot_mat_to_4x4(mat)
    return mat

def rot_mat_to_4x4(rot_mat_3x3):
    rot_mat_4x4 = np.eye(4)
    rot_mat_4x4[:3, :3] = rot_mat_3x3.copy()
    return rot_mat_4x4

def trans_vec_to_4x4(trans_vec):
    trans_mat_4x4 = np.eye(4)
    trans_mat_4x4[:3, 3] = trans_vec.copy()
    return trans_mat_4x4

def mat_to_4x4(mat):
    new_mat_4x4 = np.eye(4)
    row_num = mat.shape[0]
    col_num = mat.shape[1]
    new_mat_4x4[:row_num, :col_num] = mat
    return new_mat_4x4

def pose_to_4x4(pose):
    projection_mat = pose_to_matrix(pose)
    projection_mat_4x4 = mat_to_4x4(projection_mat)
    return projection_mat_4x4

def cam_mat_to_blender_mat(mat_4x4):
    new_mat = mat_to_4x4(pose_to_4x4(np.array([np.pi, 0, 0, 0, 0, 0])))
    new_mat = np.dot(mat_4x4, new_mat)
    return new_mat

# def car_cam_mat_to_cam_mat(mat_4x4):
#     new_mat = mat_to_4x4(pose_to_4x4(np.array([0, np.pi, 0, 0, 0, 0])))
#     new_mat = np.dot(mat_4x4, new_mat)
#     return new_mat

def cam_mat_to_ground_normal(cam_mat):
    return cam_mat[2, :]

def ground_normal_to_cam_mat(ground_normal):
    ground_z_axis = ground_normal[:3]
    cam_xaxis = np.array([1.0, 0.0, 0.0])
    ground_x_axis = cam_xaxis - cam_xaxis.dot(ground_z_axis) * ground_z_axis
    ground_x_axis = ground_x_axis / np.linalg.norm(ground_x_axis)
    ground_y_axis = np.cross(ground_z_axis, ground_x_axis)
    ground_y_axis = ground_y_axis / np.linalg.norm(ground_y_axis)
    c2g_rot = np.vstack([ground_x_axis, ground_y_axis, ground_z_axis])  # (3, 3)
    c2g_matrix = np.eye(4)
    c2g_matrix[:3, :3] = c2g_rot
    c2g_matrix[2, 3] = ground_normal[3]
    g2c_matrix = np.linalg.inv(c2g_matrix)
    E = g2c_matrix
    return E

def get_plane_2_road_mat(ground_normal: np.array, original_loction: np.array):
    R_mat_ground = mat_to_4x4(ground_normal_to_cam_mat(ground_normal)[:3, :3])
    R_location = original_loction[:3]
    R_mat = np.dot(rot_vec_to_4x4(R_location), R_mat_ground)
    location = np.array([original_loction[3], original_loction[4], cal_z_from_ground_normal(ground_normal, original_loction[[3, 4]])])
    mat_4x4 = np.eye(4)
    mat_4x4[:3, :3] = R_mat[:3, :3]
    mat_4x4[:3, 3] = location
    return mat_4x4
    
    

def cal_z_from_ground_normal(ground_normal: np.array, x_y: np.array):
    return -(ground_normal[0] * x_y[0] + ground_normal[1] * x_y[1] + ground_normal[3]) / ground_normal[2]