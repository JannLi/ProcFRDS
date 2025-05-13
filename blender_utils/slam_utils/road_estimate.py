import numpy as np
from .projection_utils import points_2d_project
from typing import Union
from sklearn.linear_model import RANSACRegressor
import open3d as o3d
from .matrix_utils import get_plane_2_road_mat, pose_to_4x4

def read_bin(bin_path, intensity=False):
    """
    读取kitti bin格式文件点云
    :param bin_path:   点云路径
    :param intensity:  是否要强度
    :return:           numpy.ndarray `N x 3` or `N x 4`
    """
    try:
        lidar_points = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    except:
        lidar_points = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 5))[..., :4]
    if not intensity:
        lidar_points = lidar_points[:, :3]
    return lidar_points

def read_pcd(pcd_file_path, intensity=False):
    """
    读取pcd文件点云
    :param pcd_file_path:   点云路径
    :param intensity:  是否要强度
    :return:           numpy.ndarray `N x 3` or `N x 4`
    """
    pcd = o3d.t.io.read_point_cloud(filename = pcd_file_path)
    positions = pcd.point.positions.numpy()
    intensities = pcd.point.intensity.numpy()
    if intensity:
        return np.hstack((positions, intensities)).astype(np.float64)
    else:
        return positions.astype(np.float64)


def fit_ground_equation(points: np.array):
    road_ransac = RANSACRegressor()
    road_ransac.fit(points[:, :2], points[:, 2])
    # ax + by + c = z
    return road_ransac.estimator_.coef_[0], road_ransac.estimator_.coef_[1], -1, road_ransac.estimator_.intercept_    
    
def fit_ground_normal(points: np.array):
    road_ransac = RANSACRegressor( residual_threshold=0.3)
    # road_ransac = RANSACRegressor()
    road_ransac.fit(points[:, :2], points[:, 2])
    # ax + by + c = z
    return np.array([-road_ransac.estimator_.coef_[0], -road_ransac.estimator_.coef_[1], 1.0, -road_ransac.estimator_.intercept_])

def get_road_points_3d_from_lidar_multi_cam(road_mask_list: list, lidar_path:Union[str, np.array], extri: list, intri: list):
    if isinstance(lidar_path, str):
        if lidar_path.endswith('.bin'):
            lidar_points = read_bin(lidar_path)
        elif lidar_path.endswith('.pcd'):
            lidar_points = read_pcd(lidar_path)
    else:
        lidar_points = lidar_path
    lidar_mask_road = np.zeros(len(lidar_points), dtype=bool)
    for i in range(len(road_mask_list)):
        _, lidar_mask = get_road_points_3d_from_lidar_signle_cam(road_mask_list[i], lidar_points, extri[i], intri[i])
        lidar_mask_road = np.logical_or(lidar_mask_road, lidar_mask)
    return lidar_points[lidar_mask_road], lidar_mask_road
    
    
def get_road_points_3d_from_lidar_signle_cam(road_mask: np.array, lidar_path:Union[str, np.array], extri:np.array, intri:np.array):
    if isinstance(lidar_path, str):
        if lidar_path.endswith('.bin'):
            lidar_points = read_bin(lidar_path)
        elif lidar_path.endswith('.pcd'):
            lidar_points = read_pcd(lidar_path)
    else:
        lidar_points = lidar_path
    ego2lidar_mat = np.linalg.inv(extri)
    lidar_points_in_img, lidar_points_in_img_3d = points_2d_project(lidar_points, ego2lidar_mat, intri)
    lidar_points_in_img = np.round(lidar_points_in_img).astype(np.int32)
    front_bound = lidar_points_in_img_3d[:, 2] > 0
    u_bound = np.logical_and(lidar_points_in_img[:, 0] >= 0, lidar_points_in_img[:, 0] < road_mask.shape[1])
    v_bound = np.logical_and(lidar_points_in_img[:, 1] >= 0, lidar_points_in_img[:, 1] < road_mask.shape[0])
    uv_bound = np.logical_and(u_bound, v_bound)
    lidar_mask = np.logical_and(front_bound, uv_bound)
    lidar_points_road_mask = road_mask[lidar_points_in_img[lidar_mask][:, 1], lidar_points_in_img[lidar_mask][:, 0]].astype(bool)
    i = 0
    for j in range(len(lidar_mask)):
        if lidar_mask[j]:
            lidar_mask[j] = lidar_points_road_mask[i]
            i += 1
    return lidar_points[lidar_mask], lidar_mask

def pose_on_the_road(location_xy, yaw, road_points, range_xy: np.array = np.array([[-2.5, 2.5], [-2.5, 2.5]])):
    if road_points is None:
        return pose_to_4x4(np.array([0, 0, yaw, location_xy[0], location_xy[1], 0]))
    points_range = np.array([[location_xy[0]+range_xy[0, 0], location_xy[0]+range_xy[0, 1]], [location_xy[1]+range_xy[1, 0], location_xy[1]+range_xy[1, 1]]])
    road_points_under_car = road_points[np.logical_and(np.logical_and(road_points[:, 0] > points_range[0, 0], road_points[:, 0] < points_range[0, 1]), np.logical_and(road_points[:, 1] > points_range[1, 0], road_points[:, 1] < points_range[1, 1]))]
    try:
        road_normal = fit_ground_normal(road_points_under_car)
        final_pose = get_plane_2_road_mat(road_normal, np.array([0, 0, yaw, location_xy[0], location_xy[1], 0]))
    except:
        final_pose = pose_to_4x4(np.array([0, 0, yaw, location_xy[0], location_xy[1], 0]))
    return final_pose