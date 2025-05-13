import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import cv2
from .matrix_utils import rot_mat_to_4x4, trans_vec_to_4x4, mat_to_4x4, cam_mat_to_blender_mat


def ePNP(points_3d: np.array, points_2d: np.array, camera_inrinsic:np.array):
    dist_coeffs = np.array([0, 0, 0, 0, 0]).astype(float)
    # retval, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, camera_inrinsic,
    #                                   dist_coeffs, reprojectionError=5.0, iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE)
    retval, rvec, tvec= cv2.solvePnP(points_3d, points_2d, camera_inrinsic,
                                      None, flags=cv2.SOLVEPNP_ITERATIVE)
    caled_rvec = rvec[:, 0]
    caled_tvec = tvec[:, 0]
    caled_rot_mat, jac = cv2.Rodrigues(caled_rvec)
    caled_rot_mat_4x4 = rot_mat_to_4x4(caled_rot_mat)
    caled_trans_mat_4x4 = trans_vec_to_4x4(caled_tvec)
    caled_cam_extr_mat = np.dot(caled_trans_mat_4x4, caled_rot_mat_4x4)
    return caled_cam_extr_mat

def points_3d_transmit(points_3d, transmit_mat):
    points_3d_homo = np.ones((points_3d.shape[0], 4))
    points_3d_homo[:, :3] = points_3d
    transmited_points_3d = np.dot(transmit_mat, points_3d_homo.T).T[:, :3]
    return transmited_points_3d

# apollo pose to matrix
def pose_to_matrix(pose):
    if np.ndim(pose) == 1:
        euler = pose[:3]
        translation = pose[3:]
        r = R.from_euler('xyz', np.array(euler), degrees=False)
        mat = r.as_matrix()
        mat = np.hstack([mat, translation.reshape((3, 1))])
        return mat


# Depth to 3D simulation
def depth_to_points_cloud_simulation(rgb_image, depth_image):
    vertices = []
    colors = []
    depth_image = grey_to_depth(depth_image / 255)
    print(depth_image[0, :].max())
    K = np.eye(3, 3)
    asuumed_f_scale = 1
    # WINDOW_WIDTH_HALF = 1080 * asuumed_f_scale
    # WINDOW_HEIGHT_HALF = 1920* asuumed_f_scale
    # K[0, 2] = WINDOW_WIDTH_HALF
    # K[1, 2] = WINDOW_HEIGHT_HALF
    K[0, 2] = 0
    K[1, 2] = 0

    fx = 1080 * asuumed_f_scale / (2.0 * math.tan(90.0 * math.pi / 360.0))
    fy = 1920 * asuumed_f_scale / (2.0 * math.tan(90.0 * math.pi / 360.0))
    K[0, 0] = fx
    K[1, 1] = fy
    K_inv = np.linalg.inv(K)
    print(K_inv)
    K_inv[0, 2] = 0
    K_inv[1, 2] = 0
    for i in range(0, rgb_image.shape[0], 4):
        for j in range(0, rgb_image.shape[1], 4):
            depth_image[i, j] = depth_image[i, j]
            # if depth_image[i, j]<150 and depth_image[i, j]>0:
            if depth_image[i, j] > 1:
                point = np.dot(K_inv, np.array(
                    [(i - 1079 / 2) * depth_image[i, j], (j - 1919 / 2) * depth_image[i, j], depth_image[i, j]]))
                color = [rgb_image[i, j, 0], rgb_image[i, j, 1], rgb_image[i, j, 2]]
                vertices.append(point)
                colors.append(color)
    vertices = np.array(vertices)
    print(vertices.max())
    colors = np.array(colors)
    vertices[:, 2] = vertices[:, 2]
    vertices = vertices * asuumed_f_scale
    return vertices, colors


def visualize_points_cloud(vertices, colors):
    colors = colors / 255
    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection='3d')
    ax.set_title('sample')
    ax.set_xlim([-60, 4])
    ax.set_ylim([-120, 120])
    ax.set_zlim([0, 255])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter3D(-vertices[:, 1], vertices[:, 0], vertices[:, 2], s=1, c=colors)
    plt.show()


def grey_to_depth(grey):
    # for i in range(grey.shape[0]):
    #     for j in range(grey.shape[1]):
    #         if grey[i, j] == 0.005 or grey[i, j] == 1:
    #             grey[i, j] = None
    #         if grey[i, j]:
    #             grey[i, j] = grey[i, j] - 1
    #             grey[i, j] = np.exp(grey[i, j]) * 5.70378
    #         if grey[i, j] == 1 or grey[i, j] == 0:
    #             grey[i, j] = None
    #         if grey[i, j]:
    #             grey[i, j] = grey[i, j] * 10000
    # grey = np.nan_to_num(grey)
    # grey = (grey-grey[grey>0].min())
    # print(grey[grey>0].min())
    depth = np.exp((grey - 1.0) * 5.70378) * 10000
    print(depth.max())
    return depth

def points_2d_project(points_3d, cam_extr, cam_intr):
    projected_points_3d = points_3d_transmit(points_3d, cam_extr)
    projected_points_3d = np.dot(cam_intr, projected_points_3d.T).T
    points_2d = projected_points_3d[:, :2].copy()
    points_2d[:, 0] = points_2d[:, 0]/projected_points_3d[:, 2]
    points_2d[:, 1] = points_2d[:, 1] / projected_points_3d[:, 2]
    return points_2d, projected_points_3d

def check_triangles_2d(vertices_2d, triangles, point_2d):
    valid_id = []
    for i in range(triangles.shape[0]):
        triangle = np.zeros((3, 2))
        triangle[0, :] = vertices_2d[triangles[i, 0]]
        triangle[1, :] = vertices_2d[triangles[i, 1]]
        triangle[2, :] = vertices_2d[triangles[i, 2]]
        if point_2d[0]<=triangle[:, 0].max() and point_2d[0]>=triangle[:, 0].min() and point_2d[1]<=triangle[:, 1].max() and point_2d[1]>=triangle[:, 1].min():
            if check_triangle_2d(triangle, point_2d):
                valid_id.append(i)
    return valid_id


def check_triangle_2d(triangle, point_2d):
    pa = triangle[0, :] - point_2d
    pb = triangle[1, :] - point_2d
    pc = triangle[2, :] - point_2d
    t1 = np.cross(pa, pb)
    t2 = np.cross(pb, pc)
    t3 = np.cross(pc, pa)
    if (t1>=0 and t2>=0 and t3>=0) or (t1<=0 and t2<=0 and t3<=0):
        return True

def find_closest_triangle(vertices, triangles, valid_triangle_ids):
    # choose the final triangle
    triangle = np.ones((3, 3)) * 1000000
    final_triangle_id = valid_triangle_ids[0]
    for triangle_id in valid_triangle_ids:
        new_triangle = np.zeros((3, 3))
        new_triangle = vertices[triangles[valid_triangle_ids[0]]]
        # new_triangle[1, :] = vertices[triangles[valid_triangle_ids[1]]]
        # new_triangle[2, :] = vertices[triangles[valid_triangle_ids[2]]]
        if new_triangle[:, 2].max() <= triangle[:, 2].max():
            triangle = new_triangle
            final_triangle_id  = triangle_id
    return triangle, final_triangle_id

# line_poins: 2*3(2 points in the line) plane_points: 3*3(3 points in the plane)
def find_line_plane_cross(line_points, plane_points):
    line_a = line_points[0, :]
    line_b = line_points[1, :]
    triangle_nm = np.cross(plane_points[0, :] - plane_points[1, :], plane_points[0, :] - plane_points[2, :])
    triangle_d = -np.dot(triangle_nm, plane_points[2, :].T)
    t = (np.dot(triangle_nm, line_a) + triangle_d) / (np.dot(triangle_nm, line_b) + triangle_d)
    return (t*line_b - line_a)/(t-1)

def point_2d_to_3d_project(point_2d, vertices, triangles, model_pose, cam_intr):
    if model_pose.shape[0] == 1 or model_pose.shape[0] == 6:
        model_mat = mat_to_4x4(pose_to_matrix(model_pose))
    else:
        model_mat = mat_to_4x4(model_pose)
    
    # calculate the 2d and 3d points of the vertices    
    vertices_3d = points_3d_transmit(vertices, model_mat)
    vertices_2d = points_2d_project(vertices, model_mat, cam_intr)
    # select the valid triangles which contains the point_2d
    valid_triangle_ids = check_triangles_2d(vertices_2d, triangles, point_2d)
    if len(valid_triangle_ids) == 0:
        return False
    else:
        # find the cross points from the ray from point_2d to the triangle plane
        cross_point = np.ones(3) * 500
        close_triangle_id = -1
        for valid_triangle_id in valid_triangle_ids:
            valid_triangle = vertices_3d[triangles[valid_triangle_id]]
            # calculate the ray from point_2d to the triangle plane
            ray_line = np.ones((2, 3))
            ray_line[0, :2] = point_2d
            ray_line[1, :2] = point_2d
            ray_line[0, :] = ray_line[0, :]*9
            ray_line[1, :] = ray_line[1, :]*10
            ray_line[0, :] = np.dot(np.linalg.inv(cam_intr), ray_line[0, :])
            ray_line[1, :] = np.dot(np.linalg.inv(cam_intr), ray_line[1, :])
            new_cross_point = find_line_plane_cross(ray_line, valid_triangle)
            if new_cross_point[2]<cross_point[2]:
                cross_point = new_cross_point
                close_triangle_id = valid_triangle_id
        cross_point = points_3d_transmit(np.array([cross_point]), np.linalg.inv(model_mat))
        return [cross_point, close_triangle_id]
    
def point_2d_to_ground(point_2d: np.array, cam_intri: np.array, cam_extri: np.array):
    # cam_extri = cam_mat_to_blender_mat(cam_extri)
    point_2d_homo = np.append(point_2d, 1)
    point_2d_projected = np.dot(np.linalg.inv(cam_intri[:3, :3]), point_2d_homo.T)
    x_2d = np.float64(point_2d_projected[0])
    y_2d = np.float64(point_2d_projected[1])
    z_2d = np.float64(point_2d_projected[2])
    ground_normal = cam_extri[2, :]
    z_cam = -np.float64(ground_normal[-1])/(np.float64(ground_normal[0])*x_2d + np.float64(ground_normal[1])*y_2d + np.float64(ground_normal[2])*z_2d)
    x_cam = x_2d * z_cam
    y_cam = y_2d * z_cam
    z_cam = z_2d * z_cam
    point_3d_cam_homo = np.array([x_cam, y_cam, z_cam, 1])  
    point_3d_world_homo = np.dot(cam_extri, point_3d_cam_homo) 
    point_3d_world_homo[2] = 0
    # x = cam_extri[0, 0] * point_3d_cam_homo[0] + cam_extri[0, 1] * point_3d_cam_homo[1] + cam_extri[0, 2] * point_3d_cam_homo[2] + cam_extri[0, 3] * point_3d_cam_homo[3]
    # point_3d_world_homo[0] = cam_extri[0, 0] * point_3d_cam_homo[0] + cam_extri[0, 1] * point_3d_cam_homo[0] + cam_extri[0, 2] * point_3d_cam_homo[0] + cam_extri[0, 3] * point_3d_cam_homo[0]
    # point_3d_world_homo[2] = cam_extri[2, 0] * point_3d_cam_homo[0] + cam_extri[2, 1] * point_3d_cam_homo[1] + cam_extri[2, 2] * point_3d_cam_homo[2] + cam_extri[2, 3] * point_3d_cam_homo[3]
    return point_3d_world_homo     

def points_2d_to_ground(points_2d: np.array, cam_intri: np.array, cam_extri: np.array):
    if len(points_2d.shape) == 1:
        points_2d = np.array([points_2d])
    points_2d_homo = points_to_homo(points_2d)
    point_2d_projected = np.dot(np.linalg.inv(cam_intri[:3, :3]), points_2d_homo.T).T
    x_2d = np.float64(point_2d_projected[:, 0])
    y_2d = np.float64(point_2d_projected[:, 1])
    z_2d = np.float64(point_2d_projected[:, 2])
    ground_normal = cam_extri[2, :]
    z_cam = -np.float64(ground_normal[-1])/(np.float64(ground_normal[0])*x_2d + np.float64(ground_normal[1])*y_2d + np.float64(ground_normal[2])*z_2d)
    x_cam = x_2d * z_cam
    y_cam = y_2d * z_cam
    z_cam = z_2d * z_cam
    point_3d_cam_homo = points_to_homo(np.vstack((x_cam, y_cam, z_cam)).T)  
    point_3d_world_homo = np.dot(cam_extri, point_3d_cam_homo.T).T 
    point_3d_world_homo[:, 2] = 0
    # x = cam_extri[0, 0] * point_3d_cam_homo[0] + cam_extri[0, 1] * point_3d_cam_homo[1] + cam_extri[0, 2] * point_3d_cam_homo[2] + cam_extri[0, 3] * point_3d_cam_homo[3]
    # point_3d_world_homo[0] = cam_extri[0, 0] * point_3d_cam_homo[0] + cam_extri[0, 1] * point_3d_cam_homo[0] + cam_extri[0, 2] * point_3d_cam_homo[0] + cam_extri[0, 3] * point_3d_cam_homo[0]
    # point_3d_world_homo[2] = cam_extri[2, 0] * point_3d_cam_homo[0] + cam_extri[2, 1] * point_3d_cam_homo[1] + cam_extri[2, 2] * point_3d_cam_homo[2] + cam_extri[2, 3] * point_3d_cam_homo[3]
    return point_3d_world_homo    

def cal_vanishing_line(img_size, cam_intri: np.array, cam_extri: np.array):
    mat = np.dot(cam_extri[2, :3], np.linalg.inv(cam_intri[:3, :3]))
    y_max0 = np.float64(-mat[2])/np.float64(mat[1])
    y_max1 = (np.float64(-mat[2])-np.float64(mat[0])*np.float64(img_size[0]))/np.float64(mat[1])
    return np.array([[0, y_max0],[img_size[0], y_max1]])

def cal_max_2d_in_ground_temp(img_size, cam_intri: np.array, cam_extri: np.array):
    point_3d_world_homo = np.array([10, 3, 0, 1])
    point_2d = np.dot(cam_intri, np.dot(np.linalg.inv(cam_extri), point_3d_world_homo))
    point_2d_res = [point_2d[0]/point_2d[2], point_2d[1]/point_2d[2]]
    print('here')
    
def cal_homo_from_intri_extri(cam_intri, cam_extri, output_size: np.array = np.array([1920, 1080]), M = None):
    P = np.dot(cam_intri[:3, :3], np.linalg.inv(cam_extri)[:3, :])
    if M is None:
        M, pxPerM = cal_M(output_size=output_size)
        new_M = M.copy()
    M[0, :] = new_M[1, :]
    M[1, :] = - new_M[0, :]
    # M = M*0.9
    # M[2, 2] = 1.0
    homo = np.linalg.inv(np.dot(P, M))
    return homo, pxPerM

def cal_M(r=20.0, output_size: np.array = np.array([1920, 1080]), fx=682.578, fy=682.578, z_cam=35.0, y_cam=0.0, x_cam=0.0):
    dx = output_size[0] / fx * z_cam
    dy = output_size[1] / fy * z_cam
    # world x, world y
    pxPerM = (output_size[1] / dy, output_size[0] / dx)
    shift = (output_size[1] / 2.0, output_size[0] / 2.0)
    shift = shift[0] + y_cam * pxPerM[0], shift[1] - x_cam * pxPerM[1]
    M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]], [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return M, pxPerM

# calculate suitable matrix M from front_cam
def cal_xy_cam_from_front_back(front_img_size: np.array, back_img_size: np.array, cam_intri_front: np.array, cam_extri_front: np.array, cam_intri_back: np.array, cam_extri_back: np.array, output_size: np.array = np.array([1920, 1080])):
    points_origin_front_homo = np.array([[0, front_img_size[0], 1], [front_img_size[1], front_img_size[0], 1]])
    points_origin_back_homo = np.array([[0, back_img_size[0], 1], [back_img_size[1], back_img_size[0], 1]])
    front_homo, pxPerM = cal_homo_from_intri_extri(cam_intri_front, cam_extri_front, output_size)
    back_homo, _ = cal_homo_from_intri_extri(cam_intri_back, cam_extri_back, output_size)
    points_front_ipm = points_homo_2_2d(np.dot(front_homo, points_origin_front_homo.T).T)
    points_back_ipm = points_homo_2_2d(np.dot(back_homo, points_origin_back_homo.T).T)
    points_ipm = np.vstack((points_front_ipm, points_back_ipm))
    points_ipm_center = np.mean(points_ipm, axis=0)
    # points_to_shift = 
    print('here')   
    
def get_ipm_mask(ori_img_size: np.array, homo: np.array, ipm_img_size: np.array, mask_field='Bottom'):
    points_origin_homo = np.array([[0, ori_img_size[0], 1], [ori_img_size[1], ori_img_size[0], 1]])
    points_ipm = points_homo_2_2d(np.dot(homo, points_origin_homo.T).T)
    # slope = (points_ipm[0, 0] - points_ipm[1, 0]) / (points_ipm[0, 1] - points_ipm[1, 1])
    # intercept = points_ipm[1, 0] - slope * points_ipm[1, 1]
    mask = np.ones((ipm_img_size[1], ipm_img_size[0]))*255
    points_ipm = points_ipm.astype(int)
    if mask_field == 'Bottom':
        mask[(points_ipm[:, 1].max()+1):ipm_img_size[1], :] = 0
    if mask_field == 'Right':
        mask[:, (points_ipm[:, 0].max()+1):ipm_img_size[0]] = 0
    if mask_field == 'Left':
        mask[:, :(points_ipm[:, 0].min()-1)] = 0
    elif mask_field == 'Top':
        mask[: (points_ipm[:, 1].min()-1), :] = 0
    return mask.astype(np.uint8)
    
def points_homo_2_2d(points: np.array):
    points = points/points[:, -1][:, np.newaxis]
    return points[:, :2]

def points_homo_2_origin(homo, points):
    if points.shape[0] == 0:
        return None
    if points.shape[1] == 2:
        reprojected_points = points_to_homo(points)
    reprojected_points = np.dot(np.linalg.inv(homo), reprojected_points.T).T
    reprojected_points[:, 0] = reprojected_points[:, 0]/reprojected_points[:, 2]
    reprojected_points[:, 1] = reprojected_points[:, 1]/reprojected_points[:, 2]
    return reprojected_points[:, :2]

def points_to_homo(points: np.array):
    return np.hstack((points.copy(), np.ones((points.shape[0], 1))))

    