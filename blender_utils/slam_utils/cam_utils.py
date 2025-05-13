import  math
import cv2
import numpy as np
from typing import Union
from scipy.interpolate import griddata
from .matrix_utils import mat_to_pose

def cam_intr_2_lens(img_size, camera_intrinsic, sensor_width, shift_scale=1.0):
    # sensor_height = sensor_width * img_size[0] / img_size[1]
    # sensor_diagonal = math.sqrt(sensor_width**2 + sensor_height**2)
    # fov_diagonal = 2 * math.atan((sensor_diagonal / 2) / (camera_intrinsic[0, 0]/100))
    # focal_length = (sensor_diagonal / 2) / math.tan(fov_diagonal / 2)
    focal_length2 = (sensor_width / max(img_size[0], img_size[1])) * max(camera_intrinsic[0, 0], camera_intrinsic[1, 1])
    shift_x = -(camera_intrinsic[0, 2] / img_size[0] - 0.5)/shift_scale
    shift_y = ((camera_intrinsic[1, 2] - img_size[1]/2.0) / img_size[0])/shift_scale
    return focal_length2, shift_x, shift_y

def fisheyeEquirectsolidimg2point3d(img_size: np.ndarray, sensor_size: float = 15, focal_length: float = 10, fov=np.pi, shift_x: float = 0.0, shift_y: float = 0.0):
    w = img_size[0]
    h = img_size[1]
    M, N = np.meshgrid(np.arange(w), np.arange(h))
    points2d = np.concatenate((M[:, :, np.newaxis], N[:, :, np.newaxis]), axis=2)
    points3d, points2d_fisheye = fisheyeEquirectsolidpoints2d2point3d(points2d, img_size, sensor_size, focal_length, fov, shift_x, shift_y)
    return points3d, points2d_fisheye
    
def fisheyeEquirectsolidpoints2d2point3d(points2d: np.ndarray, img_size: np.ndarray, sensor_size: float = 15, focal_length: float = 10, fov = np.pi, shift_x: float = 0.0, shift_y: float = 0.0):
    cx = img_size[0] / 2.0
    cy = img_size[1] / 2.0
    scale = sensor_size/focal_length
    R = 1.0
    u = points2d[:, :, 0]
    v = points2d[:, :, 1]
    x = ( u- cx) * scale * R / ( img_size[0] ) 
    y = ( v - cy) * scale * R / ( img_size[0]  )
    alpha = fov / 4
    limitr = 2 * R * np.sin(alpha)
    r1 = np.sqrt(x**2 + y**2)

    r1[r1==0] = R
    r1_area_mask = r1 < limitr
    # r1[r1>2*R] = R
    r1 = r1[r1_area_mask]
    x = x[r1_area_mask]
    y = y[r1_area_mask]
    u = u[r1_area_mask]
    v = v[r1_area_mask]
    
    tha = 2 * np.arcsin(r1/2/R)
    r = R * np.sin(tha)
    cos_phia = x / r1
    sin_phia = y / r1
    phia = np.arctan2(y,x)
    X = r * cos_phia
    Y = r * sin_phia
    Z = R * np.cos(tha) 
    points3d = np.concatenate((X[..., np.newaxis], Y[..., np.newaxis], Z[..., np.newaxis]), axis=1)
    points2d_fisheye = np.concatenate((u[..., np.newaxis], v[..., np.newaxis]), axis=1)
    return points3d, points2d_fisheye

def points_3d_to_distort_2d(points3d, distort_K, distort_coeff, cam_extri, cam_model='opencv_omini'):
    if len(points3d.shape) == 2:
        points3d = points3d[np.newaxis, :, :]
    points3d = points3d.astype(np.float64)
    rvec = cv2.Rodrigues(cam_extri[:3, :3])[0]
    tvec = cam_extri[:3, 3]
    if cam_model == 'opencv_omini':
        points_2d_distort, _ = cv2.omnidir.projectPoints(points3d, rvec=rvec, tvec=tvec, K=distort_K, D=distort_coeff[:4], xi=distort_coeff[4])
    elif cam_model == 'opencv_fisheye':
        points_2d_distort, _ = cv2.fisheye.projectPoints(points3d, rvec=rvec, tvec=tvec, K=distort_K, D=distort_coeff)
    return points_2d_distort

# map from one image to another image
def map_project_2d(original_points_2d, ori_img_size, projected_points_2d):
    m, n = [i for i in range(ori_img_size[0])], [j for j in range(ori_img_size[1])]
    M, N = np.meshgrid(m, n)
    res_map1 = griddata(original_points_2d.reshape((-1, 2)), projected_points_2d[0].reshape(-1), (M, N), method='cubic').astype(np.float32)
    res_map2 = griddata(original_points_2d.reshape((-1, 2)), projected_points_2d[1].reshape(-1), (M, N), method='cubic').astype(np.float32)
    res_map1[res_map1<0] = np.float32('nan')
    res_map2[res_map2<0] = np.float32('nan')
    return res_map1, res_map2

# img from one image to another images
def img_project_2d(original_points_2d, ori_img, projected_points_2d, projected_img_size):
    ori_img_size = (ori_img.shape[1], ori_img.shape[0])
    map1, map2 = map_project_2d(original_points_2d, ori_img_size, projected_points_2d)
    new_img = np.zeros((projected_img_size[1], projected_img_size[0], ori_img.shape[2]), dtype=np.uint8)
    for i in range(map1.shape[1]):
        for j in range(map2.shape[0]):
            if not np.isnan(map1[j, i]) and not np.isnan(map2[j, i]):
                new_coord = (int(map1[j, i]), int(map2[j, i]))
                if new_coord[0] < projected_img_size[0] and new_coord[1] < projected_img_size[1]:
                    new_img[new_coord[0], new_coord[1], :] = ori_img[j, i, :]
    return new_img

class DistiortTool():
    # h and w means original height and width of image
    def __init__(self):
        pass
    def update_map(self, distort_coeff, distort_K = None, distort_size = None, undistort_K = None, undistort_size = None, cam_extri = None, use_distort = False):
        self.update_cam_info(distort_coeff, distort_K, distort_size, undistort_K, undistort_size, cam_extri)
        self.update_map_undistort()
        if use_distort:
            self.update_map_distort()
            
    def update_cam_info(self, distort_coeff, distort_K = None, distort_size = None, undistort_K = None, undistort_size = None, cam_extri = None, cam_model = 'opencv_omini'):
        if undistort_size is None and undistort_K is None:
            undistort_size = distort_size
            undistort_K = distort_K
        elif undistort_size is None:
            undistort_w = int(undistort_K[0, 2] * 2)
            undistort_h = int(undistort_K[1, 2] * 2)
            undistort_size = (undistort_w, undistort_h)
        elif undistort_K is None:
            undistort_K = np.eye(3)
            undistort_K[0, 0] = distort_K[0, 0]
            undistort_K[1, 1] = distort_K[1, 1]
            undistort_K[0, 2] = undistort_size[0] / 2
            undistort_K[1, 2] = undistort_size[1] / 2
        if distort_size is None and distort_K is None:
            distort_size = undistort_size
            distort_K = undistort_K
        elif distort_size is None:
            distort_w = int(distort_K[0, 2] * 2)
            distort_h = int(distort_K[1, 2] * 2)
            distort_size = (distort_w, distort_h)
        elif distort_K is None:
            distort_K = np.eye(3)
            distort_K[0, 0] = undistort_K[0, 0]
            distort_K[1, 1] = undistort_K[1, 1]
            distort_K[0, 2] = distort_size[0] / 2
            distort_K[1, 2] = distort_size[1] / 2
        self.distort_coeff = distort_coeff
        self.distort_size = distort_size
        self.distort_K = distort_K
        self.undistort_size = undistort_size
        self.undistort_K = undistort_K
        if len(self.undistort_K.shape) == 1:
            self.undistort_K = np.reshape(self.undistort_K, (3, 3))
        self.cam_extri = cam_extri
        self.cam_model = cam_model
        if len(self.distort_coeff) == 5:
            self.cam_model = 'opencv_omini'
        elif len(self.distort_coeff) == 4:
            self.cam_model = 'opencv_fisheye'
        
            
    def update_map_undistort(self):
        if self.cam_model == 'opencv_omini':
            xi = self.distort_coeff[4]
            distort_coeff = self.distort_coeff[:4]
            self.map_undistort, self.map_undistort2 = cv2.omnidir.initUndistortRectifyMap(self.distort_K, distort_coeff, xi=np.array(xi), R=np.eye(3), P = self.undistort_K, size = self.undistort_size, m1type = cv2.CV_16SC2, flags=1)
        else:
            distort_coeff = self.distort_coeff
            self.map_undistort, self.map_undistort2 = cv2.fisheye.initUndistortRectifyMap(self.distort_K, distort_coeff, R=np.eye(3), P = self.undistort_K, size = self.undistort_size, m1type = cv2.CV_16SC2)
        
    def update_map_distort(self):
        if not hasattr(self, 'map_undistort'):
            self.update_map_undistort()
        if self.undistort_size[0]/self.distort_size[0] > 2.0:
            map_undistort = self.map_undistort.copy()
            map_undistort = cv2.resize(map_undistort, (self.distort_size[0]*2, self.distort_size[1]*2), interpolation=cv2.INTER_LINEAR)
        else:
            map_undistort = self.map_undistort.copy()
        map_undistort_size = (map_undistort.shape[1], map_undistort.shape[0])
        m = [i for i in range(map_undistort_size[0])]
        n = [j for j in range(map_undistort_size[1])]
        distort_points_map = np.meshgrid(m, n)
        M, N = np.meshgrid(m, n)
        self.map_distort1 = griddata(map_undistort.reshape((-1, 2)), distort_points_map[0].reshape(-1), (M, N), method='cubic')
        self.map_distort2 = griddata(map_undistort.reshape((-1, 2)), distort_points_map[1].reshape(-1), (M, N), method='cubic')
        self.map_distort1 = self.map_distort1.astype(np.float32)[:self.distort_size[1], :self.distort_size[0]]
        self.map_distort2 = self.map_distort2.astype(np.float32)[:self.distort_size[1], :self.distort_size[0]]
        self.map_distort1 = self.map_distort1 * self.undistort_size[0] / map_undistort_size[0]
        self.map_distort2 = self.map_distort2 * self.undistort_size[1] / map_undistort_size[1]
        self.map_distort = np.concatenate((self.map_distort1[:, :, np.newaxis], self.map_distort2[:, :, np.newaxis]), axis=2)
        
    def update_fisheye_blr_info(self, img_size: np.ndarray, sensor_size: float = 3*15.0, focal_length: float = 10.0, fov: float = np.pi * 1.5, shift_x: float = 0.0, shift_y: float = 0.0):
        self.blr_focal_length = focal_length
        self.blr_shift_x = shift_x
        self.blr_shift_y = shift_y
        self.blr_sensor_size = sensor_size
        self.blr_img_size = img_size
        self.blr_fov = fov
        
    def update_map_fisheye2distort(self):
        points3d_ball, points2d_fisheye = fisheyeEquirectsolidimg2point3d(self.blr_img_size, self.blr_sensor_size, self.blr_focal_length, self.blr_fov, self.blr_shift_x, self.blr_shift_y)
        points3d_ball = points3d_ball.reshape((-1, 3))
        points2d_distort = self.points_3d_cam_to_distort_2d(points3d_ball)
        # points2d_distort = points2d_distort.reshape((self.blr_img_size[1], self.blr_img_size[0], 2))
        # self.map_fisheye2distort = points2d_distort_map.astype(np.float32)
        # self.map_fisheye2distort1 = self.map_fisheye2distort[:, :, 0]
        # self.map_fisheye2distort2 = self.map_fisheye2distort[:, :, 1]
        M, N = np.meshgrid(np.arange(self.blr_img_size[0]), np.arange(self.blr_img_size[1]))
        self.map_fisheye2distort1 = griddata(points2d_distort.reshape((-1, 2)), points2d_fisheye[...,0].reshape(-1), (M, N), method='cubic')
        self.map_fisheye2distort2 = griddata(points2d_distort.reshape((-1, 2)), points2d_fisheye[...,1].reshape(-1), (M, N), method='cubic')
        self.map_fisheye2distort1 = self.map_fisheye2distort1.astype(np.float32)
        self.map_fisheye2distort2 = self.map_fisheye2distort2.astype(np.float32)
        self.map_fisheye2distort = np.concatenate((self.map_fisheye2distort1[:, :, np.newaxis], self.map_fisheye2distort2[:, :, np.newaxis]), axis=2)
        
    def update_map_distort2fisheye(self):
        points3d_ball, points2d_fisheye = fisheyeEquirectsolidimg2point3d(self.blr_img_size, self.blr_sensor_size, self.blr_focal_length, self.blr_fov, self.blr_shift_x, self.blr_shift_y)
        points3d_ball = points3d_ball.reshape((-1, 3))
        points2d_distort = self.points_3d_cam_to_distort_2d(points3d_ball)
        self.map_distort2fisheye1 = np.zeros((self.blr_img_size[1], self.blr_img_size[0]), dtype=np.float32)
        self.map_distort2fisheye1[:] = np.nan
        self.map_distort2fisheye1[points2d_fisheye[...,1], points2d_fisheye[...,0]] = points2d_distort[..., 0]
        self.map_distort2fisheye2 = np.zeros((self.blr_img_size[1], self.blr_img_size[0]), dtype=np.float32)
        self.map_distort2fisheye2[:] = np.nan
        self.map_distort2fisheye2[points2d_fisheye[...,1], points2d_fisheye[...,0]] = points2d_distort[..., 1]
        self.map_distort2fisheye = np.concatenate((self.map_distort2fisheye1[:, np.newaxis], self.map_distort2fisheye2[:, np.newaxis]), axis=1)
        
    def img_fisheye2distort(self, img):
        if not hasattr(self, 'map_fisheye2distort'):
            self.update_map_fisheye2distort()
        new_img = cv2.remap(img, self.map_fisheye2distort1, self.map_fisheye2distort2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return new_img
    
    def img_distort2fisheye(self, img):
        if not hasattr(self, 'map_fisheye2distort'):
            self.update_map_fisheye2distort()
        new_img = cv2.remap(img, self.map_distort2fisheye1, self.map_distort2fisheye2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return new_img
        
        
    def K_scale_and_zoom(self, K, scale, zoom):
        new_K = K.copy()
        # zoom control large and small of objects
        new_K[0, 0] = K[0, 0] * zoom
        new_K[1, 1] = K[1, 1] * zoom
        # scale control the size of img (when scaling, the size of img must also change)
        new_K[0, 2] = K[0, 2] * scale
        new_K[1, 2] = K[1, 2] * scale
        return new_K
    
    def load_from_dict(self, dict):
        for key in dict:
            setattr(self, key, dict[key])
        
    def img_distort(self, img):
        if not hasattr(self, 'map_distort'):
            self.update_map_distort()
        new_img = cv2.remap(img, self.map_distort1, self.map_distort2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return new_img
    
    def img_undistort(self, img):
        if not hasattr(self, 'map_undistort'):
            self.update_map_undistort()
        new_img = cv2.remap(img, self.map_undistort, self.map_undistort2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return new_img 
    
    def points_3d_cam_to_distort_2d(self, points3d):
        if len(points3d.shape) == 2:
            points3d = points3d[np.newaxis, :, :]
        points3d = points3d.astype(np.float64)
        rvec = cv2.Rodrigues(np.eye(3))[0]
        tvec = np.array([0, 0, 0]).astype(np.float64)
        if self.cam_model == 'opencv_omini':
            points_2d_distort, _ = cv2.omnidir.projectPoints(points3d, rvec=rvec, tvec=tvec, K=self.distort_K, D=self.distort_coeff[:4], xi=self.distort_coeff[4])
        elif self.cam_model == 'opencv_fisheye':
            points_2d_distort, _ = cv2.fisheye.projectPoints(points3d, rvec=rvec, tvec=tvec, K=self.distort_K, D=self.distort_coeff)
        return points_2d_distort
    
    def points_3d_to_distort_2d(self, points3d):
        if len(points3d.shape) == 2:
            points3d = points3d[np.newaxis, :, :]
        points3d = points3d.astype(np.float64)
        rvec = cv2.Rodrigues(self.cam_extri[:3, :3])[0]
        tvec = self.cam_extri[:3, 3]
        if self.cam_model == 'opencv_omini':
            points_2d_distort, _ = cv2.omnidir.projectPoints(points3d, rvec=rvec, tvec=tvec, K=self.distort_K, D=self.distort_coeff[:4], xi=self.distort_coeff[4])
        elif self.cam_model == 'opencv_fisheye':
            points_2d_distort, _ = cv2.fisheye.projectPoints(points3d, rvec=rvec, tvec=tvec, K=self.distort_K, D=self.distort_coeff)
        return points_2d_distort
    
    def add_distort_to_box(self, box):
        old_box_in_normal = np.array(box).astype(int)
        for i in range(4):
            old_box_in_normal[i] = max(0, old_box_in_normal[i])
        if old_box_in_normal[0] == old_box_in_normal[2]:
            old_box_in_normal[2] = old_box_in_normal[2] + 1
        if old_box_in_normal[1] == old_box_in_normal[3]:
            old_box_in_normal[3] = old_box_in_normal[3] + 1
        old_box = self.map1[old_box_in_normal[1]: old_box_in_normal[3], old_box_in_normal[0]: old_box_in_normal[2], :]
        new_box = [old_box[:, :, 0].min(), old_box[:, :, 1].min(), old_box[:, :, 0].max(), old_box[:, :, 1].max()]
        new_box = np.array(new_box)
        new_width = int(self.h*self.target_size[0]/self.target_size[1])
        new_box[[0, 2]] = new_box[[0, 2]] - (self.w - new_width)//2
        new_box[[0, 2]] = new_box[[0, 2]] * (self.target_size[0] / new_width)
        new_box[[1, 3]] = new_box[[1, 3]] * (self.target_size[1] / self.h)
        # new_box = [self.map1[old_box_in_normal[1], old_box_in_normal[0], 0], self.map1[old_box_in_normal[1], old_box_in_normal[0], 1], self.map1[old_box_in_normal[3], old_box_in_normal[2] ,0], self.map1[old_box_in_normal[3], old_box_in_normal[2], 1]]
        return new_box
        
    def add_distort_to_label(self, label_dict, ori_normal_img_size):
        new_label_dict = label_dict
        full_boxes = label_dict['bbox_targets']
        front_boxes = label_dict['front_targets']
        side_boxes = label_dict['side_targets']
        for i in range(full_boxes.shape[0]):
            old_full_box = full_boxes[i, :4].copy()
            old_full_box[[0, 2]] = old_full_box[[0, 2]]/ori_normal_img_size[1]*self.w
            old_full_box[[1, 3]] = old_full_box[[1, 3]]/ori_normal_img_size[0]*self.h
            old_front_box = front_boxes[i, :4].copy()
            old_front_box[[0, 2]] = old_front_box[[0, 2]]/ori_normal_img_size[1]*self.w
            old_front_box[[1, 3]] = old_front_box[[1, 3]]/ori_normal_img_size[0]*self.h
            old_side_box = side_boxes[i, :4].copy()
            old_side_box[[0, 2]] = old_side_box[[0, 2]]/ori_normal_img_size[1]*self.w
            old_side_box[[1, 3]] = old_side_box[[1, 3]]/ori_normal_img_size[0]*self.h
            new_full_box = self.add_distort_to_box(old_full_box)
            new_front_box = new_full_box.copy()
            new_side_box = new_full_box.copy()
            # if old_front_box[0] > old_full_box[2]:
            #     old_front_box[0] = old_full_box[2]
            # if old_front_box[2] > old_full_box[2]:
            #     old_front_box[2] = old_full_box[2]
            for j in range(4):
                if abs(old_front_box[j] - old_full_box[j]) < 2:
                    new_front_box[j] = new_full_box[j]
                else:
                    if j % 2 == 0:
                        scale = (old_front_box[j] - old_full_box[0])/ (old_full_box[2] - old_full_box[0])
                        new_front_box[j] = new_full_box[0] + scale * (new_full_box[2] - new_full_box[0])
                    else:
                        scale = (old_front_box[j] - old_full_box[1])/ (old_full_box[3] - old_full_box[1])
                        new_front_box[j] = new_full_box[1] + scale * (new_full_box[3] - new_full_box[1])
                if abs(old_side_box[j] - old_full_box[j]) < 2:
                    new_side_box[j] = new_full_box[j]
                else:
                    if j % 2 == 0:
                        scale = (old_side_box[j] - old_full_box[0])/ (old_full_box[2] - old_full_box[0])
                        new_side_box[j] = new_full_box[0] + scale * (new_full_box[2] - new_full_box[0])
                    else:
                        scale = (old_side_box[j] - old_full_box[1])/ (old_full_box[3] - old_full_box[1])
                        new_side_box[j] = new_full_box[1] + scale * (new_full_box[3] - new_full_box[1])
                new_label_dict['bbox_targets'][i, :4] = new_full_box
                new_label_dict['front_targets'][i, :4] = new_front_box
                new_label_dict['side_targets'][i, :4] = new_side_box
        return new_label_dict
    
