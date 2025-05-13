import numpy as np
from .img_utils.mask_utils import del_lane_line, get_road_mask, mask_to_img
from .projection_utils import cal_homo_from_intri_extri, points_homo_2_origin, points_2d_to_ground
import cv2

# get ipm masj of road
def mask_2_road_ipm(mask_img: np.array, cam_intri: np.array, cam_extri: np.array, ipm_size: np.array=[3840, 2160], line_color: np.array=np.array([220, 209, 234]), road_color: np.array = np.array([128,  64, 128])):
    mask_img = del_lane_line(mask_img, line_color, road_color)
    mask, road_mask_rgb = get_road_mask(mask_img, road_color)
    # cv2.imwrite('temp/pure_road_mask_no_lane.png', road_mask_rgb)
    # mask, road_lane_mask = self.get_road_lane_mask(mask_img)
    mask = mask_to_img(mask)
    free_space_mask_ipm, homo, pxPerM = img_2_ipm(mask, cam_intri, cam_extri, ipm_size)
    return free_space_mask_ipm, mask, homo, pxPerM

# img 2 ipm
def img_2_ipm(img: np.array, cam_intri: np.array, cam_extri: np.array, ipm_size: np.array=[3840, 2160]):
    homo, pxPerM = cal_homo_from_intri_extri(cam_intri, cam_extri, ipm_size)
    img_mask = np.ones_like(img)*255
    if len(img_mask.shape) == 3:
        img_mask = img_mask[:, :, 0]
    img_size = np.array([img.shape[1], img.shape[0]])
    origin_point = [img_size[0], img_size[1], 1]
    new_point = np.dot(homo, origin_point)
    new_point = new_point/new_point[2]
    new_point_floor = np.floor(new_point.copy()[:2]).astype(int)
    new_point_ceil = np.ceil(new_point.copy()[:2]).astype(int)
    new_point_round = np.round(new_point.copy()[:2]).astype(int)
    new_point_choice = [new_point_floor, new_point_ceil, new_point_round]
    repro_point = np.dot(np.linalg.inv(homo), new_point)
    repro_point = repro_point/repro_point[2]
    img_ipm = cv2.warpPerspective(img, homo, (ipm_size[0], ipm_size[1]))
    img_mask_ipm = cv2.warpPerspective(img_mask, homo, (ipm_size[0], ipm_size[1]))
    flag = False
    if (new_point_round[0] + 3 >= img_mask_ipm.shape[1]) or (new_point_round[1] + 3>= img_mask_ipm.shape[0]) or (new_point_round[0] -3 <0) or (new_point_round[1] -3 <0):
        return img_ipm, homo, pxPerM 
    for i in range(7):
        for j in range(7):
            new_point_choice = new_point_round.copy()
            new_point_choice[0] = new_point_choice[0] + i -3
            new_point_choice[1] = new_point_choice[1] + j -3
            if (img_mask_ipm[new_point_choice[1], new_point_choice[0]]).all()>0:
                flag = True
                break
        if flag:
            break
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_mask_ipm, connectivity=8)
    label = labels[new_point_choice[1], new_point_choice[0]]
    img_mask_ipm[np.logical_not(labels==label)] = 0            
    img_ipm[img_mask_ipm == 0] = 0
    return img_ipm, homo, pxPerM

# search available location in ipm mask accroding to box
def get_random_alternative_location_in_ipm(ipm_free_space_mask: np.array, box_corners_ipm: np.array, get_all = True, extra_space = 8):
    # get some ipm locations
    random_locations_in_mask_y, random_locations_in_mask_x = np.meshgrid(np.arange(0, ipm_free_space_mask.shape[0] - 1, 10), np.arange(0, ipm_free_space_mask.shape[1] - 1, 5))
    random_offset = np.random.randint(-5, 5)
    # random_offset = 0
    random_locations_in_mask_y = random_locations_in_mask_y + random_offset
    random_locations_in_mask_x = random_locations_in_mask_x + random_offset
    # w first, h second
    random_locations_in_mask = np.vstack((random_locations_in_mask_x.flatten(), random_locations_in_mask_y.flatten())).T
    # for numpy h first, w second
    random_locations_in_mask = random_locations_in_mask[(ipm_free_space_mask/255).astype(bool)[random_locations_in_mask[:, 1], random_locations_in_mask[:, 0]]]
    # get car model box in ipm in origin
    # self.cal_blr2pixel_scale_in_ipm(cam_intri, cam_extri, np.array([ipm_free_space_mask.shape[1], ipm_free_space_mask.shape[0]]))
    np.random.shuffle(random_locations_in_mask)
    alternative_locations_in_ipm = filter_alternative_location(ipm_free_space_mask, random_locations_in_mask, box_corners_ipm, get_all = get_all, extra_space = extra_space)
    return alternative_locations_in_ipm  

def filter_alternative_location(mask_ipm: np.array, locations: np.array, box_corners: np.array, get_all = True, extra_space = 8):
    if get_all:
        alternative_locations = []
        for i in range(locations.shape[0]):
            location = locations[i, :]
            box_corners_in_loction = move_box(box_corners, location)
            if check_location(box_corners_in_loction, mask_ipm, extra_space = extra_space):
                alternative_locations.append(location)
        return np.array(alternative_locations)
    else:
        for i in range(locations.shape[0]):
            location = locations[i, :]
            box_corners_in_loction = move_box(box_corners, location)
            if check_location(box_corners_in_loction, mask_ipm, extra_space = extra_space):
                return location
        return None

def move_box(box_corners: np.array, location: np.array):
    box_corners_in_loction = box_corners.copy()
    box_corners_in_loction[:, 0] = box_corners_in_loction[:, 0] + location[0]
    box_corners_in_loction[:, 1] = box_corners_in_loction[:, 1] + location[1]
    return box_corners_in_loction

def get_poly_in_ipm(poly_points: np.array, mask_img: np.array, expand_kernel_size: int = 16):
        mask_poly = np.zeros(mask_img.shape[:2], dtype=np.uint8)
        mask_poly = cv2.fillPoly(mask_poly, [poly_points.astype(np.int32)], 255)
        # in case slightly expanding the mask_poly
        kernel = np.ones((expand_kernel_size, expand_kernel_size), np.uint8)
        mask_poly = cv2.dilate(mask_poly, kernel, iterations = 1)
        mask_poly = mask_poly.astype(bool)
        return mask_poly
    
def check_location(poly_points: np.array, mask_img: np.array, extra_space = 8):
        expand_kernel_size = extra_space
        if np.min(poly_points[:, 0] - expand_kernel_size) < 0 or np.max(poly_points[:, 0])>mask_img.shape[1]+expand_kernel_size or np.min(poly_points[:, 1] - expand_kernel_size) < 0 or np.max(poly_points[:, 1] + expand_kernel_size)>mask_img.shape[0]+1:
            return False
        mask_poly = get_poly_in_ipm(poly_points, mask_img, expand_kernel_size)
        mask_road_inv = ~mask_img.copy().astype(bool)
        return not(np.any(np.logical_and(mask_road_inv, mask_poly)))
    
def locations_2_blr(ipm_img: np.array, locations: np.array, cam_intri: np.array, cam_extri: np.array):
    homo, pxPerM = cal_homo_from_intri_extri(cam_intri, cam_extri, np.array([ipm_img.shape[1], ipm_img.shape[0]]))
    scale = pxPerM[0]
    if locations[0] == 0:
        return None
    locations_in_origin = points_homo_2_origin(homo, locations)
    Locations_in_blr = points_2d_to_ground(locations_in_origin, cam_intri, cam_extri)
    return Locations_in_blr  

def point_3d_to_ipm(location_xy: np.array, pxPerM, ipm_size):
    return np.array([ipm_size[0]/2 - location_xy[1]*pxPerM[1], ipm_size[1]/2 - location_xy[0]*pxPerM[0]])

def points_3d_to_ipm(points_3d: np.array, pxPerM, ipm_size):
    points_ipm = np.concatenate(- points_3d[:, 1]*pxPerM[1] + ipm_size[0]/2, -points_3d[:, 0]*pxPerM[0] + ipm_size[1]/2)
    return points_ipm
    # return np.array([ipm_size[0]/2 - location_xy[1]*pxPerM[1], ipm_size[1]/2 - location_xy[0]*pxPerM[0]])