import numpy as np
import cv2
# common utils
def mask_to_img(mask):
    return mask.astype(np.uint8)*255

def img_to_mask(img):
    if len(img.shape) == 3:
        img_mask = img[:,:,-1].copy()
    else:
        img_mask = img.copy()
    return (img_mask/255).astype(bool)

def combine_masks(mask_file_list: list ,use_or = True):
    assert(len(mask_file_list)>0)
    mask_list = []
    for mask in mask_file_list:
        if isinstance(mask, str):
            mask = cv2.imread(mask)
        mask = mask_to_img(mask)
        mask_list.append(mask)
    if use_or:
        init_mask = np.zeros_like(mask_list[0]).astype(bool)
        for mask in mask_list:      
            init_mask = np.logical_or(init_mask, mask)
    else:
        init_mask = np.ones_like(mask_list[0]).astype(bool)
        for mask in mask_list:
            init_mask = np.logical_and(init_mask, mask)
    return mask_to_img(init_mask)

# speically for road mask
# del lane line in mask
def del_lane_line(mask_img: np.ndarray, line_color: np.ndarray=np.array([220, 209, 234]), road_color: np.array = np.array([128,  64, 128]), cross_color = np.array([244, 35, 232])):
    mask = np.all(mask_img == line_color, axis=2)
    mask_img[mask, :] = road_color
    mask2 = np.all(mask_img == cross_color, axis=2)
    mask_img[mask2, :] = road_color
    return mask_img

def get_road_mask(mask_img: np.ndarray, road_color: np.ndarray=np.array([128,  64, 128])):
    mask = np.all(mask_img == road_color, axis=2)
    road_mask_rgb = cv2.bitwise_and(mask_img, mask_img, mask=mask.astype(np.uint8))
    return mask, road_mask_rgb

def get_car_mask(mask_img: np.array, car_color: np.array=np.array([30,  170, 250])):
    mask = np.all(mask_img == car_color, axis=2)
    car_mask_rgb = cv2.bitwise_and(mask_img, mask_img, mask=mask.astype(np.uint8))
    return mask, car_mask_rgb

def get_road_lane_mask(mask_img: np.ndarray, road_lane_color: np.ndarray=np.array([220, 209, 234])):
    mask = np.all(mask_img == road_lane_color, axis=2)
    road_lane_mask_rgb = cv2.bitwise_and(mask_img, mask_img, mask=mask.astype(np.uint8))
    return mask, road_lane_mask_rgb

def combine_ipm_imgs(img_file_list: list):
    assert(len(img_file_list)>0)
    img_list = []
    for img in img_file_list:
        if isinstance(img, str):
            img = cv2.imread(img)
        img_list.append(img)
    init_mask = np.zeros_like(img_list[0])
    for img in img_list:
        init_mask = np.logical_or(init_mask, img)
    return mask_to_img(init_mask)

def get_box_from_mask(mask: np.ndarray, x1y1x2y2: bool = True):
    coords = np.where(mask > 0)
    # 找到最小和最大的x和y坐标
    sorted_x = np.sort(coords[0])
    sorted_y = np.sort(coords[1])
    min_x = sorted_x[4]
    min_y = sorted_y[4]
    max_x = sorted_x[-5]
    max_y = sorted_y[-5]
    if x1y1x2y2:
        return np.array([min_x, min_y, max_x, max_y])
    else:
        return np.array([min_x, min_y, max_x-min_x, max_y-min_y])
