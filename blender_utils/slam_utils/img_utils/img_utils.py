import cv2
import numpy as np
from .mask_utils import img_to_mask
from typing import Union
import os
import imgaug as ia
from imgaug  import augmenters as iaa

def combine_car_shadow_bg2(mask_img, shadow_img, bg_img, bg_img_mask = None, shadow_scale = 1.0):
    mask_only_objetcs = mask_img[:,:,3].copy()
    mask_only_shadow = shadow_img[:,:,3].copy()
    mask_only_shadow[mask_only_objetcs>0] = 0
    mask_only_shadow_ori = mask_only_shadow.copy()
    thres = 200
    mask_only_shadow[mask_only_shadow_ori>=thres] = (mask_only_shadow[mask_only_shadow_ori>=thres] - thres) * 2.0
    mask_only_shadow[mask_only_shadow_ori<thres] = 0
    # cv2.imwrite('test0_shadow.jpg', mask_only_shadow)
    shadow_img[:, :, 3][mask_only_shadow_ori>0] = mask_only_shadow[mask_only_shadow_ori>0]
    if bg_img_mask is not None:
        shadow_img[:,:,3][bg_img_mask] = 0
    return combine_rgba_and_rgb(shadow_img, bg_img, alpha_scale=1.0)

def combine_rgba_and_rgb(rgba_img, rgb_img, alpha_scale=1.0):
    mask = rgba_img[:,:,3].copy()
    # cv2.imwrite('temp/mask.png', mask)
    mask = np.expand_dims(mask,axis=2)
    mask = mask.repeat(3, axis=2)
    if mask.dtype == 'uint8':
        mask = mask.astype('float') / 255.0
    elif mask.dtype == 'uint16':
        mask = mask.astype('float')/65535.0
    # mask[mask<0.5] = 0
    mask = mask * alpha_scale
    mask_not = 1-mask
    img_back = rgb_img.astype('float') * mask_not
    img_front = rgba_img[:, :, :3].copy()
    # img_front[:, :, 2] = 255
    img_front = img_front.astype('float') * mask
    new_img = img_back+img_front
    new_img = new_img.astype(np.uint8)
    return new_img

def combine_car_shadow_bg(car_img, shadow_img, bg_img, bg_img_mask = None, shadow_scale = 0.6, alpha_scale=1.0):
    mask_car = car_img[:,:,3].copy()
    mask_car_shadow = shadow_img[:,:,3].copy()
    mask_car[mask_car<=64] = 0
    mask_car_rgb = np.expand_dims(mask_car.copy(),axis=2)
    mask_car_rgb = mask_car_rgb.repeat(3, axis=2)
    shadow_img[:, :, :3][img_to_mask(mask_car_rgb)] = car_img[:, :, :3][img_to_mask(mask_car_rgb)]
    shadow_img = shadow_img.astype('float')
    # mask_shadow = np.logical_and(mask_car_shadow>0, (mask_car==0))
    # shadow_img[:, :, 3][mask_shadow] *= shadow_scale
    shadow_img[:, :, 3][np.logical_and(mask_car_shadow < 255, mask_car_shadow > 64)] *= shadow_scale
    shadow_img[:, :, :3][img_to_mask(mask_car_rgb)] = car_img[:, :, :3][img_to_mask(mask_car_rgb)]
    shadow_img = shadow_img.astype('uint8')
    # shadow_img[:, :, 3][mask_shadow] *= shadow_scale
    # shadow_img.astype('float')[:, :, 3][np.logical_and(mask_car_shadow<255, mask_car_shadow>64)] *= shadow_scale
    if bg_img_mask is not None:
        shadow_img[:,:,3][bg_img_mask] = 0
    new_img = combine_rgba_and_rgb_for_car2temp(shadow_img, bg_img, alpha_scale)
    return new_img

def combine_rgba_and_rgb_for_car2temp(rgba_img, rgb_img, alpha_scale=1.0):
    mask = rgba_img[:,:,3].copy()
    mask[mask<=64] = 0
    # mask[mask==128] = 200
    # cv2.imwrite('temp/mask.png', mask)
    mask = np.expand_dims(mask,axis=2)
    mask = mask.repeat(3, axis=2)
    if mask.dtype == 'uint8':
        mask = mask.astype('float') / 255.0
    elif mask.dtype == 'uint16':
        mask = mask.astype('float')/65535.0
    mask = mask * alpha_scale
    mask_not = 1-mask
    img_back = rgb_img.astype('float') * mask_not
    img_front = rgba_img[:, :, :3].copy()
    # img_front[:, :, 2] = 255
    img_front = img_front.astype('float') * mask
    new_img = img_back+img_front
    new_img = new_img.astype(np.uint8)
    return new_img

# from top left
def points_anticlockwise(points: np.array):
    points = np.array(points)
    center_point = np.mean(points, axis=0)
    part1_points = points[points[:, 0] >= center_point[0], :]
    part2_points = points[points[:, 0] < center_point[0], :]
    if len(part1_points) == 2 and part1_points[0][1] == part1_points[1][1]:
        part1_points = part1_points[np.argsort(-part1_points[:, 0])]
    else:
        part1_points = part1_points[np.argsort(-part1_points[:, 1])]
    if len(part2_points) == 2 and part2_points[0][1] == part2_points[1][1]:
        part2_points = part2_points[np.argsort(part2_points[:, 0])]
    else:
        part2_points = part2_points[np.argsort(part2_points[:, 1])]
    sorted_points = np.concatenate((part1_points, part2_points), axis=0)
    return sorted_points


# some funtions for video
def video_to_img(video_file_path: str, save_path: str = 'temp/video_img'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cap = cv2.VideoCapture(video_file_path)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(save_path, str(i).zfill(4) + '.png'), frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return save_path

def img_to_video(img_folder_path, save_path, fps=30):
    # 获取文件夹中所有图片的列表
    images = [img for img in os.listdir(img_folder_path) if img.endswith(".jpeg") or img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(img_folder_path, images[0]))
    height, width, layers = frame.shape

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(img_folder_path, image)))

    cv2.destroyAllWindows()
    video.release()
    
def img_path_list_to_video(img_path_list, save_path, fps=30):
    # 获取文件夹中所有图片的列表
    frame = cv2.imread(img_path_list[0])
    height, width, layers = frame.shape

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for image in img_path_list:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    
def combine_ipm_imgs(imgs_list: list):
    img_init = np.zeros_like(imgs_list[0])
    for i in range(len(imgs_list)):
        img_new = imgs_list[i].copy()
        img_new[np.all(img_init != [0, 0, 0], axis=2)] = 0
        img_init = img_init + img_new
    return img_init


def img_to_backenv(img: Union[str, np.array]):
    if isinstance(img, str):
        env_img = cv2.imread(img)
    else:
        env_img = img.copy()
    alpha = 0.6  # 降低对比度
    beta = 100   # 增加亮度
    env_img = cv2.convertScaleAbs(env_img, alpha=alpha, beta=beta)

    # # 将图像从BGR颜色空间转换为HSV颜色空间
    env_img = cv2.cvtColor(env_img, cv2.COLOR_BGR2HSV)

    # 调整饱和度通道的值
    # env_img[:, :, 0] = env_img[:, :, 0] * 0.15
    env_img[:, :, 1] = env_img[:, :, 1] * 0.4
    # 随机化亮度 (simulate noise)
    # value_mask = np.random.rand(int(env_img.shape[0]/4), int(env_img.shape[1]/4))*0.1 + 0.95
    # value_mask = cv2.resize(value_mask, (env_img.shape[1], env_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # env_img = env_img.astype(float)
    # env_img[int(env_img.shape[0]/2):, :, 2] = env_img[int(env_img.shape[0]/2):, :, 2] * value_mask[:int(env_img.shape[0]/2):, :]
    # env_img[:, :, 2][env_img[:, :, 2] >= 255] = 255
    # env_img = env_img.astype(np.uint8)         
    # env_img[:, :, 2] = cv2.GaussianBlur(env_img[:, :, 2], (5, 5), 0)
    # 将图像从HSV颜色空间转换回BGR颜色空间
    env_img = cv2.cvtColor(env_img, cv2.COLOR_HSV2BGR)
    return env_img


class ImgAugTool():
    def __init__(self):
        pass
    def add_motion_blur(self, img):
        sometimes_strong = lambda aug: iaa.Sometimes(1.0, aug)
        seq = iaa.Sequential([            
                              sometimes_strong(iaa.MotionBlur(k=15))])
        img = seq(image=img.copy())
        return img