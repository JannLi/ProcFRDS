import cv2
import pickle
import numpy as np
import pandas as pd
import sys

# with open('../test/layouts_new/roadside_parallel/roadside_parallel.pkl', 'rb') as f:
#     data = pickle.load(f, encoding='bytes')
#     f.close()
# grid = data['grid_data']

# pic = grid*255
# cv2.imwrite('test.png', pic)

# csv_file = sys.argv[1]
# df = pd.read_csv(csv_file, encoding='utf-8')
# img = np.zeros((1000, 1000))
# for _, row in df.iterrows():
#     loc = [float(row['x']), float(row['y'])]
#     loc = [int(i*10)+500 for i in loc]
#     cv2.circle(img, loc, 3, (255, 255, 255), 3)
# cv2.imwrite('test.png', img)

def distort_image(fisheye_path, camera_mat, dist_coeff):
    raw_img = cv2.imread(fisheye_path)
    map1, map2 = cv2.initUndistortRectifyMap(camera_mat, dist_coeff, np.eye(3, 3), camera_mat, (3840, 2160), cv2.CV_16SC2)
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_mat, dist_coeff, np.eye(3, 3), camera_mat, (1920, 1536), cv2.CV_32FC1)
    distort_img = cv2.remap(raw_img, map1, map2, cv2.INTER_LINEAR)
    return distort_img

camera_mat = np.array([[1905.4635010979264, 0, 1920], [0, 1905.2555526601791, 1080], [0, 0, 1]])
dist_coeff = [3.8491571681249135, 1.4827764013208131, 2.56763932376e-05, 7.3760046561999999e-06, 0.047911284359424398, 4.2148976885941263, 2.7748518175286554, 0.30055794005929243]

img_path = '/home/sczone/disk1/share/3d/blender_slots/ProcFRPS/test/results/others/1/results/front_120_0.png'
img2 = distort_image(img_path, camera_mat, np.array(dist_coeff))
cv2.imwrite('test2.png', img2)