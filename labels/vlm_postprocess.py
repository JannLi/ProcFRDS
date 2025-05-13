import cv2
import os
import sys
import random
import numpy as np

vlm_path = sys.argv[1]
img_path = os.path.join(vlm_path, 'images')
light_path = os.path.join(vlm_path, 'masks1')
cross_path = os.path.join(vlm_path, 'masks2')

names = os.listdir(light_path)
for i, name in enumerate(names):
    if not 'front' in name:
        continue
    print(name+'----------------'+str(i)+'/'+str(len(names)))
    light_mask = cv2.imread(os.path.join(light_path, name))
    light_mask = cv2.cvtColor(light_mask, cv2.COLOR_BGR2GRAY)
    cross_mask = cv2.imread(os.path.join(cross_path, name))
    cross_mask = cv2.cvtColor(cross_mask, cv2.COLOR_BGR2GRAY)
    speed = random.choice([30, 40])
    speed_info = '推荐车速：'+str(speed)+'km/h'
    if len(np.unique(light_mask)) > 5:
        light_info = '红绿灯：红灯，注意停车等待'
    else:
        light_info = '红绿灯：无'
    if len(np.unique(cross_mask)) > 3:
        cross_info = '交叉路口：有交叉路口，减速慢行'
    else:
        cross_info = '交叉路口：无'
    with open(os.path.join(img_path, name.replace('png', 'txt')), 'r') as f:
        cones = f.read()
    cone_info = '锥桶等障碍物：'
    if 'left' in cones and 'right' in cones and 'middle' in cones:
        cone_info += '前方有锥桶，注意避让'
    elif 'left' in cones and 'right' in cones:
        cone_info += '左右侧有锥桶，注意避让'
    elif 'left' in cones and 'middle' in cones:
        cone_info += '左侧与中间有锥桶，注意避让'
    elif 'middle' in cones and 'right' in cones:
        cone_info += '右侧与中间有锥桶，注意避让'
    elif 'left' in cones:
        cone_info += '左侧有锥桶，注意避让'
    elif 'right' in cones:
        cone_info += '右侧有锥桶，注意避让'
    elif 'middle' in cones:
        cone_info += '中间有锥桶，注意避让'
    else:
        cone_info += '无'
    prompt = '。 '.join([speed_info, light_info, cone_info, cross_info])
    with open(os.path.join(vlm_path, 'prompt.txt'), 'a') as f:
        f.write(os.path.join(img_path, name)+' '+prompt)
        f.write('\n')