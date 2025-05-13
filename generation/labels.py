import bpy
import json
import cv2
import numpy as np
import mathutils

def get_ground_projection_bounding_box(obj):
    # 获取对象的世界空间矩阵
    matrix_world = obj.matrix_world

    # 获取对象的本地空间边界框顶点
    local_bbox_corners = obj.bound_box[:]

    # 将边界框顶点从本地空间转换到世界空间
    world_bbox_corners = [matrix_world @ mathutils.Vector(corner) for corner in local_bbox_corners]
    ground_projection_points = [(x, y) for x, y, z in world_bbox_corners]

    # 计算投影点的最小和最大X和Z坐标
    min_x = min(p[0] for p in ground_projection_points)
    max_x = max(p[0] for p in ground_projection_points)
    min_y = min(p[1] for p in ground_projection_points)
    max_y = max(p[1] for p in ground_projection_points)

    return [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]

def world2pixel(pts, ratio=0.02, w=150, h=80):
    # ratio: 一像素代表的世界距离, w,h：世界距离长宽
    pixel_locs = []
    for pt in pts:
        pixel_pt = [int((pt[0]+w/2)/ratio), int((pt[1]+h/2)/ratio)]
        pixel_locs.append(pixel_pt)
    return np.array(pixel_locs)

def process_slots(json_file, img):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for key in data:
        slot = data[key]
        mark_pts = world2pixel(slot['pts'])
        if slot['forbidden']:
            color = (0, 0, 255)
        elif slot['locked'] or slot['occupied']:
            color = (0, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.fillPoly(img, [mark_pts], color)
    return img

def process_collects(img, collect_name):
    collect = bpy.data.collections[collect_name]
    for obj in collect.objects:
        bbox = get_ground_projection_bounding_box(obj)
        cv2.fillPoly(img, [world2pixel(bbox)], (255, 0, 0))
    return img

def process_objs(img):
    process_collects(img, 'pillars')
    process_collects(img, 'limiters')
    process_collects(img, 'walls')
    process_collects(img, 'locks')
    process_collects(img, 'obstacles')
    for obj in bpy.data.objects:
        if obj.name.startswith('body'):
            bbox = get_ground_projection_bounding_box(obj)
            cv2.fillPoly(img, [world2pixel(bbox)], (255, 0, 0))
    return img

if __name__ == '__main__':
    blend_path = '3.blend'
    json_path = '3.json'
    img = np.ones((4000, 7500, 3), dtype=np.uint8)*255
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    process_slots(json_path, img)
    process_objs(img)
    cv2.imwrite('8.png', img)