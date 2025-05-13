import bpy
import json
import numpy as np
import mathutils
import csv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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

def process_slots(json_file, target_pt):
    target = Point(target_pt)
    with open(json_file, 'r') as f:
        data = json.load(f)
    slot_pts = []
    for key in data:
        slot = data[key]['pts']
        slot_poly = Polygon(slot)
        if not slot_poly.contains(target):
        # if slot['forbidden']:
            slot_pts.append(slot)
    return slot_pts

def process_collects(collect_name):
    collect = bpy.data.collections.get(collect_name)
    bbox_pts = []
    if not collect:
        return bbox_pts
    for obj in collect.objects:
        bbox = get_ground_projection_bounding_box(obj)
        bbox_pts.append(bbox)
    return bbox_pts

def process_objs(marking_pts_json):
    locs_info = dict()
    locs_info['pillars'] = process_collects('pillars')
    locs_info['limiters'] = process_collects('limiters')
    locs_info['walls'] = process_collects('walls')
    locs_info['locks'] = process_collects('locks')
    locs_info['obsctacles'] = process_collects('obstacles')
    locs_info['cars'] = []
    for obj in bpy.data.objects:
        if obj.name.startswith('body'):
            bbox = get_ground_projection_bounding_box(obj)
            locs_info['cars'].append(bbox)
    locs_info['slots'] = process_slots(marking_pts_json, [7.75362, 0.234739])
    return locs_info

def process_csv(locs_info, csv_file):
    # start = [12.5981, -6.89906, np.pi]
    start = [13.5981, -7.89906, 3.14159]
    end = [7.75362, 1.234739, -np.pi/2]
    count = 0
    locs = []
    vertex = []
    for key in locs_info:
        objs = locs_info[key]
        for obj in objs:
            for item in obj[::-1]:
                locs += item
            vertex.append(len(obj))
            count += 1
    res = start + end + [count] + vertex + locs
    # res = ','.join(str(i) for i in res)
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(res)
    

if __name__ == '__main__':
    blend_path = '/home/sczone/disk1/share/3d/blender_slots/code/results/3.blend'
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    locs_info = process_objs('/home/sczone/disk1/share/3d/blender_slots/code/results/3.json')
    process_csv(locs_info, '/home/sczone/disk1/share/3d/blender_slots/pathpred/AutomatedValetParking-main/BenchmarkCases/3.csv')