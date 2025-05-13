import numpy as np
import bpy
import time
import pickle
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

    return [min_x, max_x, min_y, max_y]

def world2array(loc, x_resolution, y_resolution, x_start, y_start):
    loc_x, loc_y = loc
    array_x = int((loc_x-x_start)/x_resolution)
    array_y = int((loc_y-y_start)/y_resolution)
    return [array_x, array_y]

def process_collects(grid_data, collect_name, x_resolution, y_resolution, x_start, y_start):
    collect = bpy.data.collections.get(collect_name)
    if collect:
        for obj in collect.objects:
            x_min, x_max, y_min, y_max = get_ground_projection_bounding_box(obj)
            x_min, y_min = world2array([x_min, y_min], x_resolution, y_resolution, x_start, y_start)
            x_max, y_max = world2array([x_max, y_max], x_resolution, y_resolution, x_start, y_start)
            grid_data[x_min:x_max, y_min:y_max] = 1
    return grid_data

def process_cars(grid_data, x_resolution, y_resolution, x_start, y_start):
    for obj in bpy.data.objects:
        if obj.name.endswith('clean'):
            x_min, x_max, y_min, y_max = get_ground_projection_bounding_box(obj)
            x_min, y_min = world2array([x_min, y_min], x_resolution, y_resolution, x_start, y_start)
            x_max, y_max = world2array([x_max, y_max], x_resolution, y_resolution, x_start, y_start)
            grid_data[x_min:x_max, y_min:y_max] = 1
    return grid_data

def main(args):
    bpy.ops.wm.open_mainfile(filepath=args.input_blend_path)
    res = dict()
    if 'parallel' in args.input_blend_path:
        map_range = (-50, 50, -6, 6)
        start = [(11.7, 2, 3.1415)]
        goal = (3.73, 4.45, 3.1556)
    elif 'vertical' in args.input_blend_path:
        map_range = (-50, 50, -9, 9)
        start = [(43.81, 1.5, 3.1415)]
        goal = (30.9, 6.95, 4.7329)        
    res['grid_data'] = np.zeros((int((map_range[1]-map_range[0])/0.1), int((map_range[3]-map_range[2])/0.1)))
    res['map_range'] = map_range
    res['start'] = start
    res['goal'] = goal
    res['grid_data'] = process_cars(res['grid_data'], 0.1, 0.1, map_range[0], map_range[2])
    res['grid_data'] = process_collects(res['grid_data'], 'limiters', 0.1, 0.1, map_range[0], map_range[2])
    res['grid_data'] = process_collects(res['grid_data'], 'locks', 0.1, 0.1, map_range[0], map_range[2])
    res['grid_data'] = process_collects(res['grid_data'], 'obstacles', 0.1, 0.1, map_range[0], map_range[2])
    with open(args.out_path, 'wb') as fo:
        pickle.dump(res, fo)
        fo.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='export griddata')
    parser.add_argument('--input_blend_path', help='input model path')
    parser.add_argument('--out_path', help='output data path')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)
