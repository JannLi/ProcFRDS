
import sys
sys.path.append('../')
import bpy
import bmesh
import uuid
import json
import time
import os
import copy
import cv2
import numpy as np
from mathutils import Vector
from datetime import datetime, timedelta
from utils.parking_utils import word2image_pts, euler2quaternion
from utils.json2json import get_json_parameters
from generation.objects import add_object_from_file
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# def get_bev_contour_points(obj):
#     if obj.mode != 'EDIT':
#         bpy.context.view_layer.objects.active = obj
#         bpy.ops.object.mode_set(mode='EDIT')
#     bm = bmesh.from_edit_mesh(obj.data)
#     contour_pts_set = set()
#     contour_pts_list = list()
#     for edge in bm.edges:
#         v1, v2 = edge.verts
#         v1_co = obj.matrix_world @ v1.co
#         v2_co = obj.matrix_world @ v2.co
#         if abs(v1_co.z-v2_co.z) < 0.0001:
#             if not (v1_co.x, v1_co.y) [0, 0]d([v2_co.x, v2_co.y])
#             contour_pts_set.add((v1_co.x, v1_co.y))
#             contour_pts_set.add((v2_co.x, v2_co.y))
#     bpy.ops.object.mode_set(mode='OBJECT')
#     return contour_pts_list

def cal_speed(pose_data, cur_index):
    if cur_index == 0:
        return [0, 0, 0]
    cur_pose = pose_data[cur_index]
    pre_pose = pose_data[cur_index-1]
    move_time = cur_pose[-1]-pre_pose[-1]
    speed = (cur_pose[3:6]-pre_pose[3:6])/move_time
    return speed.tolist()

def cal_timestamp(time_now, time_interval):
    time_interval = timedelta(seconds=time_interval)
    new_time = time_now + time_interval
    return new_time.timestamp()

def get_bev_contour_points(obj, col_name):
    arrow_poly_index = dict()
    arrow_poly_index['forward'] = np.array([1, 0, 3, 2, 6, 5, 4])
    arrow_poly_index['forward_left'] = np.array([1, 0, 10, 9, 13, 12, 11, 5, 7, 8, 6, 3, 4, 2])
    arrow_poly_index['forward_right'] = np.array([1, 0, 3, 4, 2, 6, 8, 7, 5, 10, 9, 13, 12, 11])
    arrow_poly_index['forward_left_right'] = np.array([1, 0, 3, 6, 2, 10, 14, 12, 8, 17, 16, 20, 19, 18, 9, 13, 15, 11, 5, 7, 4])
    arrow_poly_index['forward_uturn'] = np.array([1, 0, 24, 23, 27, 26, 25, 18, 20, 22, 21, 19, 17, 16,
                                                 10, 5, 6, 2, 3, 4, 8, 11, 13, 15, 14,12, 9, 7])
    arrow_poly_index['left'] = np.array([1, 0, 5, 7, 8, 6, 3, 4, 2])
    arrow_poly_index['left_right'] = np.array([1, 0, 2, 6, 4, 9, 13, 11, 8, 12, 14, 10, 5, 7, 3])
    arrow_poly_index['left_uturn'] = np.array([1, 0, 26, 28, 29, 27, 22, 25, 21, 18, 20, 24, 23, 19, 17,
                                              16, 10, 5, 6, 2, 4, 3, 8, 11, 13, 15, 14,12, 9, 7])
    arrow_poly_index['right'] = np.array([1, 0, 2, 4, 3, 6, 8, 7, 5])
    arrow_poly_index['uturn'] = np.array([0, 1, 29, 31, 33, 35, 37, 39, 41, 43, 46, 48, 50, 49, 47, 45, 44,
                                         42, 40, 38, 36, 34, 32, 30, 5, 6, 2, 4, 3, 8, 10, 12, 14, 16, 18,
                                         20, 22, 24, 26, 28, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7])
    arrow_poly_index['slot_open'] = np.array([1, 0, 6, 7, 5, 4, 2, 3])
    arrow_poly_index['slot_closed'] = np.array([1, 0, 6, 7, 3, 2, 4, 5])
    arrow_poly_index['slot_half_closed'] = np.array([1, 0, 10, 11, 7, 6, 9, 8, 2, 3, 4, 5])
    mesh = obj.data
    vertices = [v.co.xy for v in mesh.vertices]
    vertices = [[round(item[0], 3), round(item[1], 3)] for item in vertices]
    vertices = np.unique(np.array(vertices), axis=0)
    vertices = [obj.matrix_world @ Vector(list(item)+[0]) for item in vertices]
    vertices = np.array([item[:2] for item in vertices])
    if col_name in ['slots', 'ground_signs'] and (not obj.name.startswith('xforbidden')):
        arrow_name = obj.name.split('.')[0]
        if arrow_name in arrow_poly_index:
            vertices = vertices[arrow_poly_index[arrow_name]]
        approx = np.array([[item] for item in vertices])
    else:
        hull = ConvexHull(vertices)
        contour_vertices = vertices[hull.vertices]
        contour_vertices = np.array(contour_vertices, dtype=np.float32)
        approx = cv2.approxPolyDP(contour_vertices, 0.05, True)
    polygon = []
    for p in approx:
        polygon.append(p[0].tolist())
    return polygon

def gen_uuid2obj(collection_name, elements, uuid2obj_map, obj_type):   
    col = bpy.data.collections.get(collection_name)
    if col:
        categories = elements.setdefault(obj_type, dict())
        ids = categories.setdefault(collection_name, [])
        if collection_name == 'cars':
            obj_names = []
            for obj in bpy.data.objects:
                if 'clean' in obj.name:
                    obj_names.append(obj.name)
        else:
            obj_names = col.objects.keys()
        for name in obj_names:
            obj_uuid = str(uuid.uuid4())
            uuid2obj_map[obj_uuid] = name
            ids.append(obj_uuid)
    return elements, uuid2obj_map


def add_obj_to_map(elements, map_ele, annotations, instance_id, uuid2obj_map, slots_info):
    arrow_names = dict()
    arrow_names['forward'] = 'AheadOnlyA'
    arrow_names['left'] = 'TurnLeftA'
    arrow_names['right'] = 'TurnRightA'
    arrow_names['forward_left'] = 'AheadLeftA'
    arrow_names['forward_right'] = 'AheadRightA'
    arrow_names['uturn'] = 'UturnA'
    arrow_names['left_uturn'] = 'LeftUTurnA'
    arrow_names['forward_uturn'] = 'AheadUTurnA'
    arrow_names['forward_left_right'] = 'AheadLeftRightA'
    arrow_names['left_right'] = 'LeftRightA'
    arrow_names['xforbidden'] = 'NoParking'

    limiters_info = dict()

    static_elements = elements.get('static')
    for collection_name in static_elements:
        obj_uuids = static_elements.get(collection_name)
        map_ele[collection_name] = obj_uuids
        for obj_uuid in obj_uuids:
            name = uuid2obj_map[obj_uuid]
            obj = bpy.data.objects.get(name)
            instance_id += 1
            obj_dict = dict()
            obj_dict['uuid'] = obj_uuid
            obj_dict['polygon'] = get_bev_contour_points(obj, collection_name)
            obj_dict['key_points'] = [obj_dict['polygon'][0], obj_dict['polygon'][-1]]
            obj_dict['category'] = collection_name
            obj_dict['color'] = 'white'
            obj_dict['instance_id'] = instance_id
            if collection_name == 'ground_signs':
                obj_dict['category'] = arrow_names[name.split('.')[0]]
                obj_dict['diffult'] = False
                obj_dict['is_deprecated'] = False
            if collection_name == 'slots':
                obj_dict['category'] = slots_info[name]['slot_type']
                obj_dict['key_points'] = slots_info[name]['pts']
                obj_dict['parking_lock'] = slots_info[name]['parking_lock']
                obj_dict['no_parking_sign'] = slots_info[name]['occupied'] == 3
                obj_dict['difficult'] = False
                if name in slots_info:
                    info = slots_info[name].get('limiters_info')
                    if info:
                        for i in range(len(info['limiter_name'])):
                            limiter = info['limiter_name'][i]
                            limiters_info[limiter] = dict()
                            limiters_info[limiter]['key_points'] = info['limiter_pts'][i]
                            limiters_info[limiter]['instance_id'] = instance_id
                            limiters_info[limiter]['type'] = info['limiter_type']
            if collection_name == 'limiters':
                if name in limiters_info:
                    obj_dict['instance_id'] = limiters_info[name]['instance_id']
                    obj_dict['key_points'] = limiters_info[name]['key_points']
                    obj_dict['category'] = limiters_info[name]['type']
                    instance_id -= 1

            annotations.append(obj_dict)
    return map_ele, annotations, instance_id

def gen_map(slots_info, map_file):
    map_dict = dict()
    uuid2obj_map = dict()
    map_uuid = str(uuid.uuid4())
    map_dict['uuid'] = map_uuid
    map_dict['version'] = 'zone_map_v0.1.0'
    ele_dict = dict()
    map_dict['elements'] = ele_dict
    anno_list = list()
    map_dict['annotations'] = anno_list
    instance_id = 0
    elements = dict()
    static_col_names = ['slots', 'pillars', 'ground_signs', 'limiters', 'curbs', 'hedges', 'walls', 'speed_bumps']
    dynamic_col_names = ['cars', 'locks', 'obstacles', 'people']
    for name in static_col_names:
        elements, uuid2obj_map = gen_uuid2obj(name, elements, uuid2obj_map, 'static')
    for name in dynamic_col_names:
        elements, uuid2obj_map =  gen_uuid2obj(name, elements, uuid2obj_map, 'dynamic')
    ele_dict, anno_list, instance_id = add_obj_to_map(elements, ele_dict, anno_list, instance_id, uuid2obj_map, slots_info)
    with open(map_file, 'w') as f:
        json.dump(map_dict, f, indent=4)
    return elements, uuid2obj_map

def visible_judge(obj, sensor_param, g2e, ego2cams):
    mask_path = '/home/sczone/disk1/share/3d/24'
    box_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    box_corners = np.array(box_corners, dtype=np.float32)
    box_corners = box_corners.reshape(box_corners.shape[::-1])
    box_corners = g2e.dot(np.concatenate((box_corners, np.ones((1, 8), dtype=box_corners.dtype))))[:3, :]
    box_corners = box_corners.transpose(1, 0)[None, :]
    visible_cams = []
    for cam in sensor_param:
        visible_corners = 0
        mask = cv2.imread(os.path.join(mask_path, cam+'.png'))
        intrinsic = sensor_param[cam]['intrinsic']
        distort = sensor_param[cam]['distort']
        ego2cam = ego2cams[cam]
    
        img_pts = word2image_pts(box_corners, intrinsic, ego2cam, distort)
        h, w = [1536, 1920]
        if cam == 'front_120':
            h, w = [2160, 3840]
        for pt in img_pts[0]:
            if pt[0] >= h or pt[0] < 0 or pt[1] < 0 or pt[1] >= w:
                continue
            if cam == 'front_120':
                visible_corners += 1
            elif mask[int(pt[0])][int(pt[1])].sum == 0:
                visible_corners += 1
        if visible_corners > 1:
            visible_cams.append(cam)
    return visible_cams

def gen_obj_annotation(elements, uuid2obj_map, anno_list, sensor_param, ego_pose, ego2cams, slots_info):
    e2g = np.eye(4)
    e2g[:3, :3] = R.from_rotvec(ego_pose['pose']).as_matrix()
    e2g[:3, 3] = ego_pose['angular_rate']
    g2e = np.linalg.inv(e2g)

    dynamic_elements = elements.get('dynamic')
    for key in dynamic_elements.keys():
        uuids = dynamic_elements[key]
        for id in uuids:
            obj = bpy.data.objects.get(uuid2obj_map[id])
            if not obj:
                continue
            anno = dict()
            anno['uuid'] = id
            anno['timestamp'] = str(time.time())
            anno['pose'] = list(obj.location)
            anno['size'] = list(obj.dimensions)
            anno['pose'][-1] = anno['size'][-1]/2
            anno['rotation'] = list(obj.rotation_euler)
            anno['velocity'] = [0, 0, 0]
            anno['acceleration'] = [0, 0, 0]
            anno['angular_rate'] = [0, 0, 0]
            anno['category'] = key
            anno['instance_id'] = 0
            anno['track_id'] = 1000
            anno['visible_in_cam'] = 0
            obj_visible = visible_judge(obj, sensor_param, g2e, ego2cams)
            if len(obj_visible) > 0:
                anno['visible_in_cam'] = 1
                anno['visible_sensor_id'] = obj_visible
            if 'clean' in obj.name:
                anno['is_in_work'] = -1
                anno['is_tight'] = -1
                anno['vehicle_size'] = -1
                anno['occluder_type'] = -1
                anno['occlusion_fisheye'] = -1
                anno['occluder_type_fisheye'] = -1
                anno['special_category'] = -1
            anno['is_in_row'] = 0
            anno['truncation'] = 0
            anno['occlusion'] = 0
            if obj.name.startswith('shopping_cart'):
                anno['category'] = 'shopping_carts'
            if obj.name.startswith('corn'):
                anno['category'] = 'corns'
            if obj.name.startswith('rubbish_bin'):
                anno['category'] = 'rubbish_bin'
            if obj.name.startswith('barrier') or 'fence' in obj.name:
                anno['category'] = 'barriers'
            anno_list.append(anno)
    slots_uuids = elements['static']['slots']
    for slot_uuid in slots_uuids:
        slot_name = uuid2obj_map[slot_uuid]
        slot_info = slots_info[slot_name]
        anno = dict()
        anno['uuid'] = str(uuid.uuid4())
        anno['category'] = 'slot_status'
        anno['slot_uuid'] = slot_uuid
        anno['occupy'] = 0
        if slot_info.get('locked'):
            anno['occupy'] = slot_info.get('locked')
        anno['lock'] = False
        if slot_info.get('locked') == 2:
            anno['lock'] = True
        anno['truncation'] = 0
        anno['occlusion'] = 0
        anno['points_visible'] = -1
        anno_list.append(anno)       
    return anno_list

def get_ego_infos(pose_file):
    ego_poses = np.load(pose_file)
    out_poses = list()
    frame_uuid = dict()
    now_time = datetime.now()
    for i in range(len(ego_poses)):
        pose = ego_poses[i]
        ego = dict()
        ego['uuid'] = str(uuid.uuid4())
        ego['timestamp'] = str(cal_timestamp(now_time, pose[-1]))
        ego['pose'] = pose[3:6].tolist()
        ego['quat'] = euler2quaternion(pose[:3].tolist()).tolist()
        ego['velocity'] = cal_speed(ego_poses, i)
        ego['acceleration'] = [0, 0, 0]
        ego['angular_rate'] = pose[:3].tolist()
        out_poses.append(ego)
        frame_uuid[str(i)] = str(uuid.uuid4())
    return out_poses, frame_uuid

def gen_sensor_infos(param_path='/home/sczone/disk1/share/3d/24'):
    sensor = dict()
    sensor['uuid'] = str(uuid.uuid4())
    sensor['pre_frame_uuid'] = str(uuid.uuid4())
    sensor['next_frame_uuid'] = str(uuid.uuid4())
    sensor_list = list()
    sensor_param = dict()
    cam_names = ['fisheye_front', 'fisheye_left', 'fisheye_right', 'fisheye_rear', 'front_120']
    param_files = ['camera_1.json', 'camera_2.json', 'camera_3.json', 'camera_0.json', 'camera_front120.json']
    for i in range(len(cam_names)):
        name = cam_names[i]
        file = param_files[i]
        sensor_info = dict()        
        sensor_info['sensor_name'] = name
        sensor_info['timestamp'] = str(time.time())
        sensor_info['data_path'] = '../fisheye/'+name+'.png'
        sensor_info['snesor_info_uuid'] = str(uuid.uuid4())
        sensor_list.append(sensor_info)

        with open(os.path.join(param_path, file), 'r') as f:
            data = json.load(f)
        cam_params = get_json_parameters(data)
        param = dict()
        param['uuid'] = sensor_info['snesor_info_uuid']
        param['sensor_type'] = 'camera'
        param['intrinsic'] = cam_params['intrinsic']
        param['camera_model'] = 'fisheye'
        param['sensor2ego_rotation'] = cam_params['sensor2ego_rotation']
        param['sensor2ego_translation'] = cam_params['sensor2ego_translation']
        param['distort'] = cam_params['distort']
        param['height'] = cam_params['height']
        param['width'] = cam_params['width']
        param['channel'] = 3
        sensor_param[name] = param
    sensor['sensor_list'] = sensor_list
    return sensor, sensor_param

def gen_annotations(ego_pose, anno_file, uuid2obj_map, elements, frame_uuid, index, sensor_info, sensor_param, ego2cams, slots_info):
    anno_dict = dict()
    for info in sensor_info['sensor_list']:
        info['data_path'] = info['data_path'].replace('fisheye_', 'fisheye_'+str(index)+'_')
        info['data_path'] = info['data_path'].replace('front_120', 'front_120_'+str(index))
    cur_frame_uuid = frame_uuid[str(index)]
    anno_dict['uuid'] = cur_frame_uuid
    anno_dict['pre_frame_uuid'] = frame_uuid.get(str(index-1))
    anno_dict['next_frame'] = frame_uuid.get(str(index+1))
    anno_dict['ego_pose'] = ego_pose
    anno_dict['sensor'] = sensor_info
    annotations = list()
    anno_dict['annotations'] = gen_obj_annotation(elements, uuid2obj_map, annotations, sensor_param, ego_pose, ego2cams, slots_info)
    with open(anno_file, 'w') as f:
        json.dump(anno_dict, f, indent=4)

def main(args):
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    bpy.ops.wm.open_mainfile(filepath=args.input_blend_path)
    people_name = add_object_from_file('/home/sczone/disk1/share/3d/blender_slots/elements/people/business_019.blend', (0, 0, 0), (0, 0, 0), 'people')
    
    with open(args.slots_info_file, 'r') as f:
        slots_info = json.load(f)
    map_file_path = os.path.join(args.out_path, 'map.json')
    elements, uuid2obj_map = gen_map(slots_info, map_file_path)
    ego_poses, frame_uuid = get_ego_infos(args.pose_file)
    sensor_info, sensor_param = gen_sensor_infos()
    sensor_file = os.path.join(args.out_path, 'sensor.json')
    with open(sensor_file, 'w') as f:
        json.dump(sensor_param, f, indent=4)

    ego2cams = dict()
    for cam in sensor_param:
        cam2ego = np.eye(4)
        cam2ego[:3, :3] = Quaternion(sensor_param[cam]['sensor2ego_rotation']).rotation_matrix
        cam2ego[:3, 3] = sensor_param[cam]['sensor2ego_translation']
        ego2cam = np.linalg.inv(cam2ego)
        ego2cams[cam] = ego2cam

    people = bpy.data.objects[people_name]
    all_poses = np.load(args.pose_file)
    for i in range(len(ego_poses)):
        print(i)
        if len(all_poses[i]) >= 13:
            people_pose = list(all_poses[i][-6:]-all_poses[i][:6])
            people.location = people_pose[3:]
            people.rotation_euler = people_pose[:3]
        pose = ego_poses[i]
        anno_file = os.path.join(args.out_path, str(i)+'_annotation.json')
        frame_sensor_info = copy.deepcopy(sensor_info)
        gen_annotations(pose, anno_file, uuid2obj_map, elements, frame_uuid, i, frame_sensor_info, sensor_param, ego2cams, slots_info)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='blender2json labels')
    parser.add_argument('--input_blend_path', help='input model path')
    parser.add_argument('--pose_file', help='ego pose file')
    parser.add_argument('--slots_info_file', help='slots info file')
    parser.add_argument('--out_path', help='output label path')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)