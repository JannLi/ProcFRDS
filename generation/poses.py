import bpy
import random
import numpy as np


def gen_entering_ego_pose(slot_center, slot_angle, slot_size, slot_type):
    xy = np.array(slot_center[:2])
    w, h = slot_size
    h_unit = h/4
    w_unit = w/24
    angle1 = np.array([np.cos(slot_angle), np.sin(slot_angle)])
    angle2 = np.array([np.sin(slot_angle), -np.cos(slot_angle)])
    parked_rotation = [0, 0, slot_angle+random.uniform(-np.pi/18, np.pi/18)]
    parked_location = xy+random.uniform(-0.2, 0.2)
    parked_pose = np.array(parked_rotation+parked_location.tolist()+[slot_center[2]])
    if slot_type == 'vertical':
        rot1 = [0, 0, slot_angle+np.pi/18+random.uniform(-np.pi/18, np.pi/18)]
        loc1 = xy+h_unit*angle1+w_unit*angle2
        pose1 = np.array(rot1+loc1.tolist()+[slot_center[2]])
        rot2 = [0, 0, slot_angle-np.pi/18+random.uniform(-np.pi/18, np.pi/18)]
        loc2 = xy+h_unit*angle1-w_unit*angle2
        pose2 = np.array(rot2+loc2.tolist()+[slot_center[2]])
        rot3 = [0, 0, slot_angle+np.pi/6+random.uniform(-np.pi/18, np.pi/18)]
        loc3 = xy+2*h_unit*angle1+3*w_unit*angle2
        pose3 = np.array(rot3+loc3.tolist()+[slot_center[2]])
        rot4 = [0, 0, slot_angle-np.pi/6+random.uniform(-np.pi/18, np.pi/18)]
        loc4 = xy+2*h_unit*angle1-3*w_unit*angle2
        pose4 = np.array(rot4+loc4.tolist()+[slot_center[2]])
        rot5 = [0, 0, slot_angle+np.pi/3+random.uniform(-np.pi/18, np.pi/18)]
        loc5 = xy+3*h_unit*angle1+6*w_unit*angle2
        pose5 = np.array(rot5+loc5.tolist()+[slot_center[2]])
        rot6 = [0, 0, slot_angle-np.pi/3+random.uniform(-np.pi/18, np.pi/18)]
        loc6 = xy+3*h_unit*angle1+6*w_unit*angle2        
        pose6 = np.array(rot6+loc6.tolist()+[slot_center[2]])
        poses = [parked_pose, pose1, pose2, pose3, pose4, pose5, pose6]
    elif slot_type == 'parallel':
        rot1 = [0, 0, slot_angle+np.pi/3+random.uniform(-np.pi/18, np.pi/18)]
        loc1 = xy+h_unit*angle1+12*w_unit*angle2
        pose1 = np.array(rot1+loc1.tolist()+[slot_center[2]])
        rot2 = [0, 0, slot_angle+np.pi/6+random.uniform(-np.pi/18, np.pi/18)]
        loc2 = xy+2*h_unit*angle1+18*w_unit*angle2
        pose2 = np.array(rot2+loc2.tolist()+[slot_center[2]])
        rot3 = [0, 0, slot_angle+random.uniform(-np.pi/18, np.pi/18)]
        loc3 = xy+3*h_unit*angle1+24*w_unit*angle2
        pose3 = np.array(rot3+loc3.tolist()+[slot_center[2]])
        poses = [parked_pose, pose1, pose2, pose3]
    return poses
        
def gen_ego_poses(road_angle, road_center, road_width, road_len, occupied_slots):
    ego_poses = list()
    road_center = np.array(road_center)
    angle1 = np.array([np.cos(road_angle), np.sin(road_angle)])
    angle2 = np.array([np.sin(road_angle), -np.cos(road_angle)])
    start_center = road_center-road_len/2*angle1
    left_center = road_center+road_width/4*angle2 # road_angle: +x anticlock
    right_center = road_center-road_width/4*angle2
    start_left_center = left_center-road_len/2*angle1
    start_right_center = right_center-road_len/2*angle1
    moving_dist = 0
    while moving_dist < road_len:
        delta_angle = random.uniform(-np.pi/18, np.pi/18)
        delta_dis = random.uniform(-0.5, 0.5)
        moving_dist += 2 + delta_dis
        center_loc = start_center+moving_dist*angle1
        center_pose = np.array([0, 0, road_angle+delta_angle, center_loc[0], center_loc[1], 0])

        right_loc = start_right_center+moving_dist*angle1
        right_pose = np.array([0, 0, road_angle+delta_angle, right_loc[0], right_loc[1], 0])

        left_loc = start_left_center+moving_dist*angle1
        left_pose = np.array([0, 0, road_angle+delta_angle+np.pi, left_loc[0], left_loc[1], 0])
        ego_poses += [center_pose, right_pose, left_pose]

    clear_slots = list(set(bpy.data.collections['slots'].objects.keys())-set(occupied_slots))
    # clear_slots = [item for item in clear_slots if item.startswith('col0') or item.startswith('col1')]

    selected_slots = clear_slots
    for key in selected_slots:
        slot = bpy.data.objects[key]
        slot_angle = slot.rotation_euler[2] + np.pi/2
        if slot_angle == road_angle:
            slot_type = 'parallel'
        else:
            slot_type = 'vertical'
        entering_poses = gen_entering_ego_pose(list(slot.location), slot_angle, slot.dimensions[:2], slot_type)
        ego_poses += entering_poses
    return ego_poses

def get_ego_poses_for_road(road, occupied_slots):
    road_angle = road.rotation_euler[2]
    road_center = road.location[:2]
    road_len = road.dimensions[0]
    road_width = road.dimensions[1]
    if len(occupied_slots) > 0 and type(occupied_slots[0]) is list:
        occupied_slots = [item for sublist in occupied_slots for item in sublist]
    ego_poses = gen_ego_poses(road_angle, road_center, road_width, road_len, occupied_slots)
    return ego_poses