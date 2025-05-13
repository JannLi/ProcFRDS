import math
import numpy as np
import bpy
import sys
from scenariogeneration import xodr

def process_road(odr):
    road_collection = bpy.data.collections.get('roads')
    if not road_collection:
        road_collection = bpy.data.collections.get('lanes')
    roads_odr, roads_real = [], []
    for i in range(len(road_collection.objects)):
        road = road_collection.objects[i]
        road_loc = road.location
        road_rot = road.rotation_euler
        road_size = road.dimensions
        if road_rot[-1] == 0:
            xstart, y_start = road_loc[0]-road_size[0]/2, road_loc[1]
        elif round(road_rot[-1], 2) == 1.57:
            xstart, y_start = road_loc[0], road_loc[1]-road_size[0]/2
        planview = xodr.PlanView(x_start=xstart, y_start=y_start, h_start=road_rot[-1])
        planview.add_geometry(xodr.Line(road_size[0]))
        rm = xodr.RoadMark(xodr.RoadMarkType.solid, 0.2)
        centerlane_1 = xodr.Lane(a=road_loc[1])
        centerlane_1.add_roadmark(rm)
        lanesec_1 = xodr.LaneSection(0, centerlane_1)

        # add a driving lane
        lane2_1 = xodr.Lane(a=road_size[1]/2)
        lane2_1.add_roadmark(rm)
        lanesec_1.add_left_lane(lane2_1)

        lane3_1 = xodr.Lane(a=road_size[1]/2)
        lane3_1.add_roadmark(rm)
        lanesec_1.add_right_lane(lane3_1)

        # add parking lanes
        lane4_1 = xodr.Lane(a=6, lane_type=xodr.LaneType.parking)
        lane4_1.add_roadmark(rm)
        lane5_1 = xodr.Lane(a=6, lane_type=xodr.LaneType.parking)
        lane5_1.add_roadmark(rm)
        lanesec_1.add_left_lane(lane4_1)
        lanesec_1.add_right_lane(lane5_1)

        ## finalize the road
        lanes_1 = xodr.Lanes()
        lanes_1.add_lanesection(lanesec_1)

        road1 = xodr.Road(i+1, planview, lanes_1)

        odr.add_road(road1)
        roads_odr.append(road1)
        roads_real.append(road)
    odr.adjust_roads_and_lanes()
    return odr, roads_odr, roads_real

def add_obj(obj, road_odr, road_real, name, slot_angle=np.pi/2, line_width=0.2):
    loc = obj.location
    size = obj.dimensions
    obj_length = size[1]
    if name == 'parking_slot':
        obj_type = xodr.ObjectType.parkingSpace
        obj_width = size[0]-line_width
        if slot_angle == np.pi/2:
            obj_hdg = obj.rotation_euler[-1]+slot_angle
        elif slot_angle == np.pi:
            obj_hdg = obj.rotation_euler[-1]
            obj_width, obj_length = obj_length, obj_width
        else:
            obj_hdg = slot_angle
            obj_width = (obj_width-size[1]/math.tan(obj_hdg))*math.sin(obj_hdg)
    else:
        obj_type = xodr.ObjectType.obstacle
        obj_width = size[0]
        obj_hdg = obj.rotation_euler[-1]+np.pi/2
    x = loc[0]-road_real.location[0]+road_real.dimensions[0]/2
    y = loc[1]-road_real.location[1]
    jerseyBarrier = xodr.Object(
        x,
        y,
        height=size[2],
        width=obj_width,
        length=obj_length,
        zOffset=0,
        hdg=obj_hdg,
        orientation=xodr.Orientation.positive,
        Type=obj_type,
        name=name,
    )
    # if slot_angle != np.pi/2:
    #     jerseyBarrier = xodr.Object(
    #         x,
    #         y,
    #         dynamic=xodr.Dynamic.no,
    #         height=size[2],
    #         zOffset=0,
    #         hdg=obj.rotation_euler[-1]+np.pi/2,
    #         orientation=xodr.Orientation.positive,
    #         Type=obj_type,
    #         name=name,
    #     )
    #     outline = xodr.Outline(id=1)
    #     outline.add_corner(xodr.CornerRoad(x-size[0]/2, y-size[1]/2, 0, 0))
    #     outline.add_corner(xodr.CornerRoad(x-size[0]/2+size[1]/math.tan(slot_angle), y+size[1]/2, 0, 0))
    #     outline.add_corner(xodr.CornerRoad(x+size[0]/2, y+size[1]/2, 0, 0))
    #     outline.add_corner(xodr.CornerRoad(x+size[0]/2-size[1]/math.tan(slot_angle), y-size[1]/2, 0, 0))
    #     jerseyBarrier.add_outline(outline)
    road_odr.add_object(jerseyBarrier)

def match_road(roads, obj):
    close_dis = 10000
    close_index = 0
    obj_loc = list(obj.location[:2])
    for i in range(len(roads)):
        road = roads[i]
        dis = np.linalg.norm(np.array([list(road.location[:2]), obj_loc]))
        if dis < close_dis:
            close_dis = dis
            close_index = i
    return close_index

def process_collection(collct_name, road_odr, road_real, name, slot_angle=np.pi/2):
    collect = bpy.data.collections.get(collct_name)
    if not collect or len(collect.objects) <= 0:
        return
    for obj in collect.objects:
        if collct_name == 'locks' and 'closed' in obj.name:
            obj_name = 'unlock'
        elif collct_name == 'obstacles':
            if 'fence' in obj.name or 'waterhouse' in obj.name or 'barrier' in obj.name:
                obj_name = 'barrier'
            elif 'shopping_cart' in obj.name:
                obj_name = 'shopping_cart'
            elif 'rubbish_bin' in obj.name:
                obj_name = 'rubbish_bin'
            else:
                obj_name = name
        else:
            obj_name = name
        road_index = match_road(road_real, obj)
        add_obj(obj, road_odr[road_index], road_real[road_index], obj_name, slot_angle)

def process_cars(road_odr, road_real):
    for obj in bpy.data.objects:
        if 'clean' in obj.name:
            road_index = match_road(road_real, obj)
            add_obj(obj, road_odr[road_index], road_real[road_index], 'player')

def blend2xodr(blend_path, xodr_path, slot_angle=np.pi/2):
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    odr = xodr.OpenDrive("parking")
    odr, roads_odr, roads_real = process_road(odr)
    process_collection('slots', roads_odr, roads_real, 'parking_slot', slot_angle)
    process_collection('pillars', roads_odr, roads_real, 'lizhu')
    process_collection('limiters', roads_odr, roads_real, 'stopper')
    process_collection('locks', roads_odr, roads_real, 'lock')
    process_collection('walls', roads_odr, roads_real, 'wall')
    process_collection('obstacles', roads_odr, roads_real, 'cone')
    process_cars(roads_odr, roads_real)

    odr.write_xml(xodr_path)   

if __name__ == '__main__':
    blend_path = sys.argv[1]
    xodr_path = sys.argv[2]
    slot_angle = sys.argv[3]
    slot_angle = math.radians(float(slot_angle))
    
    blend2xodr(blend_path, xodr_path, slot_angle)
    bpy.ops.wm.open_mainfile(filepath=blend_path)