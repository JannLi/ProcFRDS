import bpy
import numpy as np
import os
import random
import json
from generation.objects import add_object_from_file, add_cars, add_forbiddens, add_limiters, add_locks, add_obstacles, gen_slot, duplicate_link_obj, uv_editing
from generation.materials import add_material_from_file, change_material_random, copy_material_from_obj, add_puddle2
from generation.background import hdri_maker_bg_dome
from generation.rendering import cal_marking_pts
from utils.addon_modifiers import add_deformations, add_dust

class RoadsideSceneCreater():
    def __init__(self, elements_path, blend_path, layout_path='', line_width=0.21, no_slots='0'):
        self.line_width = line_width
        self.elements_path = elements_path
        self.blend_path = blend_path
        self.material_path = os.path.join(self.elements_path, 'materials')
        self.hdri_path = os.path.join(self.elements_path, 'hdri/roadside')
        self.ground_type = random.choice(['brick', 'concrete'])
        self.puddle = random.choice([False, False, True])
        self.road_length = 6
        self.scene_length = 100
        self.sidewalk = random.choice([True, False])
        self.no_slots = no_slots
        if self.no_slots == '1':
            self.sidewalk = False
        self.curb = random.choice([True, False])
        self.slot_type = random.choice(['paralllel', 'vertical', 'slant'])
        self.slot_line_type = random.choice(['open', 'closed', 'half_closed'])
        self.slot_width = 3
        self.slot_height = 6
        self.slot_angle = np.pi/2
        if self.slot_type == 'slant':
            self.slot_angle = np.pi/3
        if len(layout_path) > 0:
            with open(layout_path, 'r') as f:
                self.layout = json.load(f)
        else:
            self.layout = dict()

    def add_layout_info(self, obj, name):
        obj_info = dict()
        obj_info['location'] = list(obj.location)
        obj_info['rotation'] = list(obj.rotation_euler)
        obj_info['size'] = list(obj.dimensions)
        if name == 'slots':
            obj_info['slot_angle'] = self.slot_angle
            if self.slot_type == 'parallel':
                obj_info['slot_angle'] = np.pi
            if self.slot_type == 'slant':
                obj_info['size'] = [self.slot_width, self.slot_height, 0]
        obj_list = self.layout.setdefault(name, [])
        obj_list.append(obj_info)

    def build_ground(self):
        if self.ground_type == 'indoor':
            add_object_from_file(os.path.join(self.elements_path, 'grounds/spare_ground.blend'), (0, 0, 0), (0, 0, 0), '')
        else:
            bpy.ops.wm.open_mainfile(filepath=os.path.join(self.elements_path, 'grounds/spare_ground.blend'))
        ground = bpy.data.objects['spare_ground']
        ground.dimensions = (self.scene_length, self.scene_length, 0)
        if self.ground_type == 'brick':
            materal_path_suffix = 'grounds/brick/'
        elif self.ground_type == 'concrete':
            materal_path_suffix = 'grounds/concrete/'
        elif self.ground_type == 'grass':
            materal_path_suffix = 'grounds/grass/'
        elif self.ground_type == 'indoor':
            materal_path_suffix = 'grounds/indoor/'
        else:
            materal_path_suffix = ''
        if len(materal_path_suffix) > 0:
            mat_name = change_material_random(ground, os.path.join(self.material_path, materal_path_suffix))
            uv_editing(ground, (2, 2))
            if self.puddle:
                add_puddle2(bpy.data.materials[mat_name], 0.5)
        camera = bpy.data.objects.get('Camera')
        if not camera:
            bpy.ops.object.camera_add()

    def build_road(self, road_mat_suffix, locations, rotations, road_length=None, collection_name='roads', road_name='', road_reach=100):
        add_object_from_file(os.path.join(self.elements_path, 'roads/spare_road.blend'), (0, 0, 0.01), (0, 0, 0), collection_name)
        road = bpy.data.objects['spare_road']
        if len(road_name) > 0:
            road.name = road_name
        if road_length == None:
            road.scale[1] = self.road_length/road.dimensions[1]
        else:
            road.scale[1] = road_length/road.dimensions[1]

        road.scale[0] = road_reach/road.dimensions[0]
        if len(road_mat_suffix) > 0:
            mat_name = change_material_random(road, os.path.join(self.material_path, road_mat_suffix))
            uv_editing(road, (2, 2))
            if self.puddle:
                add_puddle2(bpy.data.materials[mat_name], 0.5)
        duplicate_link_obj(road, locations, rotations)
        bpy.data.objects.remove(road)

    def build_slots(self, slot_mat_suffix, slot_width, slot_height, locations, rotations, slot_thickness=0.005):
        slot, marking_pts = gen_slot(self.slot_angle, slot_width, slot_height, self.slot_line_type, 0.2, thickness=slot_thickness)
        if len(slot_mat_suffix) > 0:
            mat_name = change_material_random(slot, os.path.join(self.material_path, slot_mat_suffix))
            uv_editing(slot, (2, 2))
            # if self.puddle:
            #     add_puddle2(bpy.data.materials[mat_name], 0.5)
        duplicate_link_obj(slot, locations, rotations)
        bpy.data.objects.remove(slot)
        return marking_pts        

    def build_base_scene(self):
        # 地面，道路，车位
        self.build_ground()
        road_mat_suffix = 'grounds/concrete/'
        roads = self.layout.get('roads')
        if roads:
            road_locations = []
            road_rotations = []
            road_sizes = []
            for road in roads:
                road_locations.append(road['location'])
                road_rotations.append(road['rotation'])
                road_sizes.append(road['size'])
            road_reach, road_length = road_sizes[0][:2]

        else:
            road_locations = [[0, 0, 0.01]]
            road_rotations = [[0, 0, 0]]
            road_reach, road_length = self.scene_length, self.road_length
        if self.no_slots == '1':
            road_mat_suffix = ''
        self.build_road(road_mat_suffix, road_locations, road_rotations, road_length=road_length, road_reach=road_reach)
        if self.no_slots == '0':
            lanes = self.layout.get('lanes')
            if lanes:
                for lane in lanes:
                    add_object_from_file(os.path.join(self.elements_path, 'roads/road1.blend'), lane['location'], lane['rotation'], 'lanes')
            else:
                    add_object_from_file(os.path.join(self.elements_path, 'roads/road1.blend'), (0, 0, 0.01), (0, 0, 0), 'lanes')

        slot_locations, slot_rotations = [], []
        slots = self.layout.get('slots')
        if slots:
            slot_angle = slots[0]['slot_angle']
            self.slot_angle = slot_angle
            slot_width, slot_height, _ = slots[0]['size']
            if slot_angle == np.pi:
                # slot_height, slot_width = slots[0]['size']
                self.slot_angle = np.pi/2
            for slot in slots:
                slot_locations.append(slot['location'])
                slot_rotations.append(slot['rotation'])
        else:
            if self.slot_type == 'parallel':
                slot_width, slot_height = self.slot_height, self.slot_width
            else:
                slot_width, slot_height = self.slot_width, self.slot_height
            x_loc, y_loc = slot_width-self.line_width, self.road_length/2+slot_height/2
            for i in range(int(self.scene_length/slot_width)):
                slot_locations += [[-self.scene_length/2+i*x_loc, -y_loc, 0], [-self.scene_length/2+i*x_loc, y_loc, 0]]
                slot_rotations += [[0, 0, 0], [0, 0, np.pi]]
            slot_angle = self.slot_angle
            if self.slot_type == 'parallel':
                slot_angle = np.pi
        slot_mat_suffix = 'slots/outdoor/'
        marking_pts = self.build_slots(slot_mat_suffix, slot_width, slot_height, slot_locations, slot_rotations)

        return slot_angle
    
    def modify_base_scene(self):
        pass

    def add_static_objs(self):
        y_loc_sidewalk = 0
        if self.sidewalk:
            self.curb = True
            sidewalk_mat_suffix = 'grounds/brick/'
            if self.slot_type == 'parallel':
                y_loc_sidewalk = self.road_length/2+self.slot_width+0.2+2
            elif self.slot_type == 'vertical':
                y_loc_sidewalk = self.road_length/2+self.slot_height+0.2+2
            else:
                y_loc_sidewalk = self.road_length/2+self.slot_height+0.2+2
            sidewalk_locations = [[0, y_loc_sidewalk, 0.01], [0, -y_loc_sidewalk, 0.01]]
            sidewalk_rotations = [[0, 0, 0], [0, 0, 0]]
            self.build_road(sidewalk_mat_suffix, sidewalk_locations, sidewalk_rotations, 4, 'sidewalks', road_name='sidewalk')

        y_loc_curb = 0
        if self.curb:
            if self.slot_type == 'parallel':
                y_loc_curb = self.road_length/2+self.slot_width+0.1
            else:
                y_loc_curb = self.road_length/2+self.slot_height+0.1
            for i in range(int(-self.scene_length/2), int(self.scene_length/2)):
                for j in [1, -1]:
                    location = [i, j*y_loc_curb, 0]
                    rotation = [0, 0, 0]
                    add_object_from_file(os.path.join(self.elements_path, 'curbs/curb1.blend'), location, rotation, 'curbs')

        if y_loc_sidewalk > 0:
            y_loc_hedge = y_loc_sidewalk+2+0.8
        elif y_loc_curb > 0:
            y_loc_hedge = y_loc_curb+0.1+0.8
        else:
            if self.slot_type == 'parallel':
                y_loc_hedge = self.road_length/2+self.slot_width+0.8
            else:
                y_loc_hedge = self.road_length/2+self.slot_height+0.8
        hedge_locations, hedge_rotations = [], []
        for i in range(int(self.scene_length/self.slot_width)):
            hedge_locations += [[-self.scene_length/2+i*(self.slot_width+0.2), y_loc_hedge, 0], [-self.scene_length/2+i*(self.slot_width+0.2), -y_loc_hedge, 0]]
            hedge_rotations += [[0, 0, 0], [0, 0, 0]]
        for j in range(len(hedge_locations)):
            location, rotation = hedge_locations[j], hedge_rotations[j]
            add_object_from_file(os.path.join(self.elements_path, 'hedges/hedge1.blend'), location, rotation, 'hedges')

    def set_background(self):
        hdri_name = random.choice([os.path.basename(name) for name in os.listdir(self.hdri_path)])
        hdri_maker_bg_dome(self.hdri_path, hdri_name.split('.')[0], 6)

    def export_layout(self):
        for obj in bpy.data.objects:
            # if obj.name.startswith('spare_ground'):
            #     self.add_layout_info(obj, 'ground')
            if obj.name.startswith('spare_road'):
                self.add_layout_info(obj, 'roads')
            elif 'pillar' in obj.name:
                self.add_layout_info(obj, 'pillars')
            # elif 'hedge' in obj.name:
            #     self.add_layout_info(obj, 'hedges')
            elif 'slot' in obj.name:
                self.add_layout_info(obj, 'slots')
            # elif 'curb' in obj.name:
            #     self.add_layout_info(obj, 'curbs')
            elif 'road' in obj.name:
                self.add_layout_info(obj, 'lanes')
        return self.layout

    def save_base_scene(self):
        bpy.ops.wm.save_mainfile(filepath=self.blend_path)

class SurfaceSceneCreater(RoadsideSceneCreater):
    def __init__(self, elements_path, blend_path, layout_path='', line_width=0.2):
        super().__init__(elements_path, blend_path, layout_path, line_width)
        self.hdri_path = os.path.join(self.elements_path, 'hdri/surface')
        # self.slots_array = random.choice(['single', 'double'])
        self.slots_array = 'single'
        self.grass_slot = False
        self.road_count = 4
    
    def build_base_scene(self):
        # if self.ground_type == 'grass':
        #     # 创建一个新的场景
        #     new_scene = bpy.data.scenes.new("NewEmptyScene")
        #     bpy.context.window.scene = new_scene
        #     for obj in bpy.data.objects:
        #         bpy.data.objects.remove(obj)
        #     gen_partical_grass(100, 10000, (0.2, 0.4, 0, 1))
        self.build_ground()
        if self.slot_type == 'parallel':
            road_dist = self.slot_width
        else:
            road_dist = self.slot_height
        if self.slots_array == 'double':
            road_dist *= 2
        road_locations, road_rotations, road_sizes = [], [], []
        slot_locations, slot_rotations = [], []
        roads = self.layout.get('roads')
        slots = self.layout.get('slots')
        if roads and slots:
            slot_angle = slots[0]['slot_angle']
            self.slot_angle = slot_angle
            for road in roads:
                road_locations.append(road['location'])
                road_rotations.append(road['rotation'])
                road_sizes.append(road['size'])
            slot_width, slot_height, _ = slots[0]['size']
            if slot_angle == np.pi:
                # slot_height, slot_width, _ = slots[0]['size']
                self.slot_angle = np.pi/2  
            for slot in slots:
                slot_locations.append(slot['location'])
                slot_rotations.append(slot['rotation'])
            road_reach, road_length = road_sizes[0][:2]           
        else:
            road_reach, road_length = self.scene_length, self.road_length 
            if self.slot_type == 'parallel':
                slot_width, slot_height = self.slot_height, self.slot_width
            else:
                slot_width, slot_height = self.slot_width, self.slot_height
            for i in range(-int(self.road_count/2), int(self.road_count/2)):
                road_locations += [[0, i*(road_dist+road_length), 0.01]]
                road_rotations += [[0, 0, 0]]
                for j in range(int(self.scene_length/slot_width)):
                    slot_locations += [[-self.scene_length/2+j*(slot_width-self.line_width), i*(road_dist+road_length)-(slot_height+road_length)/2, 0]]
                    slot_rotations += [[0, 0, 0]]
                if self.slots_array == 'double':
                    for k in range(int(self.scene_length/slot_width)):
                        slot_locations += [[-self.scene_length/2+k*(slot_width-self.line_width), i*(road_dist+road_length)+(slot_height+road_length)/2, 0]]
                        slot_rotations += [[0, 0, np.pi]]
            slot_angle = self.slot_angle
            if self.slot_type == 'parallel':
                slot_angle = np.pi
        road_mat_suffix = random.choice(['grounds/concrete/', ''])
        self.build_road(road_mat_suffix, road_locations, road_rotations, road_length=road_length, road_reach=road_reach)
        
        slot_mat_suffix = 'slots/outdoor/'
        marking_pts = self.build_slots(slot_mat_suffix, slot_width, slot_height, slot_locations, slot_rotations)
        return slot_angle

    def add_static_objs(self):
        if self.grass_slot:
            grass_name = random.choice(os.listdir(os.path.join(self.elements_path, 'slots/grass')))
            grass_path = os.path.join(self.elements_path, 'slots/grass/'+grass_name)
            slots = bpy.data.collections.get('slots')
            grass_slots = random.sample(slots.objects.keys(), 30)
            for name in grass_slots:
                slot = bpy.data.objects.get(name)
                add_object_from_file(grass_path, list(slot.location), list(slot.rotation_euler), 'grass', (slot.dimensions[0]-2*self.line_width, slot.dimensions[1]-2*self.line_width, 0))

class UndergroundSceneCreater(RoadsideSceneCreater):
    def __init__(self, elements_path, blend_path, layout_path='', line_width=0.2):
        super().__init__(elements_path, blend_path, layout_path, line_width)
        self.ground_type = 'indoor'
        self.road_length = 7
        self.slot_height = 5.501
        self.slot_width = 2.501

    def build_ceiling(self):
        bpy.ops.wm.open_mainfile(filepath=os.path.join(self.elements_path, 'base/inner_base1.blend'))

    def build_base_scene(self):
        self.build_ceiling()
        self.build_ground()
        slot_mat_suffix = 'slots/indoor/'
        slot_locations, slot_rotations = [], []

        roads = self.layout.get('lanes')
        slots = self.layout.get('slots')
        pillars = self.layout.get('pillars')
        if roads and slots and pillars:
            for road in roads:
                add_object_from_file(os.path.join(self.elements_path, 'roads/road1.blend'), road['location'], road['rotation'], 'lanes')
            slot_width, slot_height, _ = slots[0]['size']
            slot_angle = slots[0]['slot_angle']
            self.slot_angle = slot_angle
            if slot_angle == np.pi:
                # slot_height, slot_width = slots[0]['size']
                self.slot_angle = np.pi/2  
            for slot in slots:
                slot_locations.append(slot['location'])
                slot_rotations.append(slot['rotation'])
            for pillar in pillars:
                add_object_from_file(os.path.join(self.elements_path, 'pillars/white_pillar.blend'), pillar['location'], pillar['rotation'], 'pillars')           

        else:
            slots_count = random.randint(3, 6)
            left_edge, right_edge = -21, 23
            if self.slot_type == 'parallel':
                slot_width, slot_height = self.slot_height, self.slot_width
            else:
                slot_width, slot_height = self.slot_width, self.slot_height
            pillar_dist_col = slots_count*slot_width+1-(slots_count-1)*self.line_width  
            road1_loc = (0, left_edge+slot_height+self.road_length/2+0.5, 0)
            road2_loc = (0, right_edge-slot_height-self.road_length/2-0.5, 0)
            add_object_from_file(os.path.join(self.elements_path, 'roads/road1.blend'), road1_loc, (0, 0, 0), 'lanes')
            add_object_from_file(os.path.join(self.elements_path, 'roads/road1.blend'), road2_loc, (0, 0, 0), 'lanes')

            pillar_loc_x = 40
            pillar_loc_ys = [road1_loc[1]-(self.road_length/2+0.5), road1_loc[1]+(self.road_length/2+0.5), road2_loc[1]-(self.road_length/2+0.5), road2_loc[1]+(self.road_length/2+0.5)]
            while pillar_loc_x >= -52:
                for j in range(len(pillar_loc_ys)):
                    y = pillar_loc_ys[j]
                    add_object_from_file(os.path.join(self.elements_path, 'pillars/white_pillar.blend'), (pillar_loc_x, y, 0), (0, 0, 0), 'pillars')
                    slot_loc_x = pillar_loc_x
                    if self.slot_type == 'slant':
                        slot_loc_x -= (-1)**j*slot_height/np.tan(self.slot_angle)/2
                    for i in range(slots_count):
                        slot_locations += [[slot_loc_x+0.5+slot_width/2*(2*i+1)-i*self.line_width, y-(-1)**j*slot_height/2, 0]]
                        slot_rotations += [[0, 0, np.pi*j]]
                pillar_loc_x -= pillar_dist_col
            slot_angle = self.slot_angle
            if self.slot_type == 'parallel':
                slot_angle = np.pi
        marking_pts = self.build_slots(slot_mat_suffix, slot_width, slot_height, slot_locations, slot_rotations)
    
        return slot_angle

    def add_static_objs(self):
        pass

    def set_background(self):
        pass

    def modify_base_scene(self):
        light = random.randint(5, 100)
        light_mat = bpy.data.materials.get('light')
        if light_mat:
            light_mat.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = light

        roads = bpy.data.collections.get('lanes')
        line_material_path = os.path.join(self.material_path, 'slots/indoor')
        line_mat = random.choice(os.listdir(line_material_path))
        line_mat_file = os.path.join(line_material_path, line_mat)
        for road in roads.objects:
            add_material_from_file(line_mat_file, line_mat.split('.')[0], road)

class SceneModifier():
    def __init__(self, elements_path, out_blend_path, slot_angle=np.pi/2, target_slot=[], limiter=True, car=True, obstacle=True, lock=True, forbidden=False, people=True, speed_bump=True, left_car=False, right_car=False):
        self.elements_path = elements_path
        self.out_blend_path = out_blend_path
        self.slot_angle = slot_angle
        self.limiter = limiter
        self.car = car
        self.obstacle = obstacle
        self.lock = lock
        self.forbidden = forbidden
        self.people = people
        self.speed_bump = speed_bump
        self.slot_type = 'slant'
        if self.slot_angle == np.pi:
            self.slot_type = 'parallel'
        elif self.slot_angle == np.pi/2:
            self.slot_type = 'vertical' 
        self.slots_info = dict()
        self.init_slot_info()
        self.target_slot = target_slot
        self.left_car = left_car
        self.right_car = right_car

    def init_slot_info(self):
        slot_angle = self.slot_angle
        if self.slot_angle == np.pi:
            slot_angle /= 2
        for slot in bpy.data.collections['slots'].objects:
            slot.location[2] = 0
            self.slots_info[slot.name] = dict()
            self.slots_info[slot.name]['slot_type'] = self.slot_type
            self.slots_info[slot.name]['pts'] = cal_marking_pts(slot, slot_angle)
            self.slots_info[slot.name]['locked'] = 0
            self.slots_info[slot.name]['occupied'] = 0
            self.slots_info[slot.name]['forbidden'] = 0
            self.slots_info[slot.name]['have_stopper'] = False
            self.slots_info[slot.name]['parking_lock'] = False

    def modify_static_objs(self):
        ground = bpy.data.objects.get('spare_ground')
        if ground:
            add_dust(ground)
        roads_co = bpy.data.collections.get('roads')
        if roads_co:
            add_dust(roads_co.objects[0])
        slots_co = bpy.data.collections.get('slots')
        if slots_co:
            add_deformations(slots_co.objects[0])
        road_line = bpy.data.objects.get('road1')
        if road_line:
            add_deformations(road_line)

    def add_dynamic_objs_on_roads(self):
        roads_co = bpy.data.collections.get('roads')
        if not roads_co:
            roads_co = bpy.data.collections.get('lanes')
        if self.speed_bump:
            for road in roads_co.objects:
                locx = [random.uniform(-road.dimensions[0]/2, -1), random.uniform(1, road.dimensions[0]/2)]
                locy = road.location[1]
                for x in locx:
                    add_object_from_file(os.path.join(self.elements_path, 'speed_bumps/speed_bump1.blend'), (x, locy, 0), (0, 0, 0), 'speed_bumps')

    def add_dynamic_objs_in_slots(self):
        slot_angle = self.slot_angle
        if self.slot_angle == np.pi:
            slot_angle = 0
        slots = bpy.data.collections['slots'].objects.keys()
        if self.limiter:
            limiter_path = os.path.join(self.elements_path, 'limiters')
            add_limiters(limiter_path, slots, slot_angle, self.slots_info)
        
        if len(self.target_slot) > 0:
            slots.remove(self.target_slot)
            slot_name_info = self.target_slot.split('.')
            if len(slot_name_info) == 1:
                left_slot = ''
                right_slot = self.target_slot+'.001'
            else:
                slot_base_name = slot_name_info[0]
                target_index = int(slot_name_info[1])
                left_slot = slot_base_name+'.'+str(target_index-2).zfill(3)
                right_slot = slot_base_name+'.'+str(target_index+2).zfill(3)

        car_slots = []
        if self.car:
            arranged_slots = []
            if self.left_car and left_slot in slots:
                arranged_slots.append(left_slot)
            if self.right_car and right_slot in slots:
                arranged_slots.append(right_slot)
            car_number = random.randint(10, 20)
            print(car_number)
            if len(slots) >= car_number:
                car_path = os.path.join(self.elements_path, 'cars/common_cars')
                car_slots = add_cars(car_path, car_number, slots, slot_angle, self.slots_info, arranged_slots)

        clear_slots = list(set(slots)-set(car_slots))
        lock_slots = []
        if self.lock:
            arranged_slots = []
            lock_number = random.randint(5, 20)
            if len(clear_slots) >= lock_number:
                lock_path = os.path.join(self.elements_path, 'locks')
                lock_slots = add_locks(lock_path, lock_number, clear_slots, slot_angle, self.slots_info, arranged_slots)

        clear_slots = list(set(clear_slots)-set(lock_slots))
        obstacle_slots, corn_slots, no_parking_slots = [], [], []
        if self.obstacle:
            arranged_slots = []
            obstacle_number = random.randint(5, 20)
            if len(clear_slots) >= obstacle_number:
                obstacle_path = os.path.join(self.elements_path, 'obstacles')
                obstacle_slots, corn_slots, no_parking_slots = add_obstacles(obstacle_path, obstacle_number, clear_slots, slot_angle, self.slots_info, arranged_slots)

        clear_slots = list(set(clear_slots)-set(obstacle_slots)-set(corn_slots)-set(no_parking_slots))
        forbidden_slots = []
        if self.forbidden:
            arranged_slots = []
            forbidden_number = random.randint(5, 10)
            if len(clear_slots) >= forbidden_number:
                forbidden_path = os.path.join(self.elements_path, 'forbiddens')
                forbidden_slots = add_forbiddens(forbidden_path, forbidden_number, clear_slots, self.slots_info, arranged_slots)

        return [car_slots, lock_slots, obstacle_slots, forbidden_slots]

    def save_output_scene(self):
        bpy.ops.wm.save_mainfile(filepath=self.out_blend_path)