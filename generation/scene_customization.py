import bpy
import random
import os
import numpy as np
from generation.scenes import RoadsideSceneCreater
from generation.objects import duplicate_link_obj, add_paved_path_generation

class GrassBrickSceneCreater(RoadsideSceneCreater):
    def __init__(self, elements_path, blend_path, layout_path='', line_width=0.21):
        super().__init__(elements_path, blend_path,layout_path, line_width)
        self.scene_length = 30
        self.slot_width = 3
        self.slot_height = 5
        self.sidewalk = False
        self.curb = False
        if self.slot_type == 'slant':
            self.slot_angle = random.choice([np.pi/4, 5*np.pi/18, 11*np.pi/36, np.pi/3])
    
    def build_base_scene(self):
        # 地面，道路，车位
        self.build_ground()
        road_mat_suffix = 'grounds/concrete/'
        road_locations = [[0, 0, 0.01]]
        road_rotations = [[0, 0, 0]]
        self.build_road(road_mat_suffix, road_locations, road_rotations, road_reach=self.scene_length)

        if self.slot_type == 'parallel':
            slot_width, slot_height = self.slot_height, self.slot_width
        else:
            slot_width, slot_height = self.slot_width, self.slot_height
        slot_locations, slot_rotations = [], []
        x_loc, y_loc = slot_width-self.line_width, self.road_length/2+self.slot_height/2
        for i in range(int(self.scene_length/slot_width)):
            slot_locations += [[-self.scene_length/2+i*x_loc, -y_loc, 0], [-self.scene_length/2+i*x_loc, y_loc, 0]]
            slot_rotations += [[0, 0, 0], [0, 0, np.pi]]
        slot_mat_suffix = random.choice(['slots/outdoor', 'grounds/brick'])
        marking_pts = self.build_slots(slot_mat_suffix, slot_width, slot_height, slot_locations, slot_rotations, slot_thickness=0.06)

        path_name = add_paved_path_generation(os.path.join(self.elements_path, 'path/Paved_Path_Generator1.3.blend'), 'random', slot_height, self.scene_length, (0, y_loc))
        path = bpy.data.objects[path_name]
        # duplicate_link_obj(path, [(-self.scene_length/2+0.32, -y_loc, 0)], [path.rotation_euler])
        
        if self.slot_type == 'parallel':
            return np.pi
        return self.slot_angle