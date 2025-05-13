import bpy
import os
import random
import math
import numpy as np
from mathutils import Vector
from blender_utils.blr import add_whole_object, get_box_3d_in_scene
from blender_utils.slam_utils.matrix_utils import pose_to_4x4
from generation.materials import add_material_from_file

def join_objs(obj_names):
    bpy.ops.object.select_all(action='DESELECT')
    for name in obj_names:
        obj = bpy.data.objects.get(name)
        if obj:
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()

def uv_editing(obj, texture_cover=(2, 2)):
    bpy.context.view_layer.objects.active = obj
    # 确保对象是可编辑的网格
    if obj.type == 'MESH' and obj.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')
        
    # 切换到UV编辑模式
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.cube_project(scale_to_bounds=True)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 计算缩放因子的逻辑这里需要根据具体需求定义，例如：
    # 假设你有一个理想的纹理覆盖宽度和高度，与模型最大尺寸对比来计算
    # 这里仅作为示意，实际情况需要具体计算
    scale_factor = Vector(obj.dimensions[i]/texture_cover[i] for i in range(len(texture_cover)))

    # 遍历UV层并调整UV坐标
    for uv_layer in obj.data.uv_layers:
        for loop in obj.data.loops:
            uv_data = uv_layer.data[loop.index]
            original_uv = Vector((uv_data.uv.x, uv_data.uv.y))
            scaled_uv = original_uv * scale_factor
            uv_data.uv = scaled_uv

    # 返回物体模式
    bpy.ops.object.mode_set(mode='OBJECT')

def add_object_from_file(file_path, location, rotation, collection='', dimension=(), material_file='', material_name=''):
    pose = np.array(rotation+location)
    obj_name = os.path.basename(file_path).split('.')[0]
    obj_name = add_whole_object(file_path, obj_name, pose_to_4x4(pose), collect=collection)
    if len(dimension) > 0:
        obj = bpy.data.objects[obj_name]
        obj.dimensions = dimension
    if len(material_file) > 0:
        add_material_from_file(material_file, material_name, obj)
        uv_editing(obj)
    return obj_name


def add_arrows(road_angle, road_center, road_width, road_len, arrow_path, road_name):
    arrow_names = os.listdir(arrow_path)
    arrow_names.remove('xforbidden.blend')
    road_center = np.array(road_center)
    angle1 = np.array([np.cos(road_angle), np.sin(road_angle)])
    angle2 = np.array([np.sin(road_angle), -np.cos(road_angle)])
    start_center = road_center-road_len/2*angle1
    left_center = road_center+road_width/4*angle2 # road_angle: +x anticlock
    right_center = road_center-road_width/4*angle2
    start_left_center = left_center-road_len/2*angle1
    start_right_center = right_center-road_len/2*angle1
    count = 0
    arrow_height = 0
    if road_name.startswith('spare_road'):
        arrow_height = 0.01
    while count < int(road_len/3):
        left_arrow = random.choice(arrow_names+['xforbidden.blend'])
        if left_arrow.startswith('xforbidden'):
            center_dist = random.uniform(count*3+4, count*3+5)
            center_loc = start_center+center_dist*angle1
            center_pose = pose_to_4x4(np.array([0, 0, road_angle, center_loc[0], center_loc[1], arrow_height]))
            xforbidden_name = add_whole_object(os.path.join(arrow_path, left_arrow), left_arrow.split('.')[0], center_pose, plane_name=road_name, collect='ground_signs')
            xforbidden = bpy.data.objects[xforbidden_name]
            xforbidden.dimensions[0] = 0.95*road_width
            xforbidden.dimensions[1] = 0.95*road_width
            count += 3
        else:
            right_arrow = random.choice(arrow_names)
            left_dist = random.uniform(count*3+1, count*3+2)
            right_dist = random.uniform(count*3+1, count*3+2)
            left_loc = start_left_center+left_dist*angle1
            right_loc = start_right_center+right_dist*angle1
            left_pose = pose_to_4x4(np.array([0, 0, road_angle, left_loc[0], left_loc[1], arrow_height]))
            right_pose = pose_to_4x4(np.array([0, 0, road_angle+np.pi, right_loc[0], right_loc[1], arrow_height]))
            left_arrow_name = add_whole_object(os.path.join(arrow_path, left_arrow), left_arrow.split('.')[0], left_pose, plane_name=road_name, collect='ground_signs')
            right_arrow_name = add_whole_object(os.path.join(arrow_path, right_arrow), right_arrow.split('.')[0], right_pose, plane_name=road_name, collect='ground_signs')
            count += 1
        
def add_arrows_on_road(road, arrow_path):
    road_angle = road.rotation_euler[2]
    road_center = road.location[:2]
    road_len = road.dimensions[0]
    road_width = road.dimensions[1]
    road_name = road.name
    add_arrows(road_angle, road_center, road_width, road_len, arrow_path, road_name)

def add_object_in_slot(slot, obj_model, angle_bias=0, x_bias=0, y_bias=0, obj_name='body', collection=''):
    location = list(slot.location)
    rotation = list(slot.rotation_euler)
    rotation[-1] += np.pi/2+angle_bias
    location[0] += x_bias
    location[1] += y_bias
    location[2] = 0
    pose = np.array(rotation+location)
    obj_name = add_whole_object(obj_model, obj_name, pose_to_4x4(pose), collect=collection)
    return obj_name

def select_car(car_path):
    car_brand = random.choice(os.listdir(car_path))
    car_brand_path = os.path.join(car_path, car_brand)
    car_model = random.choice(os.listdir(car_brand_path))
    car_model_path = os.path.join(car_brand_path, car_model)
    files = os.listdir(car_model_path)
    for f in files:
        if f.endswith('.blend'):
            return os.path.join(car_model_path, f)

def change_slot_line_type(line_path, line_type):
    slots_collection = bpy.data.collections['slots']
    slot_names = slots_collection.objects.keys()
    new_names = dict()
    for ori_name in slot_names:
        slot = slots_collection.objects[ori_name]
        location = get_box_3d_in_scene(ori_name)
        pose = np.array(list(slot.rotation_euler)+list(np.average(location, axis=0)[:2])+[0.07])
        # bpy.context.scene.objects.unlink(slot)
        bpy.data.objects.remove(slot)
        model_name = add_whole_object(os.path.join(line_path, line_type+'.blend'), line_type, pose_to_4x4(pose), collect='slots')
        namelist = new_names.setdefault(ori_name.split('_')[0], [])
        namelist.append(model_name)
    for key in new_names:
        for name in new_names[key]:
            bpy.data.objects[name].name = key+'_'+name

def add_limiters(limiter_path, slots, slot_angle, slots_info):
    limiter_collection = bpy.data.collections.get('limiters')
    if limiter_collection:
        bpy.data.collections.remove(limiter_collection)
    limiter_file = random.choice(os.listdir(limiter_path))
    # limiter_file = 'rubber_block_limiter1.blend'
    limiter_name = limiter_file.split('.')[0]
    for l in slots:
        slot = bpy.data.objects[l]
        angle_bias = slot_angle-np.pi/2
        x = 2 * np.sin(slot.rotation_euler[2]-np.pi/2+slot_angle)
        y = -2 * np.cos(slot.rotation_euler[2]-np.pi/2+slot_angle)
        slots_info[l]['have_stopper'] = True
        limiter_info = dict()
        limiter_info['limiter_pts'] = []
        limiter_info['limiter_name'] = []
        if limiter_name.startswith('rubber_block'):
            x1, y1 = x+0.55*np.sin(slot_angle), y-0.55*np.cos(slot_angle)
            x2, y2 = x-0.55*np.sin(slot_angle), y+0.55*np.cos(slot_angle)
            limiter_locs = [[x1, y1], [x2, y2]]
        else:
            limiter_locs = [[x, y]]
        for loc in limiter_locs:
            new_limiter_name = add_object_in_slot(slot, os.path.join(limiter_path, limiter_file), angle_bias, x_bias=loc[0], y_bias=loc[1], obj_name=limiter_name, collection='limiters')       
            limiter = bpy.data.objects[new_limiter_name]
            limiter_corners = [limiter.matrix_world @ Vector(corner) for corner in limiter.bound_box]
            limiter_info['limiter_pts'].append([list(limiter_corners[4][:2]), list(limiter_corners[7][:2])])
            limiter_info['limiter_name'].append(new_limiter_name)
        if limiter_name.startswith('strip_metal'):
            limiter_info['limiter_type'] = 'metal'
        elif limiter_name.startswith('rubber_block'):
            limiter_info['limiter_type'] = 'rubber'
        else:
            limiter_info['limiter_type'] = 'cement'
        slots_info[l]['limiters_info'] = limiter_info

def add_cars(car_path, number, clear_slots, slot_angle, slots_info, arranged_slots):
    assert number <= len(clear_slots)
    car_slots = random.sample(clear_slots, number)
    car_slots = list(set(car_slots)|set(arranged_slots))
    for l in car_slots:
        slot = bpy.data.objects[l]
        car_model_path = select_car(car_path)
        car_name = os.path.basename(car_model_path).split('.')[0]
        angle_bias = random.uniform(-np.pi/18, np.pi/18)-np.pi/2+slot_angle
        new_car_name = add_object_in_slot(slot, car_model_path, angle_bias, random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), obj_name=car_name, collection='cars')
        slots_info[l]['occupied'] = 1
    return car_slots

def add_locks(lock_path, number, clear_slots, slot_angle, slots_info, arranged_slots):
    assert number <= len(clear_slots)
    lock_slots = random.sample(clear_slots, number)
    lock_slots = list(set(lock_slots)|set(arranged_slots))
    for l in lock_slots:
        slot = bpy.data.objects[l]
        angle_bias = -np.pi/2+slot_angle
        lock = random.choice(os.listdir(lock_path))
        new_lock_name = add_object_in_slot(slot, os.path.join(lock_path, lock), angle_bias=angle_bias, obj_name=lock.split('.')[0], collection='locks')
        slots_info[l]['parking_lock'] = True
        if not 'closed' in lock:
            slots_info[l]['locked'] = 2
    return lock_slots

def add_obstacles(obstacle_path, number, clear_slots, slot_angle, slots_info, arranged_slots):
    selected_slots = random.sample(clear_slots, number)
    selected_slots = list(set(selected_slots)|set(arranged_slots))
    corn_slots = []
    obstacle_slots = []
    no_parking_slots = []
    for l in selected_slots:
        slot = bpy.data.objects[l]
        x = -2.5 * np.sin(slot.rotation_euler[2]-np.pi/2+slot_angle)
        y = 2.5 * np.cos(slot.rotation_euler[2]-np.pi/2+slot_angle)
        obstacle = random.choice(os.listdir(obstacle_path))
        if obstacle.startswith('corn'):
            corn_slots.append(l)
            slots_info[l]['occupied'] = 2
        elif obstacle.startswith('no_parking'):
            no_parking_slots.append(l)
            slots_info[l]['occupied'] = 3
        else:
            obstacle_slots.append(l)
            slots_info[l]['occupied'] = 4
        angle_bias = 0
        if slot_angle == np.pi:
            angle_bias = np.pi/2
        new_obstable_name = add_object_in_slot(slot, os.path.join(os.path.join(obstacle_path, obstacle)), angle_bias, x_bias=x, y_bias=y, obj_name=obstacle.split('.')[0], collection='obstacles')
    return obstacle_slots, corn_slots, no_parking_slots

def add_forbiddens(forbidden_path, number, clear_slots, slots_info, arranged_slots):
    assert number <= len(clear_slots)
    forbidden_slots = random.sample(clear_slots, number)
    forbidden_slots = list(set(forbidden_slots)|set(arranged_slots))
    forbidden = random.choice(os.listdir(forbidden_path))
    for l in forbidden_slots:
        slot = bpy.data.objects[l]
        add_object_in_slot(slot, os.path.join(forbidden_path, forbidden), obj_name=forbidden.split('.')[0], collection='forbiddens')
        slots_info[l]['forbidden'] = True 
    return forbidden_slots

def array_copy(obj, x, y, count):
    bpy.ops.object.modifier_add(type='ARRAY')
    array_modifier = obj.modifiers['Array']
    array_modifier.count = count
    array_modifier.relative_offset_displace[0] = x
    array_modifier.relative_offset_displace[1] = y
    bpy.ops.object.modifier_apply(modifier='Array')
    return obj

def cut_out(main_object, cutter_object):
    bpy.context.view_layer.objects.active = main_object
    main_object.select_set(True)
    bool_modifier = main_object.modifiers.new("BooleanModifier", 'BOOLEAN')
    bool_modifier.object = cutter_object
    bool_modifier.operation = 'DIFFERENCE'
    bpy.ops.object.modifier_apply(modifier=bool_modifier.name)
    bpy.data.objects.remove(cutter_object, do_unlink=True)

def create_poly_plane(poly_pts, poly_name='Polygon'):
    faces = range(len(poly_pts))
    mesh = bpy.data.meshes.new(poly_name)
    mesh.from_pydata(poly_pts, [], [faces])
    mesh.update()
    return mesh

def create_poly_obj(poly_pts, obj_name, thickness):
    mesh = create_poly_plane(poly_pts)
    for obj in bpy.data.objects:
        if obj.name in ['Light', 'Cube']:
            bpy.data.objects.remove(obj)
    bpy.ops.mesh.primitive_cube_add()
    obj_out = bpy.data.objects['Cube']
    obj_out.data = mesh
    obj_out.name = obj_name

    bpy.context.view_layer.objects.active = obj_out
    obj_out.select_set(True)
    solidify_mod = obj_out.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_mod.thickness = thickness
    solidify_mod.use_even_offset = True
    solidify_mod.offset = 0
    bpy.ops.object.modifier_apply(modifier=solidify_mod.name)
    return obj_out

def gen_slot(angle, width, height, line_type, line_width=0.2, collection='slots', thickness=0.005):
    if len(collection) > 0:
        new_collection = bpy.data.collections.get(collection)
        if not new_collection:
            new_collection = bpy.data.collections.new(name=collection)
            bpy.context.scene.collection.children.link(new_collection)
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection]
    
    pt0_out = [height/2/np.tan(angle)+width/2, height/2]
    pt1_out = [height/2/np.tan(angle)-width/2, height/2]
    pt2_out = [-height/2/np.tan(angle)-width/2, -height/2]
    pt3_out = [-height/2/np.tan(angle)+width/2, -height/2]
    
    height_in = height-2*line_width
    width_in = width-2*line_width/np.sin(angle)
    pt0_in = [height_in/2/np.tan(angle)+width_in/2, height_in/2]
    pt1_in = [height_in/2/np.tan(angle)-width_in/2, height_in/2]
    pt2_in = [-height_in/2/np.tan(angle)-width_in/2, -height_in/2]
    pt3_in = [-height_in/2/np.tan(angle)+width_in/2, -height_in/2]

    pt_out_open0 = [pt0_out[0]-line_width/np.sin(angle), pt0_out[1]]
    pt_out_open1 = [pt1_out[0]+line_width/np.sin(angle), pt1_out[1]]

    pt_out_half0 = (3/4*np.array(pt0_out)+1/4*np.array(pt1_out)).tolist()
    pt_out_half1 = (1/4*np.array(pt0_out)+3/4*np.array(pt1_out)).tolist()
    pt_in_half0 = [pt_out_half0[0], pt_out_half0[1]-line_width]
    pt_in_half1 = [pt_out_half1[0], pt_out_half1[1]-line_width]

    if line_type == 'closed':
        out_pts = [pt0_out, pt1_out, pt2_out, pt3_out]
        in_pts = [pt0_in, pt1_in, pt2_in, pt3_in]
        obj_out = create_poly_obj(np.array([pt+[0] for pt in out_pts]), 'slot_closed', thickness)
        obj_in = create_poly_obj(np.array([pt+[0] for pt in in_pts]), 'slot_closed_cutter', thickness)
        cut_out(obj_out, obj_in)
        marking_pts = in_pts
    elif line_type == 'half_closed':
        out_pts = [pt0_out, pt_out_half0, pt_in_half0, pt0_in, pt3_in, pt2_in, pt1_in, pt_in_half1, pt_out_half1, pt1_out, pt2_out, pt3_out]
        obj_out = create_poly_obj(np.array([pt+[0] for pt in out_pts]), 'slot_half_closed', thickness)
        marking_pts = [pt0_in, pt1_in, pt2_in, pt3_in]
    elif line_type == 'open':
        out_pts = [pt0_out, pt_out_open0, pt3_in, pt2_in, pt_out_open1, pt1_out, pt2_out, pt3_out]
        obj_out = create_poly_obj(np.array([pt+[0] for pt in out_pts]), 'slot_open', thickness)
        marking_pts = [pt_out_open0, pt_out_open1, pt2_in, pt3_in]
    return obj_out, marking_pts

def duplicate_link_obj(obj, locations, rotations):
    obj.select_set(True)
    for j in range(len(locations)):
        location = locations[j]
        rotation = rotations[j]
        bpy.ops.object.duplicate_move_linked()
        new_obj_name = [name for name in bpy.data.objects.keys() if name.startswith(obj.name)][-1]
        new_obj = bpy.data.objects[new_obj_name]
        new_obj.location = location
        new_obj.rotation_euler = rotation
    bpy.ops.object.select_all(action='DESELECT')    

def gen_partical_grass(scale, count, color):
    bpy.ops.object.camera_add()
    bpy.ops.mesh.primitive_plane_add(size=scale, enter_editmode=False, location=(0, 0, 0.02))
    plane_obj = bpy.context.object
    plane_obj.name = 'grass_ground'
#    bpy.ops.object.partical_system_add()
#    ps = bpy.data.particals['ParticalSettings']
    bpy.ops.object.modifier_add(type='PARTICLE_SYSTEM')
    ps = plane_obj.particle_systems[0].settings

    # 访问粒子系统设置
    ps.type = 'HAIR'
    ps.use_advanced_hair = True
    ps.count = count
    ps.hair_length = 0.01

    ps.brownian_factor = 0.02 # physics

    ps.child_type = 'SIMPLE' # children
    ps.child_percent = 20
    ps.rendered_child_count = 20
    ps.child_size = 1
    ps.child_size_random = 0.3
    ps.clump_factor = -0.2
    ps.clump_shape = 0.5

    ps.root_radius = 0.4 # hair shape
    ps.tip_radius = 0.1

    material = bpy.data.materials.new(name="GrassMaterial")
    plane_obj.data.materials.append(material)
    material.diffuse_color = color  # 绿色
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # 清除默认节点
    for node in nodes:
        nodes.remove(node)

    # 添加新节点
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    output = nodes.new(type='ShaderNodeOutputMaterial')

    # 连接节点
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # 设置BSDF节点参数
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.5

def add_paved_path_generation(path_file, collection_name, width, length, location, curve=False):
    new_collection = bpy.data.collections.new(name='paved_path')
    bpy.context.scene.collection.children.link(new_collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children['paved_path']
    # path_names = ['ALHAMBRA PATH', 'ALHAMBRA02 PATH', 'AZTECH PATH', 'CIRCLES PATH', 'CLASSIC01 PATH', 
    #             'CLASSIC02 PATH', 'CLASSIC03 PATH', 'CROSSES PATH', 'ESCHER PATH', 'HERRINGBONE PATH', 
    #             'MEDIEVAL01 PATH', 'MEDIEVAL02 PATH', 'MEDIEVAL03 PATH', 'PLANETS PATH', 'ROMAN PATH', 
    #             'SQUARES PATH', 'STARS PATH', 'SUNS PATH', 'WOODEN PATH', 'YELLOW BRICK PATH']
    path_size = dict()
    path_size['MEDIEVAL01 PATH'] = [0.464, 0.468]
    path_size['CLASSIC01 PATH'] = [0.403, 0.403]
    path_size['PLANETS PATH'] = [0.632, 0.628]
    path_size['CROSSES PATH'] = [0.635, 0.635]
    path_size['AZTECH PATH'] = [0.664, 0.669]
    path_size['SQUARES PATH'] = [0.609, 0.852]
    path_size['ALHAMBRA PATH'] = [0.41, 0.41]
    path_size['MEDIEVAL02 PATH'] = [1.11, 1.06]
    path_size['HERRINGBONE PATH'] = [0.471, 0.467]
    path_size['ESCHER PATH'] = [0.854, 0.856]
    path_size['ROMAN PATH'] = [1.14, 1.13]
    path_size['ALHAMBRA02 PATH'] = [0.422, 0.427]
    path_size['CIRCLES PATH'] = [0.604, 0.61]
    path_size['MEDIEVAL03 PATH'] = [0.62, 0.619]
    # path_size['YELLOW BRICK PATH'] = [1.16, 1.1]
    path_size['CLASSIC02 PATH'] = [0.443, 0.454]
    path_size['SUNS PATH'] = [0.851, 0.866]
    path_size['WOODEN PATH'] = [0.535, 0.537]
    path_size['STARS PATH'] = [0.673, 0.647]
    path_size['CLASSIC03 PATH'] = [0.84, 0.853]
    if collection_name == 'random':
        collection_name = random.choice(list(path_size.keys()))
    # collection_name = 'CLASSIC03 PATH'
    bpy.ops.wm.append(directory=path_file + "/Collection/", filename=collection_name)
    path_name = collection_name.replace(' ', '')
    path_name = path_name.replace('PATH', '')
    path = bpy.data.objects[path_name]
    path.location = (location[0]-length/2+0.32, location[1], 0)
    if not curve:
        curve = path.modifiers.get('Curve')
        path.modifiers.remove(curve)

    path.modifiers['GeometryNodes']['Input_33'] = False # border
    path.modifiers['GeometryNodes']['Input_6'] = round(length/path_size[collection_name][0])
    path.modifiers['GeometryNodes']['Input_5'] = round(width/path_size[collection_name][1])
    path.modifiers['GeometryNodes']['Input_10'] = 1.0 # old tiles
    path.modifiers['GeometryNodes']['Input_15'] = 0 # tilt
    path.modifiers['GeometryNodes']['Input_29'] = random.uniform(-1, 1)

    path.modifiers['GeometryNodes']['Input_13'] = random.uniform(0, 2) # grass size
    path.modifiers['GeometryNodes']['Input_12'] = random.uniform(0, 5) # grass amount
    path.modifiers['GeometryNodes']['Input_32'] = 0.75 # ground height

    # path.dimensions[0] = length
    # path.dimensions[1] = width
    return collection_name.split(' ')[0]

def add_front_obstacles(ego_pose, dist, obstacle):
    ego_yaw = ego_pose[2]
    angle_bias = random.uniform(-np.pi/6, np.pi/6)
    target_yaw = random.uniform(-np.pi, np.pi)
    target_x = ego_pose[3]+dist*math.cos(ego_yaw+angle_bias)
    target_y = ego_pose[4]+dist*math.sin(ego_yaw+angle_bias)
    target_name = add_object_from_file(obstacle, (target_x, target_y, 0), (0, 0, target_yaw), 'obstacle')
    if angle_bias <= -np.pi/12:
        res = 'right'
    elif angle_bias >= np.pi/12:
        res = 'left'
    else:
        res = 'middle'
    return res

def add_cone(ego_pose, model_path='/home/sczone/disk1/share/3d/blender_slots/elements/obstacles'):
    for obj in bpy.data.objects:
        if obj.name.startswith('corn'):
            bpy.data.objects.remove(obj)
    dist = random.uniform(5, 15)
    cone = random.choice(['corn1', 'corn2', 'corn3', 'corn4'])
    cone_file = os.path.join(model_path, cone+'.blend')
    cone_num = random.choice([1, 2, 3])
    cone_dires = []
    for i in range(cone_num):
        dist += 0.4
        cone_dire = add_front_obstacles(ego_pose, dist, cone_file)
        cone_dires.append(cone_dire)
    return cone_dires