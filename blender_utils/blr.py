import numpy as np
import os
from typing import Union
import bpy
import cv2
from .slam_utils.matrix_utils import pose_to_4x4, cam_mat_to_blender_mat, mat_to_4x4
from .slam_utils.cam_utils import cam_intr_2_lens, DistiortTool
import json
from .slam_utils.img_utils.img_utils import combine_car_shadow_bg, combine_rgba_and_rgb, combine_car_shadow_bg2
from .slam_utils.img_utils.mask_utils import mask_to_img, img_to_mask
import torch
import uuid
import sys
import shutil
import contextlib

exp_scene = os.path.join(os.path.dirname(__file__), r'./exp_scene/exp_car_sun_plane2.blend')
hdri_img_path = os.path.join(os.path.dirname(__file__), r'./exp_scene/hdri_img.hdr')

# scene operations
def load_scene(model_path: str = None):
    if model_path is not None:
        bpy.ops.wm.open_mainfile(filepath=model_path)
        
def create_empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
         

# set objs as initial state
def set_as_init(model_path:str, model_name:str = None, save_file_path: Union[bool, str] = False):
    load_scene(model_path)
    if model_name is None:
        bpy.ops.object.select_all(action='SELECT')
    else:
        bpy.ops.object.select_all(action='DESELECT')
        objs = get_obj_and_children(model_path, model_name)
        for obj in objs:
            obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if isinstance(save_file_path, str):
        bpy.ops.wm.save_mainfile(filepath=os.path.abspath(save_file_path))
    else:
        if save_file_path:
            bpy.ops.wm.save_mainfile(filepath=os.path.abspath(model_path))


# get specific objs
def get_obj_and_children(model_path: str, model_name: str):
    load_scene(model_path)
    parent_obj = bpy.data.objects[model_name]
    obj_list = get_children_in_scene(parent_obj, [])
    return obj_list

def get_children_in_scene(parent_obj: str, obj_list: list):
    # 将当前物体添加到结果列表
    obj_list.append(parent_obj)

    # 获取物体的所有直接子物体
    children = list(parent_obj.children)

    # 遍历子物体
    for child in children:
        # 递归调用获取子物体的子物体
        obj_list = get_children_in_scene(child, obj_list)
    return obj_list
    

# corners operations
def choose_ground_corners(box_corners: np.array):  
    box_corners_ground = box_corners[[0, 3, 7, 4], :3]
    return box_corners_ground  

def create_scene_sun_plane_cam_bg(background_img_path:str, background_img: np.array, cam_intri: np.array, cam_extri: np.array, use_world: bool=True, render_samples: int = 64, strength = 4.0):
    img_size = (background_img.shape[1], background_img.shape[0])
    create_scene(img_size, render_samples)
    set_camera_intri_extri(cam_intri, cam_extri, img_size, background_img_path)
    if background_img_path is not None:
        background_img_path = os.path.abspath(background_img_path)
        set_back_envri(background_img_path, use_world, strength)
    # add ground plane
    directory = exp_scene
    inner_path = r'Object'
    obj_name = r'Plane'
    bpy.ops.wm.append(filepath=os.path.join(directory, inner_path, obj_name),
                    directory=os.path.join(directory, inner_path),
                    filename=obj_name)
    obj = bpy.data.objects['Plane']
    obj.matrix_world = np.eye(4).T
    obj.scale = [1, 1, 1]
    # add sun
    obj_name = r'Temp_Sum'
    bpy.ops.wm.append(filepath=os.path.join(directory, inner_path, obj_name),
                    directory=os.path.join(directory, inner_path),
                    filename=obj_name)        
    obj = bpy.data.objects['Temp_Sum']
    obj.data.energy = 3.0
    obj.rotation_euler = (0, 0, 0)  

def put_model_in_img(model_path:str, model_name: str, model_scale: float, background_img_path:str, background_img: np.array, location_3d: np.array, cam_intri: np.array, cam_extri: np.array, use_world: bool=True, render_samples: int = 64, on_th_plane: bool = True, on_the_ground: bool = True, origin_scene_path: str = None):
        if isinstance(origin_scene_path, str):
            bpy.ops.wm.open_mainfile(filepath=origin_scene_path)
        background_img_path = os.path.abspath(background_img_path)
        img_size = (background_img.shape[1], background_img.shape[0])
        create_scene(img_size, render_samples)
        # set camera
        sensor_width = 36
        focal_length, shift_x, shift_y = cam_intr_2_lens(img_size, cam_intri, sensor_width)
        cam = bpy.data.objects['Camera']
        # set camera background image
        cam.data.show_background_images = True
        bg  = cam.data.background_images.new()
        bg.image = bpy.data.images.load(background_img_path)
        # set camera param
        cam_mat = cam_mat_to_blender_mat(cam_extri)
        set_camera(sensor_width, focal_length, shift_x, shift_y)
        cam.matrix_world = cam_mat.T
        # set car model
        if model_path is not None:
            car_pose = location_3d
            if car_pose.shape[0] == 4 and car_pose.shape[1] == 4:
                car_mat = car_pose
            else:
                car_mat = pose_to_4x4(car_pose)       
            add_object(model_path, model_name, car_mat, model_scale)
            obj = bpy.data.objects[model_name]
            obj.scale = (model_scale, model_scale, model_scale)
            box_corners = np.array([np.dot(np.array(obj.matrix_world), np.append(np.array(corner), 1)) for corner in obj.bound_box])
            model_pose = obj.rotation_euler
        else:
            box_corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]])

        # add ground plane
        directory = exp_scene
        inner_path = r'Object'
        obj_name = r'Plane'
        bpy.ops.wm.append(filepath=os.path.join(directory, inner_path, obj_name),
                        directory=os.path.join(directory, inner_path),
                        filename=obj_name)
        obj = bpy.data.objects['Plane']
        if on_th_plane and model_path is not None:
            obj.matrix_world = car_mat.T
            obj.scale = [1, 1, 1]
        # add sun
        obj_name = r'Temp_Sum'
        bpy.ops.wm.append(filepath=os.path.join(directory, inner_path, obj_name),
                        directory=os.path.join(directory, inner_path),
                        filename=obj_name)        
        obj = bpy.data.objects['Temp_Sum']
        obj.data.energy = 3.0
        if model_name is not None:
            obj.location = bpy.data.objects[model_name].location
        obj.rotation_euler = (0, 0, 0)    
        # shadow on the right
        # obj.rotation_euler = (-33.047/180*np.pi, 32.542/180*np.pi, 14.446/180*np.pi)
        # change environment
        set_back_envri(background_img_path, use_world)
        bpy.context.view_layer.update()
        box_corners = box_corners[:, :3]
        if on_the_ground:
            box_corners[:, 2][np.abs(box_corners[:, 2])<0.01] = 0
        return box_corners
    
    
def set_back_envri(img_path: str, use_world: bool=True, strength = 4.0, mirror_ball: bool = False):
        # change environment
        world = bpy.context.scene.world
        world.use_nodes = True
        node_tree = world.node_tree    
        for node in node_tree.nodes:
            node_tree.nodes.remove(node)
        world_node = node_tree.nodes.new(type='ShaderNodeOutputWorld')
        bg_shader_node = node_tree.nodes.new(type='ShaderNodeBackground')
        bg_shader_node.inputs[1].default_value = strength
        bg_img_node = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
        bg_img_node.image = bpy.data.images.load(os.path.abspath(img_path))
        # bg_img_node.projection = 'MIRROR_BALL'
        node_tree.links.new(bg_img_node.outputs['Color'], bg_shader_node.inputs['Color'])
        node_tree.links.new(bg_shader_node.outputs['Background'], world_node.inputs['Surface'])
        node_tree.nodes['Environment Texture'].projection = 'MIRROR_BALL'
        world.use_nodes = use_world

def get_model_name_in_blr(model_path: Union[str, None] = None):
    if model_path is not None:
        if model_path.endswith('blend'):
            bpy.ops.wm.open_mainfile(filepath=model_path)
        else:
            bpy.ops.wm.read_homefile()
            for obj in bpy.data.objects:
                bpy.data.objects.remove(obj)
            if model_path.endswith('fbx'):
                bpy.ops.import_scene.fbx(filepath=model_path)
            elif model_path.endswith('obj'):
                bpy.ops.import_scene.obj(filepath=model_path)
            elif model_path.endswith('glb'):
                bpy.ops.import_scene.gltf(filepath=model_path)
    for obj in bpy.context.selectable_objects:
        if obj.parent is None:
            break
    # bpy.ops.wm.save_mainfile(filepath=os.path.abspath('temp/just_for_memory.blend'))
    return obj.name

def add_whole_object(model_path: str, model_name: str, projection_mat_4x4: np.array, model_scale: Union[float, np.ndarray]=1.0, on_the_plane: bool = False, plane_name='Plane', new_name='', collect=''):   
    exsited_obj = set(bpy.data.objects.keys())
    ori_model_name = model_name
    i = 1
    while model_name in exsited_obj:
        model_name = ori_model_name + '.' + str(i).zfill(3)
        i += 1
    if ori_model_name in exsited_obj:
        obj_to_copy_list = get_obj_and_children(None, ori_model_name)
        for obj in obj_to_copy_list:
            obj.select_set(True)
        bpy.ops.object.duplicate_move_linked()
        bpy.ops.object.select_all(action='DESELECT')
    else:
        if model_path.endswith('fbx'):
            bpy.ops.import_scene.fbx(filepath=model_path)
        elif model_path.endswith('obj'):
            bpy.ops.import_scene.obj(filepath=model_path)
        elif model_path.endswith('glb'):
            bpy.ops.import_scene.gltf(filepath=model_path)
        elif model_path.endswith('blend'):
        # 打开目标文件
            if len(collect) > 0:
                new_collection = bpy.data.collections.get(collect)
                if not new_collection:
                    new_collection = bpy.data.collections.new(name=collect)
                    bpy.context.scene.collection.children.link(new_collection)
                bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collect]
            with bpy.data.libraries.load(model_path, link=False) as (data_from, data_to):
            # 获取所有的 collection 名称
                collection_names = data_from.collections
            if len(collection_names) > 0:
                # 从目标文件中加载一个 collection
                bpy.ops.wm.append(directory=model_path + "/Collection/", filename=collection_names[0])
                # move to the same "Collection"
                source_collection_name = collection_names[0]
                # bpy.data.collections[source_collection_name].matrix_world = np.eye(4).T
                target_collection_name = "Collection"

                # 获取源集合和目标集合
                source_collection = bpy.data.collections[0]
                target_collection = bpy.data.collections[1]
                if not(source_collection.name == target_collection.name):
                    # 如果源集合和目标集合都存在，执行移动操作
                    if source_collection and target_collection:
                        for obj in source_collection.objects:
                            # if 'WGT' in obj.name:
                            #     continue
                            target_collection.objects.link(obj)

                        # 解绑源集合中的所有物体
                        for obj in source_collection.objects:
                            source_collection.objects.unlink(obj)
            else:
                # 如果没有 collection，加载所有的物体
                bpy.ops.wm.append(directory=model_path + "/Object/", filename=ori_model_name)
                for obj in bpy.data.objects:
                    if not obj.name in exsited_obj:
                        model_name = obj.name
                        break

        if len(bpy.data.objects) == len(exsited_obj):
            obj_to_copy_list = get_obj_and_children(None, ori_model_name)
            for obj_name in obj_to_copy_list:
                bpy.data.objects[obj_name].select_set(True)
            bpy.ops.object.duplicate_move_linked()
            bpy.ops.object.select_all(action='DESELECT')
        

    if isinstance(model_scale, float):
        scale_x = model_scale
        scale_y = model_scale
        scale_z = model_scale
    else:
        scale_x = model_scale[0]
        scale_y = model_scale[1]
        scale_z = model_scale[2]
    # for obj in bpy.data.objects:
    #     print(obj)
    obj = bpy.data.objects[model_name]

    # obj.location = (0, 0, 0)
    # obj.rotation_euler = (0, 0, 0)
    obj.matrix_world = projection_mat_4x4.T
    obj.scale[0] = scale_x
    obj.scale[1] = scale_y
    obj.scale[2] = scale_z
    obj.scale = (scale_x, scale_y, scale_z)
    if on_the_plane:
        ori_plane = bpy.data.objects[plane_name]
        plane = ori_plane.copy()
        # plane.data.materials.clear()
        bpy.context.collection.objects.link(plane)
        plane.matrix_world = projection_mat_4x4.T
        plane.scale = [obj.dimensions[0]*0.9, obj.dimensions[1]*0.9, 1.0]
        if plane.scale[0] < 2:
            plane.scale[0] = 2
        if plane.scale[1] < 2:
            plane.scale[1] = 2
    if len(new_name) > 0:
        obj.name = new_name
        model_name = new_name
    bpy.context.view_layer.update()   
    return model_name

def copy_obj_and_children(obj_name: str):
    exsisted_obj_names = []
    for obj in bpy.data.objects:
        exsisted_obj_names.append(obj.name)
    obj_to_copy_list = get_obj_and_children(None, obj_name)
    for obj in obj_to_copy_list:
        obj.select_set(True)
    bpy.ops.object.duplicate()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name not in exsisted_obj_names and obj_name in obj.name:
            return obj.name
        
def del_obj_and_children(obj_name: str):
    obj_list = get_obj_and_children(None, obj_name)
    for obj in obj_list:
        obj.select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')
    
def add_object(model_path: str, model_name: str, projection_mat_4x4: np.array, model_scale: float=1.0, on_the_plane: bool = False):
    exsited_obj = []
    ori_model_name = model_name
    for obj in bpy.data.objects:
        exsited_obj.append(obj.name)
    i = 1
    while model_name in exsited_obj:
        model_name = ori_model_name + '.' + str(i).zfill(3)
        i += 1
    if model_path.endswith('fbx'):
        bpy.ops.import_scene.fbx(filepath=model_path)
    elif model_path.endswith('obj'):
        bpy.ops.import_scene.obj(filepath=model_path)
    elif model_path.endswith('blend'):
        bpy.ops.wm.append(directory=model_path + "Object", filename=model_name)
    elif model_path.endswith('glb'):
        bpy.ops.import_scene.gltf(filepath=model_path)
    if len(bpy.data.objects) == len(exsited_obj):
        obj = bpy.data.objects[ori_model_name].copy()
        bpy.context.collection.objects.link(obj)
    scale_x = model_scale
    scale_y = model_scale
    scale_z = model_scale
    # for obj in bpy.data.objects:
    #     print(obj)
    obj = bpy.data.objects[model_name]

    # obj.location = (0, 0, 0)
    # obj.rotation_euler = (0, 0, 0)
    obj.matrix_world = projection_mat_4x4.T
    obj.scale[0] = scale_x
    obj.scale[1] = scale_y
    obj.scale[2] = scale_z
    obj.scale = (scale_x, scale_y, scale_z)
    if on_the_plane:
        plane = bpy.data.objects['Plane'].copy()
        bpy.context.collection.objects.link(plane)
        plane.matrix_world = projection_mat_4x4.T
        plane.scale = [obj.dimensions[0]*2, obj.dimensions[1]*2, 0]
        if plane.scale[0] < 3:
            plane.scale[0] = 3
        if plane.scale[1] < 3:
            plane.scale[1] = 3
    bpy.context.view_layer.update()
    
def set_backlight(scale: float):
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (scale, scale, scale, 1)
    
    

    
def set_render_resolution(img_size: np.array):
    scene = bpy.context.scene
    scene.render.resolution_x = img_size[0]
    scene.render.resolution_y = img_size[1]
    
def create_scene(img_size, render_samples: int = 64, use_engine = 'EEVEE', use_device = 'GPU'):
    # empty_blend_path = r'exp_scene/empty_scene.blend'
    # bpy.ops.wm.open_mainfile(filepath=empty_blend_path)
    create_empty_scene()
    if not torch.cuda.is_available():
        use_device = 'CPU'
    bpy.ops.wm.read_homefile()
    if use_engine == 'CYCLES':
        set_cycles_render(img_size, render_samples, use_device)
    elif use_engine == 'EEVEE':
        set_eevee_render(img_size, render_samples)
    
    world = bpy.data.worlds['World']
    # Ensure no background node
    world.use_nodes = True
    set_backlight(0.1)
    # try:
    #     world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
    # except KeyError:
    #     pass
    try:
        bpy.data.objects.remove(bpy.data.objects['Cube'])
    except KeyError:
        pass
    try:
        bpy.data.objects.remove(bpy.data.objects['Light'])
    except KeyError:
        pass
    for obj in bpy.data.objects:
        if obj.name != 'Camera':
            bpy.data.objects.remove(obj)
            
def set_camera_intri_extri(cam_intri, cam_extri, img_size: np.array =np.array([1920, 1080]), img_path: str = None):
    # set camera
    sensor_width = 36
    focal_length, shift_x, shift_y = cam_intr_2_lens(img_size, cam_intri, sensor_width)
    cam = bpy.data.objects['Camera']
    # set camera background image
    if img_path is not None:
        cam.data.show_background_images = True
        cam.data.background_images.clear()
        bg  = cam.data.background_images.new()
        bg.image = bpy.data.images.load(img_path)
    # set camera param
    cam_mat = cam_mat_to_blender_mat(cam_extri)
    set_camera(sensor_width, focal_length, shift_x, shift_y)
    # cam.data.angle = 0.6911112070083618
    # 7.358891487121582, -6.925790786743164, 4.958309173583984
    cam.data.sensor_width = sensor_width
    cam.data.lens = focal_length
    cam.data.shift_x = shift_x
    cam.data.shift_y = shift_y
    cam.data.clip_end = 300
    cam.matrix_world = cam_mat.T
    
def set_camera(sensor_width, focal_length, shift_x = 0, shift_y = 0,):
    cam = bpy.data.objects['Camera']
    # cam.data.angle = 0.6911112070083618
    # 7.358891487121582, -6.925790786743164, 4.958309173583984
    cam.location.x = 0
    cam.location.y = 0
    cam.location.z = 0
    cam.rotation_euler = [np.pi, 0, 0]
    cam.data.sensor_width = sensor_width
    cam.data.lens = focal_length
    cam.data.shift_x = shift_x
    cam.data.shift_y = shift_y
    cam.data.clip_end = 300
    
    
def set_rndr(output_path):
    rndr = bpy.context.scene.render
    rndr.image_settings.color_mode ='RGBA'
    bpy.data.scenes["Scene"].render.image_settings.file_format = 'PNG'
    bpy.data.scenes["Scene"].render.filepath = output_path
    
def get_obj_box_ipm(pxPerM: float, box_corners_3d: np.array):
    box_corners_ground = choose_box_corners_ground(box_corners_3d)
    box_corners_ipm_origin = box_corners_ground * pxPerM
    return box_corners_ipm_origin    
    
def get_obj_box_ground(model_path: str, model_name='Car Rig', model_scale = 1.0, model_euler: float=np.pi/2):
    box_corners = get_obj_box(model_path, model_name, model_euler, model_euler)
    box_corners_ground = box_corners[[0, 3, 7, 4], :2]
    return box_corners_ground  

def choose_box_corners_ground(box_corners: np.array):  
    box_corners_ground = box_corners[[0, 3, 7, 4], :2]
    return box_corners_ground  

def get_obj_box_ipm_from_3d(pxPerM: float, box_corners_3d: np.array):
    return choose_box_corners_ground(box_corners_3d) * pxPerM

def get_obj_box(model_path: str, model_name='Car Rig', model_scale = 1.0, model_euler=np.pi/2):
    bpy.ops.wm.read_homefile()
    model_pose = np.array([0, 0, model_euler, 0, 0, 0])
    model_mtr = pose_to_4x4(model_pose)
    add_object(model_path, model_name, model_mtr, model_scale)
    obj = bpy.data.objects[model_name]
    box_corners = np.array([np.dot(np.array(obj.matrix_world), np.append(np.array(corner), 1)) for corner in obj.bound_box])
    os.makedirs('temp', exist_ok=True)
    bpy.ops.wm.save_mainfile(filepath=os.path.abspath('temp/just_for_memory.blend'))
    return box_corners

# change the initial state of model
def change_model_origin(model_path: str, model_name: str, target_model_pose: np.array = None, target_model_scale: np.array = None, save_file_path: Union[bool, str] = False):
    if isinstance(model_path, str):
        bpy.ops.wm.open_mainfile(filepath=model_path)
    obj = bpy.data.objects[model_name]
    if target_model_pose is not None:
        if len(target_model_pose.shape) == 1:
            model_mat = pose_to_4x4(target_model_pose)
        else:
            model_mat = target_model_pose
    else:
        model_mat = np.array(obj.matrix_world)
    if target_model_scale is None:
        target_model_scale = np.array(obj.scale)
    obj.matrix_world = model_mat.T
    obj.scale = target_model_scale
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.context.scene.tool_settings.use_transform_data_origin = True
    bpy.context.view_layer.update()
    bpy.ops.transform.resize(value=(1/obj.scale.x, 1/obj.scale.y, 1/obj.scale.z))
    bpy.ops.transform.rotate(value=obj.rotation_euler[2], orient_axis='Z')
    bpy.ops.transform.rotate(value=obj.rotation_euler[1], orient_axis='Y')
    bpy.ops.transform.rotate(value=obj.rotation_euler[0], orient_axis='X')
    bpy.ops.transform.translate(value=-obj.location)
    bpy.context.view_layer.update()
    # bpy.context.scene.tool_settings.use_transform_data_origin = False
    if isinstance(save_file_path, str):
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        bpy.ops.wm.save_mainfile(filepath=os.path.abspath(save_file_path))
    else:
        if save_file_path:
            bpy.ops.wm.save_mainfile(filepath=os.path.abspath(model_path))
            
            
def apply_to_default(model_name: str):
    obj = bpy.data.objects[model_name]
    bpy.context.scene.tool_settings.use_transform_data_origin = True
    bpy.context.view_layer.update()
    bpy.ops.transform.resize(value=(1/obj.scale.x, 1/obj.scale.y, 1/obj.scale.z))
    bpy.ops.transform.rotate(value=obj.rotation_euler[2], orient_axis='Z')
    bpy.ops.transform.rotate(value=obj.rotation_euler[1], orient_axis='Y')
    bpy.ops.transform.rotate(value=obj.rotation_euler[0], orient_axis='X')
    bpy.ops.transform.translate(value=-obj.location)
    bpy.context.view_layer.update()
    bpy.context.scene.tool_settings.use_transform_data_origin = False
    
            
def items_to_parent(model_path: str, parent_item_name: str, save_file_path: Union[bool, str] = False):
    if isinstance(model_path, str):
        bpy.ops.wm.open_mainfile(filepath=model_path)
    bpy.ops.object.select_all(action='SELECT')
    selected_objects = bpy.context.selected_objects
    bpy.context.view_layer.objects.active = bpy.data.objects[parent_item_name]
    # 获取当前选择的目标物体
    target_object = bpy.context.active_object

    # 获取当前选择的其他物体列表
    selected_objects = bpy.context.selected_objects
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)  # 使用操作设置父子关系
    bpy.ops.object.select_all(action='DESELECT')
    # 将选中的其他物体设置为目标物体的子物体
    # for obj in selected_objects:
    #     if obj != target_object:
    #         bpy.ops.object.select_all(action='DESELECT')  # 取消选择所有物体
    #         target_object.select_set(True)  # 选择目标物体
    #         bpy.context.view_layer.objects.active = target_object  # 设置活动物体
            
    #         obj.select_set(True)  # 重新选择当前物体
    # bpy.context.view_layer.update()
    if isinstance(save_file_path, str):
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        bpy.ops.wm.save_mainfile(filepath=os.path.abspath(save_file_path))
    else:
        if save_file_path:
            bpy.ops.wm.save_mainfile(filepath=os.path.abspath(model_path))

def get_box_3d_in_scene(model_name: str):
    if model_name == 'whole':
        all_box_corners = []
        for obj in bpy.context.selectable_objects: 
            if obj.type ==  'MESH':
                box_corners = np.array([np.dot(obj.matrix_world, np.append(np.array(corner), 1)) for corner in obj.bound_box])
                all_box_corners.append(box_corners)
        all_box_corners = np.concatenate(all_box_corners, axis=0)
        x_min = np.min(all_box_corners[:, 0])
        y_min = np.min(all_box_corners[:, 1])
        z_min = np.min(all_box_corners[:, 2])
        x_max = np.max(all_box_corners[:, 0])
        y_max = np.max(all_box_corners[:, 1])
        z_max = np.max(all_box_corners[:, 2])
        box_corners = np.array([[x_min, y_min, z_min, 1], [x_min, y_min, z_max, 1], [x_min, y_max, z_max, 1], [x_min, y_max, z_min, 1], [x_max, y_min, z_min, 1], [x_max, y_min, z_max, 1], [x_max, y_max, z_max, 1], [x_max, y_max, z_min, 1]])
        return box_corners
    obj = bpy.data.objects.get(model_name)
    if obj is None:
        return None
    box_corners = np.array([np.dot(obj.matrix_world, np.append(np.array(corner), 1)) for corner in obj.bound_box])
    return box_corners  

def get_box_3d_origin(model_name: str):
    if model_name == 'whole':
        all_box_corners = []
        for obj in bpy.context.selectable_objects: 
            box_corners = np.array([np.dot(np.eye(4), np.append(np.array(corner), 1)) for corner in obj.bound_box])
            all_box_corners.append(box_corners)
        all_box_corners = np.concatenate(all_box_corners, axis=0)
        x_min = np.min(all_box_corners[:, 0])
        y_min = np.min(all_box_corners[:, 1])
        z_min = np.min(all_box_corners[:, 2])
        x_max = np.max(all_box_corners[:, 0])
        y_max = np.max(all_box_corners[:, 1])
        z_max = np.max(all_box_corners[:, 2])
        box_corners = np.array([[x_min, y_min, z_min, 1], [x_min, y_min, z_max, 1], [x_min, y_max, z_max, 1], [x_min, y_max, z_min, 1], [x_max, y_min, z_min, 1], [x_max, y_min, z_max, 1], [x_max, y_max, z_max, 1], [x_max, y_max, z_min, 1]])
        return box_corners
    obj = bpy.data.objects.get(model_name)
    if obj is None:
        return None
    box_corners = np.array([np.dot(np.eye(4), np.append(np.array(corner), 1)) for corner in obj.bound_box])
    return box_corners  
         
def clean_model(model_path: str, model_name: str, target_dimension: np.array=None, target_model_scale: np.array = None, save_file_path: Union[bool, str] = False):
    if isinstance(model_path, str):
        if model_path.endswith('blend'):
            bpy.ops.wm.open_mainfile(filepath=model_path)
        else:
            create_empty_scene()    
            add_object(model_path, model_name, np.eye(4), 1.0)
    box_corners = get_box_3d_origin(model_name)
    box_corners_ground = box_corners.copy()
    box_corners_ground = box_corners_ground[box_corners_ground[:,2].argsort()[:4], :]
    box_corners_ground_center = np.mean(box_corners_ground, axis=0)
    if len(bpy.data.objects) > 1:
        items_to_parent(None, model_name, save_file_path=False)
    obj = bpy.data.objects[model_name]
    obj.location = -box_corners_ground_center[:3]
    if target_dimension is not None:
        target_dimension_scale_sort = target_dimension.argsort()
        original_dimension_scale_sort = np.array(obj.dimensions).argsort()
        scale = target_dimension[target_dimension_scale_sort] / np.array(obj.dimensions)[original_dimension_scale_sort]
        scale = scale[original_dimension_scale_sort.argsort()]
    change_model_origin(None, model_name, None, target_model_scale, save_file_path=False)
    if isinstance(save_file_path, str):
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        bpy.ops.wm.save_mainfile(filepath=os.path.abspath(save_file_path))
    else:
        if save_file_path:
            bpy.ops.wm.save_mainfile(filepath=os.path.abspath(model_path))    
            
            
def other_formats_2_blend(model_path: str, save_file_path: Union[bool, str] = False):
    create_empty_scene()
    if model_path.endswith('fbx'):
        bpy.ops.import_scene.fbx(filepath=model_path)
    elif model_path.endswith('obj'):
        bpy.ops.import_scene.obj(filepath=model_path)
    elif model_path.endswith('glb'):
        bpy.ops.import_scene.gltf(filepath=model_path)
    if isinstance(save_file_path, str):
        if not save_file_path.endswith('blend'):
            save_file_path += '.blend'
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        bpy.ops.wm.save_mainfile(filepath=os.path.abspath(save_file_path))
    else:
        if save_file_path:
            save_file_path = model_path.replace(model_path.split('.')[-1], 'blend')
            bpy.ops.wm.save_mainfile(filepath=os.path.abspath(save_file_path))    

def get_model_info(model_path: str, model_name: Union[str, None] = None, items_info_list: Union[list, None] = None, save_path: Union[str, bool] = False):
    bpy.ops.wm.open_mainfile(filepath=model_path)
    model_info_dict = {'model_path': model_path}
    model_info_dict['model_file_name'] = os.path.basename(model_path)
    if model_name is None:
        model_name = get_model_name_in_blr()
    model_info_dict['model_name'] = model_name
    box_corners_dict = {}
    # if items_info_list is None or len(items_info_list) == 0:
    item_info = {'box_corners': get_box_3d_in_scene('whole').tolist()}
    item_info['label'] = model_name
    box_corners_dict['whole'] = item_info
    if items_info_list is not None and len(items_info_list) > 0:
        for i in range(len(items_info_list)):
            item_info = items_info_list[i]
            item_info['box_corners'] = get_box_3d_in_scene(item_info['item_name']).tolist()
            if 'label' not in item_info.keys():
                item_info['label'] = item_info['item_name']
            box_corners_dict[item_info['item_name']] = item_info
    model_info_dict['box_corners'] = box_corners_dict
    if isinstance(save_path, str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(model_info_dict, f)
    elif save_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path.replace('.' + model_path.split('.')[-1], '.json'), 'w') as f:
            json.dump(model_info_dict, f)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return model_info_dict
            
def change_model_origin_temp(model_path: str, save_file_path: Union[bool, str] = False):
    bpy.ops.wm.open_mainfile(filepath=model_path)
    for obj in bpy.context.selectable_objects:
        # if obj.name.startswith('body') or obj.name.startswith('wheel'):
        obj.select_set(True)
    selected_objects = bpy.context.selected_objects
    bpy.context.view_layer.objects.active = bpy.data.objects[os.path.basename(model_path).split('.')[0]]
    bpy.ops.object.join()
    for obj in bpy.context.selectable_objects:
        # if obj.name.startswith('body') or obj.name.startswith('wheel'):
        obj.select_set(True)
    obj = bpy.data.objects[os.path.basename(model_path).split('.')[0]]
    obj.scale = [1.79627/obj.dimensions[2], 1.79627/obj.dimensions[2], 1.79627/obj.dimensions[2]]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if isinstance(save_file_path, str):
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        bpy.ops.wm.save_mainfile(filepath=os.path.abspath(save_file_path))
    else:
        if save_file_path:
            bpy.ops.wm.save_mainfile(filepath=os.path.abspath(model_path))    
            
            
def clean_car_model_temp(model_path: str, save_file_path:str):
    bpy.ops.wm.open_mainfile(filepath=model_path)
    bpy.data.objects['Car Rig'].data.pose_position = 'REST'
    # bpy.context.scene.collection.objects.unlink(bpy.data.objects['Car Rig'])
    # 遍历场景中的所有物体
    for obj in bpy.context.selectable_objects:
        # 检查物体的数据是否是多用户的
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    bpy.ops.object.select_all(action='SELECT')
    # for obj in bpy.context.selectable_objects:
    #     # 检查物体的数据是否是多用户的
    #     if obj.data.users > 1:
    #         continue
    #     obj.select_set(True)
    # bpy.data.objects['Car Rig'].select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    # try:
    #     items_to_parent(None, 'body', False)
    # except:
    #     bpy.data.objects['Body'].name = 'body'
    #     items_to_parent(None, 'body', False)
    # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # apply_to_default('Car Rig')
    # bpy.data.objects['Car Rig'].matrix_world = pose_to_4x4(np.array([0, 0, np.pi/2, 0, 0, 0])).T
    # apply_to_default('Car Rig')
    obj = bpy.data.objects['Car Rig']
    obj.matrix_world = np.dot(np.array(obj.matrix_world), pose_to_4x4(np.array([0, 0, np.pi/2, 0, 0, 0]))).T
    # obj = bpy.data.objects['body']
    # obj.matrix_world = np.dot(np.array(obj.matrix_world), pose_to_4x4(np.array([0, 0, np.pi*2, 0, 0, 0]))).T
    # obj = bpy.data.objects['wheel.Bk.L']
    # obj.matrix_world = np.dot(np.array(obj.matrix_world), pose_to_4x4(np.array([0, 0, np.pi*2, 0, 0, 0]))).T
    # obj = bpy.data.objects['wheel.Bk.R']
    # obj.matrix_world = np.dot(np.array(obj.matrix_world), pose_to_4x4(np.array([0, 0, np.pi*2, 0, 0, 0]))).T
    # obj = bpy.data.objects['wheel.Ft.L']
    # obj.matrix_world = np.dot(np.array(obj.matrix_world), pose_to_4x4(np.array([0, 0, np.pi*2, 0, 0, 0]))).T
    # obj = bpy.data.objects['wheel.Ft.R']
    # obj.matrix_world = np.dot(np.array(obj.matrix_world), pose_to_4x4(np.array([0, 0, np.pi*2, 0, 0, 0]))).T
    # for obj in bpy.context.selectable_objects:
    # # 检查物体的数据是否是多用户的
    #     # if obj.data.users > 1:
    #     #     obj.data = obj.data.copy()
    #     #     # continue
    #     # print(obj.name)
    #     if obj.name == 'Car Rig':
    #         continue
    #     # obj.matrix_world = np.dot(np.array(obj.matrix_world), pose_to_4x4(np.array([0, 0, np.pi*2, 0, 0, 0]))).T
    #     obj.matrix_world = np.eye(4).T
        # obj.select_set(True)
        # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    # bpy.data.objects['Car Rig'].data.pose_position = 'REST'
    bpy.ops.wm.save_as_mainfile(filepath=save_file_path)
                
def clean_car_model_temp2(model_path: str, save_file_path:str):
    bpy.ops.wm.open_mainfile(filepath=model_path)
    bpy.data.objects['Car Rig'].data.pose_position = 'REST'
    model_name = 'body'
    if bpy.data.objects.get('body') is None:
        model_name = 'Body'
    # bpy.context.scene.collection.objects.unlink(bpy.data.objects['Car Rig'])
    # 遍历场景中的所有物体
    for obj in bpy.context.selectable_objects:
        # 检查物体的数据是否是多用户的
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    bpy.ops.object.select_all(action='SELECT')
    bpy.data.objects['Car Rig'].select_set(False)
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.data.objects.remove(bpy.data.objects['Car Rig'])
    items_to_parent(None, model_name, False)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.data.objects[model_name].rotation_euler = [0, 0, np.pi/2]
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # bpy.data.objects['Car Rig'].data.pose_position = 'REST'
    bpy.ops.wm.save_as_mainfile(filepath=save_file_path)
    return model_name

def set_eevee_render(img_size: np.ndarray, render_samples: int = 64):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = img_size[0]
    scene.render.resolution_y = img_size[1]
    scene.render.film_transparent = True
    scene.eevee.gi_auto_bake = True
    scene.eevee.taa_render_samples = render_samples
    
def replace_all_materials():
    pass

def render_res(silent: bool = False, temp_output = False, target='norm'):
    if target == 'seg':
        bpy.context.scene.cycles.samples = 1
        bpy.context.scene.display_settings.display_device = 'sRGB'
        bpy.context.scene.view_settings.view_transform = 'Raw'
        bpy.context.scene.cycles.use_denoising = False
        bpy.context.scene.render.use_simplify = True
    if silent:
        if temp_output:
            temp_file = uuid.uuid4().hex + '.txt'
            open(temp_file, 'w').close()
            old = os.dup(sys.stdout.fileno())
            sys.stdout.flush()
            os.close(sys.stdout.fileno())
            fd = os.open(temp_file, os.O_WRONLY)

            # do the rendering
            bpy.ops.render.render(write_still=True)

            # disable output redirection
            os.close(fd)
            os.dup(old)
            os.close(old)
            os.remove(temp_file)
        else:
            # output nothing
            old = os.dup(sys.stdout.fileno())
            sys.stdout.flush()
            os.close(sys.stdout.fileno())
            os.open(os.devnull, os.O_WRONLY)
            bpy.ops.render.render(write_still=True)
            
            os.close(1)
            os.dup(old)
            os.close(old)
    else:
        bpy.ops.render.render(write_still=True)

def render_with_fisheye(ori_img, fisheye_info, distort_tool: DistiortTool, rendered_img_shadow_path = None, rendered_img_no_shadow_path = None, ori_img_mask = None, mask_only = False, render_silent=False, target='norm'):
    cam = bpy.data.objects['Camera']
    cam.data.type = 'PANO'
    # cam.data.cycles.panorama_type = 'FISHEYE_EQUISOLID'
    cam.data.panorama_type = 'FISHEYE_EQUISOLID'
    # cam.data.cycles.fisheye_fov = distort_tool.blr_fov
    cam.data.fisheye_fov = distort_tool.blr_fov
    cam.data.shift_x = distort_tool.blr_shift_x
    cam.data.shift_y = distort_tool.blr_shift_y
    # cam.data.cycles.fisheye_lens = distort_tool.blr_focal_length
    cam.data.fisheye_lens = distort_tool.blr_focal_length
    cam.data.sensor_width = distort_tool.blr_sensor_size
    cam_img_size = distort_tool.blr_img_size
    plane = bpy.data.objects.get('Plane')
    if plane is not None:
        plane.data.materials.clear()
    for obj in bpy.context.scene.objects:
        if 'Plane' in obj.name:
            obj.is_shadow_catcher = True
    if mask_only:
        render_samples = 8
    else:
        render_samples = 8
    set_cycles_render(img_size=cam_img_size//2, render_samples=render_samples)
    obj_set = []
    default_fisheye_mask = cv2.imread(os.path.join(os.path.dirname(__file__), 'default_blr_fisheye_mask.png'), -1)[..., -1]
    default_fisheye_mask = cv2.resize(default_fisheye_mask, (cam_img_size[0]//2, cam_img_size[1]//2))
    default_fisheye_mask = img_to_mask(default_fisheye_mask)
    for obj in bpy.context.scene.objects:
        if 'Plane' in obj.name:
            obj_set.append(obj)
    objs_invisible_in_rendering(obj_set)
    set_rndr(output_path=rendered_img_no_shadow_path)
    # bpy.ops.render.render(write_still=True)
    # turn off all the lights and use background image as light
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.data.energy = 0.0
    render_res(temp_output= not render_silent, target=target)
    # render_res(target=target)
    img_mask = cv2.imread(rendered_img_no_shadow_path, -1)
    img_no_shadow = cv2.imread(rendered_img_no_shadow_path, -1)
    img_mask[..., -1][default_fisheye_mask] = 0
    img_mask_ori = img_mask.copy()
    # distort_tool.load_from_dict(fisheye_info)
    # distort_tool.update_fisheye_blr_info(cam_img_size, 3*15, 10.0, np.pi*1.5)
    # distort_tool.update_map_fisheye2distort()
    objs_visible_in_rendering(obj_set)
    if not mask_only: 
        set_rndr(output_path=rendered_img_shadow_path)
        # bpy.ops.render.render(write_still=True)
        render_res(temp_output= not render_silent, target=target)
        img_with_shadow = cv2.imread(rendered_img_shadow_path, -1)
        only_shadow_mask = img_with_shadow[..., -1] - img_no_shadow[..., -1]
        thres = 95
        only_shadow_mask[only_shadow_mask<thres] = thres
        only_shadow_mask = (only_shadow_mask - thres) * 1.22
        only_shadow_mask[only_shadow_mask>255] = 255
        only_shadow_mask = only_shadow_mask.astype(np.uint8)
        # only_shadow_mask[only_shadow_mask<thres] = 0
        img_with_shadow[..., -1] = only_shadow_mask + img_no_shadow[..., -1]
        img_with_shadow = cv2.resize(img_with_shadow, (cam_img_size[0], cam_img_size[1]))
        img_with_shadow = distort_tool.img_fisheye2distort(img_with_shadow)
        if ori_img_mask is not None:
            img_with_shadow[..., 3][ori_img_mask] = 0
        # combined_img = combine_car_shadow_bg2(img_mask, img_shadow, ori_img, ori_img_mask)
        combined_img = combine_rgba_and_rgb(img_with_shadow, ori_img)
        img_mask = img_with_shadow[..., -1].copy()
        img_mask[img_mask>1] = 255
        img_mask[ori_img_mask] = 0
        # combined_img = cv2.imread(rendered_img_no_shadow_path, -1)
        # combined_img = distort_tool.img_fisheye2distort(combined_img)
        # combined_img = combine_rgba_and_rgb(combined_img, ori_img)
    else:
        img_mask = distort_tool.img_fisheye2distort(img_mask_ori[..., -1])
        img_mask[img_mask<255] = 0
        combined_img = img_mask.copy()
    return combined_img, img_mask

def full_render_with_fisheye(ori_img, distort_tool: DistiortTool, rendered_img_shadow_path = None, render_silent=False, target='norm'):
    cam = bpy.data.objects['Camera']
    cam.data.type = 'PANO'
    cam.data.panorama_type = 'FISHEYE_EQUISOLID'
    cam.data.fisheye_fov = distort_tool.blr_fov
    cam.data.shift_x = distort_tool.blr_shift_x
    cam.data.shift_y = distort_tool.blr_shift_y
    cam.data.fisheye_lens = distort_tool.blr_focal_length
    cam.data.sensor_width = distort_tool.blr_sensor_size
    cam_img_size = distort_tool.blr_img_size
    render_samples = 64
    # set_cycles_render(img_size=cam_img_size//2, render_samples=render_samples)
    set_cycles_render(img_size=cam_img_size, render_samples=render_samples)
    set_rndr(output_path=rendered_img_shadow_path)
    render_res(temp_output= not render_silent, target=target)
    img_with_shadow = cv2.imread(rendered_img_shadow_path, -1)
    # img_with_shadow = cv2.resize(img_with_shadow, (cam_img_size[0], cam_img_size[1]))
    img_with_shadow = distort_tool.img_fisheye2distort(img_with_shadow)
    combined_img = combine_rgba_and_rgb(img_with_shadow, ori_img)
    return combined_img

    
def render_with_eevee(ori_img, rendered_img_shadow_path = None, rendered_img_no_shadow_path = None, ori_img_mask = None, render_silent=True):
    out_put_img_size  = np.array([ori_img.shape[1], ori_img.shape[0]])
    os.makedirs('temp', exist_ok=True)
    if rendered_img_shadow_path is None:
        rendered_img_shadow_path = 'temp/rendered_img_shadow.png'
    if rendered_img_no_shadow_path is None:
        rendered_img_no_shadow_path = 'temp/rendered_img_no_shadow.png'
    # change plane material
    plane = bpy.data.objects.get('Plane')
    if plane is not None:
        plane.data.materials[0] = bpy.data.materials['Material.002']
    scene = bpy.context.scene
    if scene.render.engine != 'BLENDER_EEVEE':
        set_eevee_render(out_put_img_size)
    # world = bpy.context.scene.world
    # world.use_nodes = False
    set_rndr(rendered_img_shadow_path)
    bpy.ops.render.render(write_still=True)
    # world.use_nodes = True
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.data.energy = 0.0   
    set_rndr(rendered_img_no_shadow_path)
    render_res(temp_output= not render_silent)
    combined_img = combine_car_shadow_bg(cv2.imread(rendered_img_no_shadow_path, -1), cv2.imread(rendered_img_shadow_path, -1), ori_img, ori_img_mask)
    img_mask = cv2.imread(rendered_img_no_shadow_path, -1)[..., -1]
    img_mask[img_mask<=64] = 0
    img_mask[ori_img_mask] = 0
    return combined_img, img_mask

def render_with_cycles2(ori_img, rendered_img_shadow_path = None, render_silent=False):
    cam = bpy.data.objects['Camera']
    cam.data.type = 'PERSP'
    out_put_img_size  = np.array([ori_img.shape[1], ori_img.shape[0]])
    set_cycles_render(img_size=out_put_img_size//2, render_samples=8)
    # set_cycles_render(img_size=out_put_img_size//5, render_samples=8)
    set_rndr(rendered_img_shadow_path)
    render_res(temp_output= not render_silent)
    img_with_shadow = cv2.imread(rendered_img_shadow_path, -1)
    img_with_shadow = cv2.resize(img_with_shadow, (ori_img.shape[1], ori_img.shape[0]))
    combined_img = combine_rgba_and_rgb(img_with_shadow, ori_img)
    return combined_img

def set_cycles_render(img_size: np.ndarray, render_samples: int = 256, use_device='GPU'):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    # if not torch.cuda.is_available():
    #     use_device = 'CPU'
    # if use_device == 'GPU':
    #     scene.cycles.device = 'GPU'
    #     scene.cycles.feature_set = 'EXPERIMENTAL'
    #     bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
    #     # 启用所有可用的 GPU 设备
    #     bpy.context.preferences.addons['cycles'].preferences.get_devices()
    #     # 将所有 GPU 设备设为活动状态
    #     for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    #         if 'NVIDIA' in device.name:
    #             device.use = True
    # else:
    #     scene.cycles.device = 'CPU' 
    #     # 启用所有可用的 GPU 设备
    #     bpy.context.preferences.addons['cycles'].preferences.get_devices()
    #     # 将所有 GPU 设备设为活动状态
    #     for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    #         device.use = True

    # another way to choose render device
    scene.cycles.device = 'GPU'
    scene.cycles.feature_set = 'EXPERIMENTAL'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
    # 启用所有可用的 GPU 设备
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    # 将所有 GPU 设备设为活动状态
    available_gpu_num = 0
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if 'NVIDIA' in device.name or 'Quadro' in device.name:
            device.use = True
            available_gpu_num = available_gpu_num + 1
    if available_gpu_num == 0:
        # use all devices
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True
    scene.render.resolution_x = img_size[0]
    scene.render.resolution_y = img_size[1]
    scene.render.film_transparent = True
    cycles = scene.cycles
    cycles.samples = render_samples
    cycles.use_progressive_refine = False
    cycles.max_bounces = 100
    cycles.min_bounces = 10
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.diffuse_bounces = 10
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 4
    cycles.volume_bounces = 0
    cycles.transparent_min_bounces = 8
    cycles.transparent_max_bounces = 64
    cycles.blur_glossy = 5
    cycles.sample_clamp_indirect = 5
    cycles.sample_as_light = True

# accelation settings
    # scene.render.use_bake_multires = True
    # cycles.use_denoising = False
    # Avoid grainy renderings (fireflies)
    

    
def render_with_cycles(ori_img, rendered_img_no_shadow_path = None, rendered_img_shadow_path = None, ori_img_mask = None, render_silent=True):
    out_put_img_size  = np.array([ori_img.shape[1], ori_img.shape[0]])
    os.makedirs('temp', exist_ok=True)
    if rendered_img_shadow_path is None:
        rendered_img_shadow_path = 'temp/rendered_img_shadow.png'
    if rendered_img_no_shadow_path is None:
        rendered_img_no_shadow_path = 'temp/rendered_img_no_shadow.png'
    plane = bpy.data.objects.get('Plane')
    if plane is not None:
        plane.data.materials.clear()
    for obj in bpy.context.scene.objects:
        if 'Plane' in obj.name:
            obj.is_shadow_catcher = True
    scene = bpy.context.scene
    if scene.render.engine != 'CYCLES':
        set_cycles_render(out_put_img_size)
    world = bpy.context.scene.world
    world.use_nodes = True
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.data.energy = 0.0   
    set_rndr(rendered_img_shadow_path)
    render_res(temp_output= not render_silent)
    # combined_img = combine_rgba_and_rgb(cv2.imread(rendered_img_shadow_path, -1), ori_img)
    img_mask = render_with_workbench(ori_img, rendered_img_no_shadow_path)
    # img_mask = cv2.imread('temp/rendered_img_mask.png', -1)
    combined_img = combine_car_shadow_bg2(img_mask, cv2.imread(rendered_img_shadow_path, -1), ori_img, ori_img_mask)
    img_mask = img_mask[..., -1]
    img_mask[ori_img_mask] = 0
    return combined_img, img_mask

def set_workbench_render(img_size: np.ndarray):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = img_size[0]
    scene.render.resolution_y = img_size[1]

def render_with_workbench(ori_img, rendered_img_mask_path: str = None):
    out_put_img_size  = np.array([ori_img.shape[1], ori_img.shape[0]])
    os.makedirs('temp', exist_ok=True)
    if rendered_img_mask_path is None:
        rendered_img_mask_path = 'temp/rendered_img_mask.png'
    plane = bpy.data.objects.get('Plane')
    if plane is not None:
        plane.data.materials.clear()
    for obj in bpy.context.scene.objects:
        if 'Plane' in obj.name:
            obj.hide_render = True
    scene = bpy.context.scene
    if scene.render.engine != 'BLENDER_WORKBENCH':
        set_workbench_render(out_put_img_size)
    world = bpy.context.scene.world
    world.use_nodes = True
    set_rndr(rendered_img_mask_path)
    bpy.ops.render.render(write_still=True)
    for obj in bpy.context.scene.objects:
        if 'Plane' in obj.name:
            obj.hide_render = False
    img_mask = cv2.imread(rendered_img_mask_path, -1)
    return img_mask

def clear_cam_light(model_path: str, save_file_path: Union[bool, str] = False):
    load_scene(model_path)
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' or obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)
        if isinstance(save_file_path, str):
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            bpy.ops.wm.save_mainfile(filepath=os.path.abspath(save_file_path))
        else:
            if save_file_path:
                bpy.ops.wm.save_mainfile(filepath=os.path.abspath(model_path))

def obj_to_center(model_path: str, model_name: str = None, save_file_path: Union[bool, str] = False):
    load_scene(model_path)
    if model_name is None:
        model_name = get_model_name_in_blr()
    obj = bpy.data.objects[model_name]
    box_corners = get_box_3d_origin(model_name)
    ground_corners = choose_ground_corners(box_corners)
    ground_center = np.mean(ground_corners, axis=0)
    obj.location = -ground_center
    return ground_center

def obj_to_center_init(model_path: str, model_name: str, save_file_path: Union[bool, str] = False):
    obj_to_center(model_path, model_name, False)
    
def objs_invisible_in_rendering(objs = None):
    if objs is None:
        objs = bpy.context.selectable_objects
    for obj in objs:
        obj.hide_render = True
        
def objs_visible_in_rendering(objs = None):
    if objs is None:
        objs = bpy.context.selectable_objects
    for obj in objs:
        obj.hide_render = False

def create_parking_scene(ori_scene: str, scene_config: dict, model_info_dict):
    pass
    