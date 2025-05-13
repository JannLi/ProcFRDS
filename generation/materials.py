import bpy
import random
import os

def new_material(material_name, rgba):
    material = bpy.data.materials.new(material_name)
    material["is_auto"] = True
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission_node = nodes.new('ShaderNodeEmission')
    mat_out_node = nodes.new('ShaderNodeOutputMaterial')
    emission_node.inputs[0].default_value = rgba
    emission_node.inputs[1].default_value = 1
    material.node_tree.links.new(emission_node.outputs['Emission'], mat_out_node.inputs['Surface'])
    return material

def copy_material(source_mat, target_obj):
    target_mat = target_obj.material_slots
    if target_mat:
        for slot in target_mat:
            slot.material = source_mat
    else:
        target_obj.data.materials.append(source_mat)

def copy_material_from_obj(source_obj, target_obj):
    materials = []
    for slot in source_obj.material_slots:
        if slot.material is not None:
            materials.append(slot.material)
    source_mat = materials[-1]
    copy_material(source_mat, target_obj)

def add_material_from_file(blend_file_path, material_name, obj):
    material = bpy.data.materials.get(material_name)
    if not material:
        with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
            data_to.materials = [name for name in data_from.materials if name == material_name]
        material = bpy.data.materials[material_name]
    copy_material(material, obj)

def change_material_random(target_obj, blend_source_path):
    target_mat = random.choice(os.listdir(blend_source_path))
    ground_mat_file = os.path.join(blend_source_path, target_mat)
    target_mat_name = target_mat.split('.')[0]
    add_material_from_file(ground_mat_file, target_mat_name, target_obj)
    return target_mat_name

def change_avmseg_material():
    # bpy.ops.wm.open_mainfile(filepath=blend_path)
    slot_mat = new_material('slots', (1, 0, 0, 1))
    forward_mat = new_material('forward', (0, 0, 1, 1))
    forward_left_mat = new_material('forward_left', (0, 1, 0, 1))
    forward_right_mat = new_material('forward_right', (0, 1, 1, 1))
    forward_left_right_mat = new_material('forward_left_right', (1, 0, 1, 1))
    forward_uturn_mat = new_material('forward_uturn', (1, 1, 0, 1))
    left_mat = new_material('left', (0.5, 0, 0, 1))
    left_right_mat = new_material('left_right', (0, 0.5, 0, 1))
    left_uturn_mat = new_material('left_uturn', (0, 0, 0.5, 1))
    right_mat = new_material('right', (0.5, 0.5, 0, 1))
    uturn_mat = new_material('uturn', (0.5, 0, 0.5, 1))
    xforbidden_mat = new_material('xforbidden', (0, 0.5, 0.5, 1))
    other_mat =  new_material('cars', (0, 0, 0, 1))
    for name in bpy.data.objects.keys():
        if name in ['Camera', 'Light'] or name.startswith('Group'):
            continue
        obj = bpy.data.objects.get(name)
        if name.startswith('forward_left_right'):
            copy_material(forward_left_right_mat, obj)
        elif name.startswith('forward_right'):
            copy_material(forward_right_mat, obj)
        elif name.startswith('forward_left'):
            copy_material(forward_left_mat, obj)
        elif name.startswith('forward_uturn'):
            copy_material(forward_uturn_mat, obj)
        elif name.startswith('forward'):
            copy_material(forward_mat, obj)
        elif name.startswith('left_right'):
            copy_material(left_right_mat, obj)
        elif name.startswith('left_uturn'):
            copy_material(left_uturn_mat, obj)
        elif name.startswith('left'):
            copy_material(left_mat, obj)
        elif name.startswith('right'):
            copy_material(right_mat, obj)
        elif name.startswith('uturn'):
            copy_material(uturn_mat, obj)
        elif name.startswith('xforbidden'):
            copy_material(xforbidden_mat, obj)
        else:
            try:
                copy_material(other_mat, obj)
            except:
                continue
    slot_collection = bpy.data.collections.get('slots')
    for child in slot_collection.objects:
        copy_material(slot_mat, child)

def get_linked_sockets(node, socket_name='', socket_type='inputs'):
    # 存储与该节点相连的所有链接信息
    linked_sockets = {
        'inputs': [],
        'outputs': []
    }
    if len(socket_name) == 0:
        # 检查输入链接
        for input_socket in node.inputs:
            for link in input_socket.links:
                linked_sockets['inputs'].append({
                    'from_node': link.from_node,
                    'from_socket': link.from_socket,
                    'to_socket': input_socket
                })
        # 检查输出链接
        for output_socket in node.outputs:
            for link in output_socket.links:
                linked_sockets['outputs'].append({
                    'from_socket': output_socket,
                    'to_node': link.to_node,
                    'to_socket': link.to_socket
                })
    else:
        if socket_type == 'inputs':
            socket = node.inputs[socket_name]
            for link in socket.links:
                linked_sockets['inputs'].append({
                    'from_node': link.from_node,
                    'from_socket': link.from_socket,
                    'to_socket': socket
                })
        else:
            socket = node.outputs[socket_name]
            for link in socket.links:
                linked_sockets['outputs'].append({
                    'from_socket': socket,
                    'to_node': link.to_node,
                    'to_socket': link.to_socket
                })
    return linked_sockets

def make_node_group(ori_mat, target_nodes):
    mat_copy = ori_mat.copy()
    mat_copy_nodes = mat_copy.node_tree.nodes
    mat_copy_links = mat_copy.node_tree.links
    output_node = mat_copy_nodes.get('Material Output')
    group_output_node = mat_copy_nodes.get('Group Output')
    if output_node and (not group_output_node):
        mat_copy_output_link = get_linked_sockets(output_node, socket_name='Surface', socket_type='inputs')
        link_from_socket = mat_copy_output_link['inputs'][0]['from_socket']
        group_output = mat_copy_nodes.new(type='NodeGroupOutput')
        mat_copy_links.new(link_from_socket, group_output.inputs['BSDF'])
    # mat_copy_nodes.remove(output_node)

    group_node = target_nodes.new(type='ShaderNodeGroup')
    group_node.node_tree = mat_copy.node_tree
    return group_node

def mix_materials(obj, mat1, mat2, mix_mat_name, mix_fac=0.5):
    mix_mat = bpy.data.materials.new(name=mix_mat_name)
    obj.data.materials.append(mix_mat)

    mix_mat.use_nodes = True
    nodes = mix_mat.node_tree.nodes
    links = mix_mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    
    # group_node1 = make_node_group(mat1, nodes)
    # group_node2 = make_node_group(mat2, nodes)
    group_node1 = nodes.new(type='ShaderNodeGroup')
    group_node1.location = (-400, 100)
    group_node1.node_tree = mat1.node_tree

    group_node2 = nodes.new(type='ShaderNodeGroup')
    group_node2.location = (-400, -100)
    group_node2.node_tree = mat2.node_tree

    # 添加 Mix Shader 节点
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (0, 0)

    # 添加输出节点
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (200, 0)

    # 连接节点
    links.new(group_node1.outputs['BSDF'], mix_shader.inputs[1])
    links.new(group_node2.outputs['BSDF'], mix_shader.inputs[2])
    links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])

#    设置混合比例
    mix_shader.inputs['Fac'].default_value = mix_fac  # 50% 材质1，50% 材质2
    return mix_mat

def add_puddle(loc, size, ground_mat):
    puddle_col = bpy.data.collections.get('puddles')
    if not puddle_col:
        puddle_col = bpy.data.collections.new(name='puddles')
        bpy.context.scene.collection.children.link(puddle_col)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children['puddles']

    bpy.ops.mesh.primitive_plane_add(size=size, location=loc, enter_editmode=False, align='WORLD')
    puddle_plane = bpy.data.objects['Plane']
    puddle_plane.name = 'puddle_plane'
    add_material_from_file('/home/sczone/disk1/share/3d/blender_slots/elements/materials/Procedural Puddle 2.0.blend', 'Procedural Puddle', puddle_plane)
    puddle_mat = bpy.data.materials.get('Procedural Puddle')
    mix_mat = mix_materials(puddle_plane, puddle_mat, ground_mat, 'puddle_ground', mix_fac=0.5)

    bpy.data.node_groups['NodeGroup'].nodes['Noise Texture'].inputs[5].default_value = random.uniform(0, 10) #平整度
    bpy.data.node_groups['NodeGroup'].nodes['Noise Texture'].inputs[5].default_value = random.uniform(0, 5) #形状

def add_puddle2(mat, mix_fac):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (0, 0)

    # 添加输出节点
    output_node = nodes.get('Material Output')
    if not output_node:
        output_node = nodes.get('材质输出')
    output_link = get_linked_sockets(output_node, socket_name='Surface', socket_type='inputs')
    link_from_socket = output_link['inputs'][0]['from_socket']

    # 连接节点
    links.new(nodes['Group'].outputs['BSDF'], mix_shader.inputs[1])
    links.new(link_from_socket, mix_shader.inputs[2])
    links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])

    #    设置混合比例
    mix_shader.inputs['Fac'].default_value = mix_fac  # 50% 材质1，50% 材质2
    bpy.data.node_groups['NodeGroup'].nodes['Noise Texture'].inputs[5].default_value = random.uniform(0, 10) #平整度
    bpy.data.node_groups['NodeGroup'].nodes['Noise Texture'].inputs[5].default_value = random.uniform(0, 5) #形状