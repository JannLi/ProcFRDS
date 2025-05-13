import os
import bpy
import addon_utils

def hdri_maker_bg_dome(lib_path, bg_id, scale):
    addon_utils.enable('hdri_maker', default_set=True)
    file_path = os.path.join(lib_path, bg_id+'.hdr')
    if not os.path.exists(file_path):
        file_path = os.path.join(lib_path, bg_id+'.exr')
    print(file_path)
    bpy.ops.hdrimaker.addbackground(filepath=file_path, invoke_browser=True)
    bpy.ops.hdrimaker.dome(options='ADD')
    bpy.data.objects['Dome Handler'].scale = (scale, scale, scale)

def world_bg_img(img_path_list, mix_fac=0.5, render_bg=False):
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("New World")
    world = bpy.context.scene.world
    world.use_nodes = True

    # 清理现有的节点
    for node in world.node_tree.nodes:
        world.node_tree.nodes.remove(node)

    # 添加背景节点和世界输出节点
    background_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')

    # 设置环境纹理路径
    img_path1 = img_path_list[0]
    environment_texture_node = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
    environment_texture_node.image = bpy.data.images.load(img_path1)
    if len(img_path_list) == 2:
        img_path2 = img_path_list[1]
        environment_texture_node2 = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
        environment_texture_node2.image = bpy.data.images.load(img_path2)
        background_node2 = world.node_tree.nodes.new(type='ShaderNodeBackground')
        mix_node = world.node_tree.nodes.new('ShaderNodeMixShader')
        world.node_tree.links.new(environment_texture_node2.outputs['Color'], background_node2.inputs['Color'])
        world.node_tree.links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
        world.node_tree.links.new(background_node.outputs['Background'], mix_node.inputs[1])
        world.node_tree.links.new(background_node2.outputs['Background'], mix_node.inputs[2])
        mix_node.inputs['Fac'].default_value = mix_fac
        world.node_tree.links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])
    else:
        world.node_tree.links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
        world.node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

    # 可选：调整背景节点的强度
    background_node.inputs['Strength'].default_value = 1.0 # 根据需要调整这个值
    if not render_bg:
        bpy.context.scene.render.film_transparent = True

def add_shadow(x, y):
    bpy.ops.mesh.primitive_plane_add(size=500, location=(x, y, -0.78), enter_editmode=False, align='WORLD')
    # bpy.context.scene.render.engine = 'CYCLES'
    # bpy.data.scenes['Scene'].cycles.device = 'GPU'
    bpy.data.objects['Plane'].is_shadow_catcher = True
    # bpy.data.materials[material_name].shadow_method = 'OPAQUE'
    # bpy.data.materials[material_name].alpha_threshold = 0.9

    # bpy.context.scene.render.film_transparent = True
    # ground_plane = bpy.context.object
    # ground_material = bpy.data.materials.new(name='Ground')
    # ground_plane.data.materials.append(ground_material)
    # ground_material.use_nodes = True
    # ground_material.blend_method = 'BLEND'
    # ground_material.shadow_method = 'OPAQUE'
    # nodes = ground_material.node_tree.nodes
    # mix_shader_node = nodes.new(type='ShaderNodeMixShader')
    # mix_shader_node.inputs[0].default_value = 0.5
    # shader_rgb_node = nodes.new(type='ShaderNodeShaderToRGB')
    # shader_trans_node = nodes.new(type="ShaderNodeBsdfTransparent")
    # shader_bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")

    # ground_material.node_tree.links.new(mix_shader_node.outputs['Shader'], nodes['Material Output'].inputs['Surface'])
    # ground_material.node_tree.links.new(shader_trans_node.outputs['BSDF'], mix_shader_node.inputs[2])
    # ground_material.node_tree.links.new(shader_rgb_node.outputs['Color'], mix_shader_node.inputs[0])
    # ground_material.node_tree.links.new(shader_bsdf_node.outputs['BSDF'], shader_rgb_node.inputs[0]) 