import bpy

def add_geometry(obj, voxel_size):
    bpy.context.view_layer.objects.active= obj
    obj.select_set(True)
    # bpy.ops.object.editmode_toggle()

    geometry_modifier = obj.modifiers.new(name='Geometry Nodes', type='NODES')
    geometry_modifier.node_group = bpy.data.node_groups.new(type='GeometryNodeTree', name='My Geometry Nodes')
    bpy.ops.node.new_geometry_node_group_assign()
    node_tree = geometry_modifier.node_group
    input_node = node_tree.nodes['Group Input']
    output_node = node_tree.nodes['Group Output']

    m2v_node = node_tree.nodes.new('GeometryNodeMeshToVolume')
    m2v_node.resolution_mode = 'VOXEL_SIZE'
    m2v_node.inputs[2].default_value = voxel_size

    v2m_node = node_tree.nodes.new('GeometryNodeVolumeToMesh')
    v2m_node.resolution_mode = 'VOXEL_SIZE'
    v2m_node.inputs[2].default_value = voxel_size

    node_tree.links.new(input_node.outputs[0], m2v_node.inputs[0])
    node_tree.links.new(m2v_node.outputs[0], v2m_node.inputs[0])
    node_tree.links.new(v2m_node.outputs[0], output_node.inputs[0])
 
    bpy.context.view_layer.update()

def add_remesh(obj, depth):
    remesh_modifier = obj.modifiers.new(name='Remesh', type='REMESH')
    remesh_modifier.mode = 'BLOCKS'
    remesh_modifier.octree_depth = depth
    remesh_modifier.use_remove_disconnected = False

def mesh2voxel(obj, voxel_size, depth):
    add_geometry(obj, voxel_size)
    add_remesh(obj, depth)