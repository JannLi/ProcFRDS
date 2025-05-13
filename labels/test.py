import bpy
import bmesh
from mathutils import Vector
import numpy as np
import open3d as o3d

def change_point_density(obj, level, render_level):
    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=True, obdata=True, material=False, animation=False)
    subdivide = obj.modifiers.new(name='subdivide', type='SUBSURF')
    subdivide.subdivision_type = 'SIMPLE'
    print(subdivide.levels, subdivide.render_levels)
    subdivide.levels = level
    # subdivide.render_levels = render_level
    bpy.ops.object.modifier_apply(modifier=subdivide.name)

# Example usage
bpy.ops.wm.open_mainfile(filepath="/home/sczone/disk1/share/temp.blend")
obj = bpy.data.objects['Cube']  # Assuming active object is the point cloud
change_point_density(obj, 2, 1)

# Export the downsampled point cloud
filepath = "/home/sczone/disk1/share/temp.ply"
bpy.ops.wm.ply_export(filepath=filepath)

pcd = o3d.io.read_point_cloud('/home/sczone/disk1/share/scene.ply')
# pcd2 = o3d.io.read_point_cloud('../ProcFRPS/test/results/4/ply/scene.ply')
print(np.unique(np.array(pcd.colors), axis=0))
print(np.array(pcd.points)[0])
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# # 可选步骤：简化网格（如果需要）
# # mesh = mesh.simplify_quadric_decimation(100000)
# voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(pcd, voxel_size=0.2)

# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)

# 可选：可视化voxel
o3d.visualization.draw_geometries([pcd])