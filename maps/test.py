import sys
sys.path.append('../')
import bpy
from generation.materials import change_material_random
from generation.objects import uv_editing

bpy.ops.wm.open_mainfile(filepath='/home/sczone/disk1/share/3d/blender_slots/elements/path/Paved_Path_Generator1.3.blend')
new_collection = bpy.data.collections.new(name='paved_path')
bpy.context.scene.collection.children.link(new_collection)
bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children['paved_path']
path_names = ['ALHAMBRA PATH', 'ALHAMBRA02 PATH', 'AZTECH PATH', 'CIRCLES PATH', 'CLASSIC01 PATH', 
            'CLASSIC02 PATH', 'CLASSIC03 PATH', 'CROSSES PATH', 'ESCHER PATH', 'HERRINGBONE PATH', 
            'MEDIEVAL01 PATH', 'MEDIEVAL02 PATH', 'MEDIEVAL03 PATH', 'PLANETS PATH', 'ROMAN PATH', 
            'SQUARES PATH', 'STARS PATH', 'SUNS PATH', 'WOODEN PATH', 'YELLOW BRICK PATH']
path_size = dict()
path_size['MEDIEVAL01 PATH'] = [0.651, 0.618]
path_size['CLASSIC01 PATH'] = [0.579, 0.55]
path_size['PLANETS PATH'] = [0.868, 0.825]
path_size['CROSSES PATH'] = [0.868, 0.825]
path_size['AZTECH PATH'] = [1.68, 1.7]
path_size['SQUARES PATH'] = [0.868, 0.825]
path_size['ALHAMBRA PATH'] = [0.579, 0.55]
path_size['MEDIEVAL02 PATH'] = [1.46, 1.38]
path_size['HERRINGBONE PATH'] = [0.651, 0.618]
path_size['ESCHER PATH'] = [1.15, 1.09]
path_size['ROMAN PATH'] = [1.45, 1.37]
path_size['ALHAMBRA02 PATH'] = [0.579, 0.55]
path_size['CIRCLES PATH'] = [0.868, 0.825]
path_size['MEDIEVAL03 PATH'] = [0.868, 0.825]
path_size['YELLOW BRICK PATH'] = [1.16, 1.1]
path_size['CLASSIC02 PATH'] = [0.579, 0.55]
path_size['SUNS PATH'] = [1.16, 1.11]
path_size['WOODEN PATH'] = [0.72, 0.69]
path_size['STARS PATH'] = [0.859, 0.832]
path_size['CLASSIC03 PATH'] = [1.11, 1.1]


for collection_name in path_names:
    # bpy.ops.wm.append(directory="/home/sczone/disk1/share/blender/grass_brick/Paved Path Generator 1.3/Paved_Path_Generator1.3.blend/Collection/", filename=collection_name)
    path_name = collection_name.replace(' ', '').replace('PATH', '')
    path = bpy.data.objects[path_name]

    path.modifiers['GeometryNodes']['Input_6'] = 1
    path.modifiers['GeometryNodes']['Input_5'] = 1
    tile_width = path.dimensions[0]
    tile_length = path.dimensions[1]
    size.append([tile_width, tile_length])

print(size)