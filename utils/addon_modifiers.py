import addon_utils
import random
import bpy

def add_dust(obj):
    addon_utils.enable('AgedFX', default_set=True)
    obj.select_set(True)
    bpy.ops.object.make_dust_selected_operator()
    bpy.context.scene.aged_fx.dust_amount = random.uniform(0.5, 1)
    bpy.context.scene.aged_fx.dust_top = random.uniform(0.5, 1)
    bpy.context.scene.aged_fx.dust_side = random.uniform(0.5, 1)
    bpy.context.scene.aged_fx.dust_scale = random.uniform(0, 20)
    obj.select_set(False)
# bpy.context.view_layer.update()

def add_deformations(obj):
    addon_utils.enable('AgedFX', default_set=True)
    obj.select_set(True)
    bpy.ops.object.make_deformations_selected_operator()
    bpy.context.scene.aged_fx.bumps = random.uniform(0.5, 1)
    bpy.context.scene.aged_fx.scratches = random.uniform(0.5, 1)
    bpy.context.scene.aged_fx.deformations_amount = random.uniform(0.5, 1)
    bpy.context.scene.aged_fx.deformations_scale = random.uniform(0, 10)
    obj.select_set(False)
# bpy.context.view_layer.update()
