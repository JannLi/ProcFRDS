import sys
sys.path.append('../')
import os
import bpy
import random
import time
import json
import numpy as np
from generation.scenes import RoadsideSceneCreater, SceneModifier, SurfaceSceneCreater, UndergroundSceneCreater
from generation.scene_customization import GrassBrickSceneCreater
from generation.rendering import render_img_mask
from generation.objects import add_arrows_on_road
from utils.parking_utils import get_target_slot

def main(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    base_scence_blend = os.path.join(args.out_path, 'base_scene.blend')
    out_blend = os.path.join(args.out_path, 'out.blend')
    out_img_path = os.path.join(args.out_path, 'results')
    slots_info_path = os.path.join(args.out_path, 'slots_info.json')
    if not os.path.exists(out_img_path):
        os.makedirs(out_img_path)
    lock, obstacle, arrows = True, True, True
    limiter = random.choice([True, False])
    if not os.path.exists(out_blend):
        if not os.path.exists(base_scence_blend):
            scene_spec = args.scene_spec
            if scene_spec == 'underground':
                creater = UndergroundSceneCreater(args.element_path, base_scence_blend, layout_path=args.layout_path)
            elif scene_spec == 'surface':
                creater = SurfaceSceneCreater(args.element_path, base_scence_blend, layout_path=args.layout_path)
            elif scene_spec == 'grassbrick':
                creater = GrassBrickSceneCreater(args.element_path, base_scence_blend, layout_path=args.layout_path)
            else:
                creater = RoadsideSceneCreater(args.element_path, base_scence_blend, layout_path=args.layout_path, no_slots=args.no_slots)

            slot_angle = creater.build_base_scene()
            creater.add_static_objs()
            creater.set_background()
            creater.modify_base_scene()
            creater.save_base_scene()
            # if creater.slot_type == 'vertical':
            #     lock, obstacle = True, True
            if len(args.layout_path) == 0:
                layout = creater.export_layout()
                with open(os.path.join(args.out_path, 'layout.json'), 'w') as f:
                    json.dump(layout, f, indent=4)
            else:
                layout = creater.layout
        else:
            bpy.ops.wm.open_mainfile(filepath=base_scence_blend)
            with open(args.layout_path, 'r') as f:
                layout = json.load(f)
            slot_angle = layout['slots'][0]['slot_angle']

        target_slot = get_target_slot(args.ego_pose_path)
        if args.no_slots == '1':
            limiter, lock, obstacle, arrows = False, False, False, False
        left_car = random.choice([True, False])
        right_car = random.choice([True, False])
        modifier = SceneModifier(args.element_path, out_blend, slot_angle=slot_angle, limiter=limiter, 
                                lock=lock, obstacle=obstacle, target_slot=target_slot, left_car=left_car, right_car=right_car)
        modifier.modify_static_objs()
        modifier.add_dynamic_objs_on_roads()
        occupied_slots = modifier.add_dynamic_objs_in_slots()

        if arrows:
            roads = bpy.data.collections.get('roads')
            if not roads:
                roads = bpy.data.collections.get('lanes')
            for road in roads.objects:
                add_arrows_on_road(road, os.path.join(args.element_path, 'arrows'))

        if args.no_slots == '1':
            slots_col = bpy.data.collections.get('slots')
            if slots_col:
                for obj in slots_col.objects:
                    bpy.data.objects.remove(obj)
            roads_col = bpy.data.collections.get('roads')
            if roads_col:
                for obj in roads_col.objects:
                    bpy.data.objects.remove(obj)

        slots_info = modifier.slots_info
        with open(slots_info_path, 'w') as f:
            json.dump(slots_info, f, indent=4)        
        modifier.save_output_scene()
    else:
        bpy.ops.wm.open_mainfile(filepath=out_blend)

    render_img_mask(out_img_path, args.ego_pose_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='random PROCFRPS')
    parser.add_argument('--layout_path', help='layout file', default='')
    parser.add_argument('--scene_spec', help='scene specifics', default='roadside')
    parser.add_argument('--ego_pose_path', help='ego pose file')
    parser.add_argument('--element_path', help='elements path', default='/home/sczone/disk1/share/3d/blender_slots/elements')
    parser.add_argument('--out_path', help='output path')
    parser.add_argument('--no_slots', help='no slot lines', default='0')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)