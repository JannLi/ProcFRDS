import sys
sys.path.append('../')
import os
import bpy
import random
import time
import numpy as np
from generation.scenes import RoadsideSceneCreater, SceneModifier, SurfaceSceneCreater, UndergroundSceneCreater
from generation.scene_customization import GrassBrickSceneCreater
from generation.rendering import render_img_mask, dump_marking_pts
from generation.poses import get_ego_poses_for_road
from generation.objects import add_arrows_on_road

def main(args):
    base_scence_blend = os.path.join(args.out_path, 'base_scene.blend')
    out_blend = os.path.join(args.out_path, 'out.blend')
    ego_poses_file = os.path.join(args.out_path, 'poses.npy')
    out_img_path = os.path.join(args.out_path, 'results')
    out_json_path = os.path.join(args.out_path, 'pts.json')
    if not os.path.exists(out_img_path):
        os.makedirs(out_img_path)
    if not os.path.exists(out_blend):
        if not os.path.exists(base_scence_blend):
            scene_spec = args.scene_spec
            if scene_spec == 'underground':
                creater = UndergroundSceneCreater(args.element_path, base_scence_blend)
            elif scene_spec == 'surface':
                creater = SurfaceSceneCreater(args.element_path, base_scence_blend)
            elif scene_spec == 'grassbrick':
                creater = GrassBrickSceneCreater(args.element_path, base_scence_blend)
            else:
                creater = RoadsideSceneCreater(args.element_path, base_scence_blend)

            slot_angle = creater.build_base_scene()
            creater.add_static_objs()
            creater.set_background()
            creater.modify_base_scene()
            creater.save_base_scene()
        else:
            bpy.ops.wm.open_mainfile(filepath=base_scence_blend)

        modifier = SceneModifier(args.element_path, out_blend, slot_angle, lock=False, obstacle=False)
        modifier.modify_static_objs()
        modifier.add_dynamic_objs_on_roads()
        occupied_slots = modifier.add_dynamic_objs_in_slots()

        roads = bpy.data.collections.get('roads')
        if not roads:
            roads = bpy.data.collections.get('lanes')
        road = random.choice(roads.objects)
        # add_arrows_on_road(road, os.path.join(args.element_path, 'arrows'))
        ego_poses = get_ego_poses_for_road(road, occupied_slots)
        np.save(ego_poses_file, np.array(ego_poses))
        modifier.save_output_scene()
        dump_marking_pts(out_json_path, occupied_slots, slot_angle)
    else:
        bpy.ops.wm.open_mainfile(filepath=out_blend)

    render_img_mask(out_img_path, ego_poses_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='random PROCFRPS')
    parser.add_argument('--scene_spec', help='scene specifics', default='roadside')
    parser.add_argument('--element_path', help='elements path', default='/home/sczone/disk1/share/3d/blender_slots/elements')
    parser.add_argument('--out_path', help='output path')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)
