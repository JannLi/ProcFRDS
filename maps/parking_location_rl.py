import bpy
import json
import numpy as np

def process_collects(collect_name):
    collect = bpy.data.collections.get(collect_name)
    res = []
    if not collect:
        return res
    for obj in collect.objects:
        target = dict()
        target['typ'] = 'l1n'
        target['pos'] = [obj.location[0], obj.location[1], obj.rotation_euler[-1]+np.pi/2]
        target['scl'] = list(obj.dimensions[:2])
        # if collect_name == 'slots':
        #     target['scl'] = [5.5, 2.5]
        res.append(target)
    return res

def process_objs():
    bst = dict()
    bst['nop'] = list()
    bst['nop'] += process_collects('pillars')
    bst['nop'] += process_collects('limiters')
    bst['nop'] += process_collects('walls')
    bst['nop'] += process_collects('locks')
    bst['nop'] += process_collects('obstacles')
    cars = []
    for obj in bpy.data.objects:
        if obj.name.startswith('body'):
            target = dict()
            target['type'] = 'l1n'
            target['pos'] = [obj.location[0], obj.location[1], obj.rotation_euler[-1]]
            target['scl'] = list(obj.dimensions[:2])
            cars.append(target)
    bst['nop'] += cars
    bst['nep'] = None
    bst['avp'] = None

    mrk = dict()
    mrk['spt'] = list()
    mrk['spt'] += process_collects('slots')
    mrk['rod'] = None
    return bst, mrk

def process_json(bst, mrk, json_file):
    out = dict()
    out['dyn'] = dict()
    out['dyn']['veh'] = list()
    ego = dict()
    ego['type'] = 'ego'
    ego['m'] = 1
    ego['v'] = [0, 5]
    ego['scl'] = [4.5, 2, 1.5]
    ego['pos_enb'] = [13.5981, -7.89906, 0, 0, np.pi]
    ego['pos_end'] = [7.75362, 1.234739, 0, 0, -np.pi/2]
    out['dyn']['veh'].append(ego)

    out['bst'] = bst
    out['mrk'] = mrk
    with open(json_file, 'w') as f:
        json.dump(out, f, indent=4)
    

if __name__ == '__main__':
    blend_path = '/home/sczone/disk1/share/3d/blender_slots/code/results/3.blend'
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    json_file = '/home/sczone/disk1/share/3d/blender_slots/code/results/3_obj.json'
    bst, mrk = process_objs()
    process_json(bst, mrk, json_file)