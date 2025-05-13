import sys
sys.path.append('../')
import bpy
import random
import time
import numpy as np
import csv
import pandas as pd
from generation.rendering import render_img_mask

def get_ego_astart(csv_path, ego_poses_file):
    ego_poses = []
    with open(csv_path, 'r') as f:
        data = csv.reader(f)
        for row in data:
            if row[0].strip().startswith('x'):
                continue
            infos = [float(item) for item in row[0].strip().split('\t')]
            pose = np.array([0, 0, infos[3], infos[1], infos[2], 0])
            ego_poses.append(pose)
    if len(ego_poses) > 200:
        ego_poses = random.sample(ego_poses, 200)   
    np.save(ego_poses_file, np.array(ego_poses))

def get_ego_opendrive(csv_path, ego_poses_file):
    ego_poses = []
    df = pd.read_csv(csv_path, encoding='utf-8')
    df_loc = df.drop('time', axis=1)
    df_loc = df_loc.drop_duplicates()
    for _, row in df_loc.iterrows():
        pose = np.array([0, 0, float(row['head']), float(row['x']), float(row['y']), 0, float(row['time'])])
        ego_poses.append(pose)
    if len(ego_poses) > 500:
        pose_index = random.sample(range(len(ego_poses)), 500)
        pose_index.sort()
        ego_poses = [ego_poses[i] for i in pose_index]
    # ego_poses = [item+np.array([0, 0, 0, -48.33, -11.27, 0]) for item in ego_poses] 
    np.save(ego_poses_file, np.array(ego_poses))   

def main(args):
    bpy.ops.wm.open_mainfile(filepath=args.blend_path)
    ego_poses_file = './ego.npy'
    if args.path_spec == 'astar':
        get_ego_astart(args.path_file, ego_poses_file)
    elif args.path_spec == 'opendrive':
        get_ego_opendrive(args.path_file, ego_poses_file)

    render_img_mask(args.out_img_path, ego_poses_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='render with predpath')
    parser.add_argument('--blend_path', help='blend path')
    parser.add_argument('--path_file', help='path file')
    parser.add_argument('--path_spec', help='astar, opendrive, rl', default='astar')
    parser.add_argument('--out_img_path', help='rendered images')
    args = parser.parse_args()

    start = time.time()
    main(args)
    print(time.time()-start)