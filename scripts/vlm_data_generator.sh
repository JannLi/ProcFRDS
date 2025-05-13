for ((i=2;i<=25;i++));do
    python vlm_data_generator.py --ego_pose_path ../../lane_marker/vlm/${i}/ego.npy --out_path ../../lane_marker/vlm/${i}/
done