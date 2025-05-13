lines=("dd" "ss" "ds" "sd")
for line in "${lines[@]}"; do
	line_type=${line}ds
	for ((i=1;i<=10;i++));do
		python lane_marker_generator.py --ego_pose_path ../../lane_marker/results/$line_type/${i}/ego.npy --out_path ../../lane_marker/results/$line_type/${i}/ --line_type $line_type
	done
done
