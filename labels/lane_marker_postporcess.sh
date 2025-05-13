lane_type=$1
for ((i=1;i<=10;i++))
do
	python lane_marker_postporcess.py --mask_path ../../lane_marker/results/${lane_type}/${i}/masks/ --out_path ../../lane_marker/results/${lane_type}/${i}/masks2/ --line_type ${lane_type}
done
