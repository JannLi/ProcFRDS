lane_type=$1
for ((i=1;i<=25;i++))
do
	python vlm_postprocess.py /home/sczone/disk1/share/3d/blender_slots/lane_marker/vlm/$i
done
