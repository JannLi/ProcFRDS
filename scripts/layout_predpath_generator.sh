while true; do
        if ! pgrep -f "layout_predpath_generator.py" > /dev/null; then
                python layout_predpath_generator.py --layout_path ../test/layouts/$1.json --scene_spec $2 --ego_pose_path ../test/predpaths/$1.npy --out_path $3
                python layout_predpath_generator.py --layout_path ../test/layouts/$4.json --scene_spec $5 --ego_pose_path ../test/predpaths/$4.npy --out_path $6
        fi
done
