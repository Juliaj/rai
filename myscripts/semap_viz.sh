python -m rai_semap.ros2.visualizer \
    --ros-args \
    -p database_path:=semantic_map.db \
    -p location_id:=default_location \
    -p update_rate:=1.0 \
    -p marker_scale:=0.3 \
    -p show_text_labels:=true