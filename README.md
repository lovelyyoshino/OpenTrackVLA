ros2 launch trackvla_ros trackvla.launch.py     server_url:=http://localhost:12180     input_type:=go2_front_video     go2_raw_encoding:=bgr8     go2_video_resolution:=360     max_inference_fps:=4.0     publish_source_image:=true     source_image_fps:=3.0     image_topic:=/frontvideostream     cmd_vel_topic:=/cmd_vel     instruction:="Follow the target person with yellow coat and white pants"

