python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "canny" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v10_canny" \
    --video_length 32 \
    --smoother_steps 19 20 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v10 \
    # --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "depth_midas" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v10_depth" \
    --video_length 32 \
    --smoother_steps 19 20 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v10 \
    # --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "canny" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v11_canny" \
    --video_length 32 \
    --smoother_steps 19 20 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v11 \
    # --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "depth_midas" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v11_depth" \
    --video_length 32 \
    --smoother_steps 19 20 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v11 \
    # --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "canny" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v10_canny_long" \
    --video_length 32 \
    --smoother_steps 19 20 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v10 \
    --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "depth_midas" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v10_depth_long" \
    --video_length 32 \
    --smoother_steps 19 20 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v10 \
    --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "canny" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v11_canny_long" \
    --video_length 32 \
    --smoother_steps 19 20 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v11 \
    --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "depth_midas" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v11_depth_long" \
    --video_length 32 \
    --smoother_steps 19 20 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v11 \
    --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "depth_midas" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v11_depth_smoother_9_10" \
    --video_length 32 \
    --smoother_steps 9 10 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v11 \
    # --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "depth_midas" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v11_depth_smoother_29_30" \
    --video_length 32 \
    --smoother_steps 29 30 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v11 \
    # --is_long_video

python inference.py \
    --prompt "a driving scene in a town, photo-realistic, sunny weather" \
    --condition "depth_midas" \
    --video_path "data/KITTI_0018.mp4" \
    --output_path "outputs/KITTI_v11_depth_smoother_39_40" \
    --video_length 32 \
    --smoother_steps 39 40 \
    --width 1224 \
    --height 370 \
    --frame_rate 1 \
    --version v11 \
    # --is_long_video
