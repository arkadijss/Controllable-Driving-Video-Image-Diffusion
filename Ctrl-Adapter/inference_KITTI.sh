python inference.py \
    --model_name "i2vgenxl" \
    --control_types "depth" \
    --huggingface_checkpoint_folder "i2vgenxl_depth" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --n_sample_frames 16 \
    --extract_control_conditions True \
    --num_inference_steps 50 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_output_folder "outputs/i2vgenxl_depth"

python inference.py \
    --model_name "i2vgenxl" \
    --control_types "canny" \
    --huggingface_checkpoint_folder "i2vgenxl_canny" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --n_sample_frames 16 \
    --extract_control_conditions True \
    --num_inference_steps 50 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_output_folder "outputs/i2vgenxl_canny"

python inference.py \
    --model_name "i2vgenxl" \
    --control_types "softedge" \
    --huggingface_checkpoint_folder "i2vgenxl_softedge" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --n_sample_frames 16 \
    --extract_control_conditions True \
    --num_inference_steps 50 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_output_folder "outputs/i2vgenxl_softedge"

python inference.py \
    --model_name "svd" \
    --control_types "depth" \
    --huggingface_checkpoint_folder "svd_depth" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --skip_conv_in True \
    --n_sample_frames 14 \
    --extract_control_conditions True \
    --num_inference_steps 25 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_output_folder "outputs/svd_depth"

python inference.py \
    --model_name "svd" \
    --control_types "canny" \
    --huggingface_checkpoint_folder "svd_canny" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --skip_conv_in True \
    --n_sample_frames 14 \
    --extract_control_conditions True \
    --num_inference_steps 25 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_output_folder "outputs/svd_canny"

python inference.py \
    --model_name "svd" \
    --control_types "softedge" \
    --huggingface_checkpoint_folder "svd_softedge" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --skip_conv_in True \
    --n_sample_frames 14 \
    --extract_control_conditions True \
    --num_inference_steps 25 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_output_folder "outputs/svd_softedge"

python inference.py \
    --model_name "i2vgenxl" \
    --control_types "depth" "segmentation" \
    --huggingface_checkpoint_folder "i2vgenxl_multi_control_adapter" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --extract_control_conditions True \
    --n_sample_frames 16 \
    --num_inference_steps 50 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_prompt_file "captions_multi.json" \
    --evaluation_output_folder "outputs/i2vgenxl_multi_control_adapter_depth_segmentation"

python inference.py \
    --model_name "i2vgenxl" \
    --control_types "canny" "segmentation" \
    --huggingface_checkpoint_folder "i2vgenxl_multi_control_adapter" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --extract_control_conditions True \
    --n_sample_frames 16 \
    --num_inference_steps 50 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_prompt_file "captions_multi.json" \
    --evaluation_output_folder "outputs/i2vgenxl_multi_control_adapter_canny_segmentation"

python inference.py \
    --model_name "i2vgenxl" \
    --control_types "depth" "canny" \
    --huggingface_checkpoint_folder "i2vgenxl_multi_control_adapter" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --extract_control_conditions True \
    --n_sample_frames 16 \
    --num_inference_steps 50 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_prompt_file "captions_multi.json" \
    --evaluation_output_folder "outputs/i2vgenxl_multi_control_adapter_depth_canny"

python inference.py \
    --model_name "i2vgenxl" \
    --control_types "depth" "segmentation" "openpose" \
    --huggingface_checkpoint_folder "i2vgenxl_multi_control_adapter" \
    --eval_input_type "frames" \
    --evaluation_input_folder "assets/evaluation/frames" \
    --extract_control_conditions True \
    --n_sample_frames 16 \
    --num_inference_steps 50 \
    --control_guidance_end 0.8 \
    --height 512 \
    --width 512 \
    --evaluation_prompt_file "captions_multi.json" \
    --evaluation_output_folder "outputs/i2vgenxl_multi_control_adapter_depth_segmentation_openpose"
