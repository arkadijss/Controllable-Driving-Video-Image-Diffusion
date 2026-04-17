import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from pathlib import Path

IMAGE_FOLDER = Path("~/data/KITTI-360_proc/lora/center_cropped").expanduser()
OUTPUT_JSONL_PATH = "data/lora/metadata.jsonl"
LORA_TOKEN = "KITTI-360 style"
backup_every_n_steps = 100

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

output_metdata = {}

qwen_prompt = "Generate concise visual tags for this image, separated by commas. Include time of day and weather. No sentences."
print(f"Using prompt: {qwen_prompt}")

with open(OUTPUT_JSONL_PATH, "w") as f:
    for idx, image_path in enumerate(IMAGE_FOLDER.glob("*.png")):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {"type": "text", "text": qwen_prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        caption = output_text[0].strip()
        entry = {
            "file_name": image_path.name,
            "text": f"{LORA_TOKEN}, {caption}"
        }

        print(f"Processed image {idx}, entry: {entry}")

        # Write the entry to the JSONL file
        f.write(json.dumps(entry) + "\n")

        if (idx + 1) % backup_every_n_steps == 0:
            f.flush()
    
    # Final flush to ensure all data is written to disk
    f.flush()

print(f"Metadata saved to {OUTPUT_JSONL_PATH}")
