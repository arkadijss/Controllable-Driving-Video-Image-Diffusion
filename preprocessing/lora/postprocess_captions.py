import json


def postprocess_caption(text, max_tags=25, lora_token="KITTI-360 style"):
    text = text.replace(lora_token, "")
    text = text.strip(".").strip(",")

    unique_tags = []
    seen_lower = set()
    for tag in text.split(","):
        tag = tag.strip().strip(".").strip(",")

        tag_lower = tag.lower()

        if tag_lower not in seen_lower:
            unique_tags.append(tag)
            seen_lower.add(tag_lower)

    unique_tags = unique_tags[:max_tags]
    assert len(unique_tags) != 0, "No tags found after postprocessing."
    text = ", ".join(unique_tags)
    text = f"{lora_token}, {text}"

    return text


def postprocess_captions(input_jsonl_path, output_jsonl_path, max_tags=25):
    with open(input_jsonl_path, "r") as f:
        lines = f.readlines()

    with open(output_jsonl_path, "w") as f:
        for line in lines:
            data = json.loads(line)
            text = data["text"]
            text = postprocess_caption(text, max_tags=max_tags)
            data["text"] = text
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    metadata_jsonl_path = "data/lora/metadata.jsonl"
    output_jsonl_path = "data/lora/metadata_postprocessed.jsonl"
    max_tags = 25
    postprocess_captions(metadata_jsonl_path, output_jsonl_path, max_tags=max_tags)
