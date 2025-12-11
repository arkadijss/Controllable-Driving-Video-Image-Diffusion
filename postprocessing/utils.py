import cv2


def frames_to_video(frame_paths, video_output_path, fps):
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width, _ = first_frame.shape

    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        writer.write(frame)

    writer.release()


def video_to_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0

    if not cap.isOpened():
        print("Error opening video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        save_file = output_dir / f"{frame_count:03d}.png"
        cv2.imwrite(str(save_file), frame)

        frame_count += 1

    cap.release()
