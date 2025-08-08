import argparse
import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from tracker.pinesort import PineSORT

def create_output_directory(output_path):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(output_path, exist_ok=True)
        print(f"Directory created at: {output_path}")
    except OSError as e:
        print(f"Error creating directory: {e}")

def make_parser():
    """Set up the argument parser for the script."""
    parser = argparse.ArgumentParser("PineSORT for Agriculture Multiple Object Tracking")

    # Dataset and Evaluation Arguments
    parser.add_argument("--path_video_imgs", default="./images", type=str, help="Path of the images of the video.")

    # Detector
    parser.add_argument("--model_path", default="./best.pt", type=str, help="Object Detector weights using Ultralytics YOLO.")

    # PineSORT Arguments
    parser.add_argument("--det_thresh", type=float, default=0.40, help="Detection threshold.")
    parser.add_argument("--min_det_thresh", type=float, default=0.30, help="Minimum detection threshold.")
    parser.add_argument("--max_age", type=int, default=5, help="Maximum number of frames to keep alive a track without detections.")
    parser.add_argument("--min_hits", type=int, default=1, help="Minimum number of hits before a track is confirmed.")
    parser.add_argument("--first_iou_threshold", type=float, default=0.30, help="First IOU threshold for association.")
    parser.add_argument("--second_iou_threshold", type=float, default=0.30, help="Second IOU threshold for association.")
    parser.add_argument("--third_iou_threshold", type=float, default=0.10, help="Third IOU threshold for association.")
    parser.add_argument("--overlap_iou_threshold", type=float, default=0.10, help="IOU threshold for suppressing redundant boxes with Kalman Filter predictions.")
    parser.add_argument("--delta_t", type=int, default=3, help="Time delta used for motion model update.")
    parser.add_argument("--inertia", type=float, default=0.2, help="Inertia coefficient for motion model.")
    parser.add_argument("--asso_func", type=str, default="eiou", help="Association function to use (e.g., 'iou', 'giou', 'eiou', 'ciou', 'diou).")
    parser.add_argument("--use_byte", action="store_true", help="Use ByteTrack-style association logic.")
    parser.add_argument("--img_width", type=int, default=1920, help="Image width.")
    parser.add_argument("--img_height", type=int, default=1088, help="Image height.")
    parser.add_argument("--camera_compensation", type=str, default="orb", help="Camera compensation technique, (e.g., 'orb', 'optisparseOptFlow', 'sift').")

    # Tracking Arguments
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="Tracking confidence threshold.")
    parser.add_argument("--track_low_thresh", type=float, default=0.1, help="Lowest detection threshold valid for tracks.")
    parser.add_argument("--new_track_thresh", type=float, default=0.7, help="New track threshold.")
    parser.add_argument("--track_buffer", type=int, default=3, help="Frames to keep lost tracks.")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="Matching threshold for tracking.")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Filter out boxes exceeding aspect ratio threshold.")
    parser.add_argument('--min_box_area', type=float, default=10, help='Filter out tiny boxes.')

    # Output save path
    parser.add_argument("--output", default="./output/track/", help="Output directory to store the tracker in MOT result, images with the ID's and in debug.")
    parser.add_argument("--output-debug", default=True, action="store_true", help="Save sequences with object detector and kalman filter predictions.")
    parser.add_argument("--output-frames", default=True, action="store_true", help="Save sequences with tracks.")

    return parser

def get_color_for_id(tid):
    """Generate a unique color for a given ID."""
    random.seed(tid)
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

if __name__ == "__main__":
    # Parse arguments
    args = make_parser().parse_args()

    # Define paths
    path_mot_dataset = os.path.join(args.path_video_imgs)
    output_base_path = args.output

    # Create directories
    create_output_directory(output_base_path)
    output_dir = os.path.join(output_base_path, "tracker")
    create_output_directory(output_dir)

    if args.output_frames:
        output_images_dir = os.path.join(output_dir, "images")
        create_output_directory(output_images_dir)
    
    output_debug_dir = None
    print("Output debug: ", args.output_debug)
    if args.output_debug:
        output_debug_dir = os.path.join(output_dir, "debug")
        create_output_directory(output_debug_dir)

    # Initialize object detector model and tracker
    model = YOLO(args.model_path)
    tracker = PineSORT()

    # Process images
    image_files = sorted([os.path.join(path_mot_dataset, f) for f in os.listdir(path_mot_dataset) if f.endswith(('png', 'jpg', 'jpeg'))])
    results = []

    for frame_id, image_file in enumerate(image_files, 1):
        prediction = model.predict(image_file)
        prediction_results = prediction[0].boxes.cpu()

        if prediction_results.shape[0] == 0:
            pred_concatenate = np.empty((0, 5))
        else:
            pred_conf = prediction_results.conf.reshape(-1, 1)
            pred_concatenate = np.concatenate((prediction_results.xyxy, pred_conf), axis=1)

        raw_img = cv2.imread(image_file)
        overlay = raw_img.copy() 

        img_info = [1920, 1088]
        img_size = [1920, 1088]

        path_save_debug = None
        if args.output_debug:
            path_save_debug = os.path.join(output_debug_dir, f"{frame_id:04d}.jpg")
        online_targets = tracker.update(pred_concatenate, raw_img, path_save_debug=path_save_debug)

        if args.output_frames:
            # Draw transparent bounding boxes for predictions
            alpha = 0.1  # Transparency level (0 = full transparent, 1 = full opaque)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

        # Draw bounding boxes for tracked objects
        for t in online_targets:
            tlwh = t[:4]
            tid = t[4]
            results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]-tlwh[0]:.2f},{tlwh[3]-tlwh[1]:.2f},-1,-1,-1,-1\n")

            if args.output_frames:
                color = get_color_for_id(tid)
                x1, y1, x2, y2 = map(int, tlwh)

                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), color, 2)

                # ID label (top-left corner)
                label = f"ID:{tid}"
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_w, text_h = text_size

                text_x, text_y = x1, max(y1 - 5, text_h + 5)  # Ensure it stays visible

                # Draw label background
                cv2.rectangle(raw_img, (text_x, text_y - text_h - 5), (text_x + text_w + 5, text_y), color, -1)

                # Draw label text
                cv2.putText(raw_img, label, (text_x + 2, text_y - 2), font, font_scale, (255, 255, 255), font_thickness)

        if args.output_frames:
            # Save the modified image
            cv2.addWeighted(overlay, alpha, raw_img, 1 - alpha, 0, raw_img)
            output_image_path = os.path.join(output_images_dir, f"{frame_id:04d}.jpg")
            cv2.imwrite(output_image_path, raw_img)

    # Save tracking results
    with open(os.path.join(output_base_path, f'output.txt'), 'w') as f:
        f.writelines(results)