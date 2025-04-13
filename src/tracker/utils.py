import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import numpy as np

def plot_and_save_image_with_detections(cv_image, bounding_boxes, detections, output_path, alpha=0.1, thickness=3):
    """
    Plots bounding boxes and detections on an image with transparency and saves the result.

    Parameters:
        cv_image (numpy.ndarray): Input image in OpenCV format (BGR).
        bounding_boxes (list): List of bounding boxes [x_min, y_min, x_max, y_max, score].
        detections (list): List of detections [x_min, y_min, x_max, y_max, score].
        output_path (str): Path to save the output image.
        alpha (float): Transparency level for bounding boxes (0 = fully transparent, 1 = opaque).
        thickness (int): Line thickness of bounding boxes.
    """
    if output_path == None:
        return
    else:
        # Convert BGR to RGB (if needed)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Create overlay for transparency
        overlay = rgb_image.copy()

        # Draw bounding boxes in blue
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max, _ = map(int, box)  # Convert to int
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)  # Blue box
            #cv2.putText(overlay, f'{x_min}', (x_min, y_min - 5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness)

        # Draw detections in red
        for detection in detections:
            x_min, y_min, x_max, y_max, _ = map(int, detection)  # Convert to int
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), -1)  # Red box
            #cv2.putText(overlay, f'{x_min}', (x_max + 5, y_min),  
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, rgb_image, 1 - alpha, 0, rgb_image)

        # Save the image
        cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_bbox_to_xywh(bbox):
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and returns it in the form
    [x, y, w, h] where x, y is the center of the box, and w and h are width and height.
    """
    w = bbox[2] - bbox[0]  # width
    h = bbox[3] - bbox[1]  # height
    x = bbox[0] + w / 2.   # center x
    y = bbox[1] + h / 2.   # center y
    return np.array([x, y, w, h]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

def convert_xywh_to_bbox(x, score=None):
    """
    Takes a bounding box in the form [x, y, w, h] and returns it in the form
    [x1, y1, x2, y2], where x1, y1 is the top left and x2, y2 is the bottom right.
    """
    x1 = x[0] - x[2] / 2.  # Top-left x-coordinate
    y1 = x[1] - x[3] / 2.  # Top-left y-coordinate
    x2 = x[0] + x[2] / 2.  # Bottom-right x-coordinate
    y2 = x[1] + x[3] / 2.  # Bottom-right y-coordinate
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        score = np.array([score])
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))
    
def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in x1, y1, x2, y2 format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of the intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Compute the areas of the individual boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def nms_with_max_overlap(boxes, threshold=0.22):
    """
    Perform Non-Maximum Suppression (NMS), keeping the box with the largest shared area.

    Args:
        boxes: List of [x_min, y_min, x_max, y_max, score]
        threshold: IoU threshold to determine overlapping boxes

    Returns:
        List of filtered boxes.
    """
    if len(boxes) == 0:
        return np.empty((0, 5))

    # Sort boxes by score in descending order
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    final_boxes = []

    while boxes:
        current_box = boxes.pop(0)
        final_boxes.append(current_box)

        non_overlapping = []
        max_overlap_box = None
        max_shared_area = 0

        for box in boxes:
            iou = compute_iou(current_box, box)
            #print("IOU: ", iou)
            if iou > threshold:
                # Calculate shared area
                shared_area = compute_iou(current_box, box) * min(
                    (current_box[2] - current_box[0]) * (current_box[3] - current_box[1]),
                    (box[2] - box[0]) * (box[3] - box[1]),
                )

                if shared_area > max_shared_area:
                    max_shared_area = shared_area
                    max_overlap_box = box
            else:
                non_overlapping.append(box)

        if max_overlap_box is not None:
            final_boxes.append(max_overlap_box)

        # Update boxes with non-overlapping ones
        boxes = non_overlapping

    return np.array(final_boxes)

def non_maximum_suppression(boxes, iou_threshold=0.25):
    """
    Perform Non-Maximum Suppression (NMS) on the boxes.
    
    Args:
        boxes (list or np.ndarray): List of boxes with scores, format [x1, y1, x2, y2, score].
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        list: Filtered boxes after applying NMS.
    """
    if len(boxes) == 0:
        return np.empty((0, 5))
    
    # Convert to numpy array if not already
    boxes = np.array(boxes)
    
    # Extract the coordinates and scores
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    
    # Sort boxes by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    filtered_boxes = []
    
    while len(sorted_indices) > 0:
        # Take the box with the highest score
        current_index = sorted_indices[0]
        current_box = boxes[current_index]
        filtered_boxes.append(current_box)
        
        # Compare this box with the rest
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]
        
        # Compute IoUs with the remaining boxes
        ious = np.array([compute_iou(current_box, box) for box in remaining_boxes])
        #print(ious)
        
        # Filter out boxes with IoU > iou_threshold
        sorted_indices = remaining_indices[ious <= iou_threshold]
    
    return np.array(filtered_boxes)

def log_cost_matrix(matrix, detections, tracks, logger):
    for id_det, det in enumerate(detections):
        for id_track, track in enumerate(tracks):
            logger.info(f"Track {track} - Det {det} = {matrix[id_det, id_track]}")
    
def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures a logger with the specified name, log file, and level.

    Args:
        name (str): Name of the logger.
        log_file (str): File path where the logs will be saved.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Avoid duplicate logs by ensuring the logger has a single handler
    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger


def merge_indexes(original_array, filtered_array, indices_filtered):
    result = []
    for idx in indices_filtered:
        matching_indices = np.where((original_array == filtered_array[idx]).all(axis=1))[0]
        result.extend(matching_indices)

    result = np.array(result)
    return result 

def merge_indexes_matches(
        original_track, filtered_track, original_det, filtered_det, indices_filtered):
    if indices_filtered.shape[0] == 0:
        return indices_filtered
    result = []
    for idx in indices_filtered:
        trk_ind = np.where((original_track == filtered_track[idx[1]]).all(axis=1))[0][0]
        det_ind = np.where((original_det == filtered_det[idx[0]]).all(axis=1))[0][0]

        result.append([det_ind, trk_ind])

    result = np.array(result)
    return result 

def is_out_margin(box, width, height):
    """
    Filters out bounding boxes that are completely outside the specified width and height.

    Parameters:
    bounding_boxes (list of lists): List of bounding boxes, where each bounding box is represented as [x1, y1, x2, y2, confidence].
    width (int): The width of the image.
    height (int): The height of the image.

    Returns:
    list of lists: Filtered list of bounding boxes that are within the specified width and height.
    """
    #filtered_boxes = []
    
    #for box in bounding_boxes:
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    
    # Check if the bounding box is completely out the image dimensions
    if (x2 < 0 or x1 > width or y2 < 0 or y1 > height):
        return True 
    return False