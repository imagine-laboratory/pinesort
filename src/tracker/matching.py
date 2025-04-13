import scipy
import lap
import math
import functools
import numpy as np
import pygmtools as pygm

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Parameters:
    box1, box2: list or tuple of length 4
        The coordinates of the bounding boxes in the format (x1, y1, x2, y2).
    Returns:
    float
        The IoU between the two bounding boxes.
    """
    # Unpack the coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[:4]  # Only take the first 4 values
    x1_2, y1_2, x2_2, y2_2 = box2[:4]  # Only take the first 4 values

    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Compute the area of the intersection rectangle
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Compute the combined area
    combined_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if combined_area == 0:
        return 0.0

    # Compute the IoU
    iou = inter_area / combined_area
    return iou

def compute_diou(box1, box2):
    """
    Compute the Distance-IoU (DIoU) between two bounding boxes.

    Parameters:
    box1, box2: list or tuple of length 4
        The coordinates of the bounding boxes in the format (x1, y1, x2, y2).

    Returns:
    float
        The DIoU between the two bounding boxes.
    """
    # Compute IoU
    iou = compute_iou(box1, box2)

    # Compute the center points of the boxes
    center_x1 = (box1[0] + box1[2]) / 2
    center_y1 = (box1[1] + box1[3]) / 2
    center_x2 = (box2[0] + box2[2]) / 2
    center_y2 = (box2[1] + box2[3]) / 2

    # Compute the Euclidean distance between the centers
    center_distance = math.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)

    # Compute the diagonal distance of the smallest enclosing box
    x1_c = min(box1[0], box2[0])
    y1_c = min(box1[1], box2[1])
    x2_c = max(box1[2], box2[2])
    y2_c = max(box1[3], box2[3])
    diagonal_distance = math.sqrt((x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2)

    # Avoid division by zero
    if diagonal_distance == 0:
        return iou  # If diagonal distance is zero, return IoU as DIoU degenerates to IoU

    # Compute DIoU
    diou = iou - (center_distance ** 2) / (diagonal_distance ** 2)

    # Normalize DIoU to the range [0, 1]
    return diou #(diou + 1) / 2.0


def compute_ciou(bbox1, bbox2):
    """
    Calculate Complete Intersection over Union (CIoU) for two bounding boxes.

    :param bbox1: Predicted bounding box as (x1, y1, x2, y2)
    :param bbox2: Ground truth bounding box as (x1, y1, x2, y2)
    :return: CIoU score scaled between 0 and 1
    """
    epsilon = 1e-7  # Small value to prevent division by zero

    # Calculate the intersection box
    xx1 = max(bbox1[0], bbox2[0])
    yy1 = max(bbox1[1], bbox2[1])
    xx2 = min(bbox1[2], bbox2[2])
    yy2 = min(bbox1[3], bbox2[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    wh = w * h

    # Calculate IoU
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = wh / (area1 + area2 - wh + epsilon)

    # Calculate center points
    centerx1 = (bbox1[0] + bbox1[2]) / 2.0
    centery1 = (bbox1[1] + bbox1[3]) / 2.0
    centerx2 = (bbox2[0] + bbox2[2]) / 2.0
    centery2 = (bbox2[1] + bbox2[3]) / 2.0

    # Calculate squared center distance
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Calculate smallest enclosing box diagonal
    xxc1 = min(bbox1[0], bbox2[0])
    yyc1 = min(bbox1[1], bbox2[1])
    xxc2 = max(bbox1[2], bbox2[2])
    yyc2 = max(bbox1[3], bbox2[3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2 + epsilon

    # Calculate aspect ratio consistency
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    # Prevent division by zero
    h2 += epsilon
    h1 += epsilon
    arctan_diff = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi ** 2)) * (arctan_diff ** 2)

    # Calculate alpha
    S = 1 - iou
    alpha = v / (S + v + epsilon)

    # Compute CIoU
    ciou = iou - (inner_diag / outer_diag) + (alpha * v)

    # Scale CIoU to [0, 1]
    return ciou #(ciou + 1) / 2.0


def compute_giou(box1, box2):
    """
    Compute the Generalized-IoU (GIoU) between two bounding boxes.

    Parameters:
    box1, box2: list or tuple of length 4
        The coordinates of the bounding boxes in the format (x1, y1, x2, y2).

    Returns:
    float
        The GIoU between the two bounding boxes.
    """
    # Compute IoU
    iou = compute_iou(box1, box2)

    # Compute the area of the smallest enclosing box
    x1_c = min(box1[0], box2[0])
    y1_c = min(box1[1], box2[1])
    x2_c = max(box1[2], box2[2])
    y2_c = max(box1[3], box2[3])
    area_c = (x2_c - x1_c) * (y2_c - y1_c)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute GIoU
    giou = iou - (area_c - (box1_area + box2_area - iou * (box1_area + box2_area))) / area_c

    return giou #(giou + 1) / 2.0 

def compute_eiou(box1, box2):
    """
    Compute the Efficient IoU (EIoU) between two bounding boxes.

    Parameters:
    box1, box2: list or tuple of length 4
        The coordinates of the bounding boxes in the format (x1, y1, x2, y2).

    Returns:
    float
        The EIoU between the two bounding boxes.
    """
    # Compute IoU
    iou = compute_iou(box1, box2)

    # Compute the center points of the boxes
    center_x1 = (box1[0] + box1[2]) / 2
    center_y1 = (box1[1] + box1[3]) / 2
    center_x2 = (box2[0] + box2[2]) / 2
    center_y2 = (box2[1] + box2[3]) / 2

    # Compute the Euclidean distance between the centers
    center_distance = math.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)

    # Compute the diagonal distance of the smallest enclosing box
    x1_c = min(box1[0], box2[0])
    y1_c = min(box1[1], box2[1])
    x2_c = max(box1[2], box2[2])
    y2_c = max(box1[3], box2[3])
    diagonal_distance = math.sqrt((x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2)

    # Compute the width and height differences
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    width_diff = (w1 - w2) ** 2
    height_diff = (h1 - h2) ** 2

    # Compute the penalty terms for width and height
    cw = x2_c - x1_c  # Width of the smallest enclosing box
    ch = y2_c - y1_c  # Height of the smallest enclosing box

    # Compute EIoU
    eiou = iou - (center_distance ** 2) / (diagonal_distance ** 2) - (width_diff) / (cw ** 2) - (height_diff) / (ch ** 2)

    return eiou

def compute_bbsindex(bbox1, bbox2, iou_only=False):
    """
    Calculate the association cost based on IoU and box similarity for individual bounding boxes.
    :param bbox1: predict of bbox (4,) (x1, y1, x2, y2)
    :param bbox2: groundtruth of bbox (4,) (x1, y1, x2, y2)
    :param iou_only: if True, only compute IoU
    :return: cost value
    """
    eps = 1e-7

    # Calculate the intersection area
    xx1 = max(bbox1[0], bbox2[0])
    yy1 = max(bbox1[1], bbox2[1])
    xx2 = min(bbox1[2], bbox2[2])
    yy2 = min(bbox1[3], bbox2[3])
    
    w_intersection = max(0., xx2 - xx1)
    h_intersection = max(0., yy2 - yy1)
    intersection = w_intersection * h_intersection

    # Calculate the union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection + eps

    # Calculate IoU
    iou = intersection / union

    if iou_only:
        return 1.0 - iou

    # Calculate the DIoU using the diou function
    diou_val = compute_diou(bbox1, bbox2)

    # Calculate box similarity
    delta_w = abs((bbox2[2] - bbox2[0]) - (bbox1[2] - bbox1[0]))
    delta_h = abs((bbox2[3] - bbox2[1]) - (bbox1[3] - bbox1[1]))

    sw = w_intersection / (w_intersection + delta_w + eps)
    sh = h_intersection / (h_intersection + delta_h + eps)

    # Calculate the BBSI
    bbsi = diou_val + sh + sw

    # Normalize the BBSI
    cost = bbsi / 3.0

    return cost

def ciou_batch(bboxes1, bboxes2):
    """
    Compute Complete IoU (CIoU) between two sets of bounding boxes.
    
    :param bboxes1: Predicted bounding boxes of shape (N, 4) in format (x1, y1, x2, y2).
    :param bboxes2: Ground truth bounding boxes of shape (M, 4) in format (x1, y1, x2, y2).
    :return: Matrix of shape (N, M) containing CIoU values.
    """
    cious = np.zeros((len(bboxes1), len(bboxes2)))
    if cious.size == 0:
        return cious

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 

    # Center distance
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Enclosing box
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    # Aspect ratio penalty
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]
    h2 = h2 + 1.
    h1 = h1 + 1.
    arctan = np.arctan(w2/h2) - np.arctan(w1/h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou 
    alpha = v / (S+v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    
    return ciou #(ciou + 1) / 2.0  # Rescale to (0, 1)

def giou_batch(bboxes1, bboxes2):
    """
    Compute Generalized IoU (GIoU) between two sets of bounding boxes.
    
    :param bboxes1: Predicted bounding boxes of shape (N, 4) in format (x1, y1, x2, y2).
    :param bboxes2: Ground truth bounding boxes of shape (M, 4) in format (x1, y1, x2, y2).
    :return: Matrix of shape (N, M) containing GIoU values.
    """
    gious = np.zeros((len(bboxes1), len(bboxes2)))
    if gious.size == 0:
        return gious

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  

    # Enclosing box
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1 
    hc = yyc2 - yyc1 
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc 
    giou = iou - (area_enclose - wh) / area_enclose
    #giou = (giou + 1.)/2.0  # Rescale to (0, 1)
    return giou

def diou_batch(bboxes1, bboxes2):
    """
    Compute Distance IoU (DIoU) between two sets of bounding boxes.
    
    :param bboxes1: Predicted bounding boxes of shape (N, 4) in format (x1, y1, x2, y2).
    :param bboxes2: Ground truth bounding boxes of shape (M, 4) in format (x1, y1, x2, y2).
    :return: Matrix of shape (N, M) containing DIoU values.
    """
    dious = np.zeros((len(bboxes1), len(bboxes2)))
    if dious.size == 0:
        return dious
    
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 

    # Center distance
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Enclosing box
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return diou #(diou + 1) / 2.0  # Rescale to (0, 1)

def iou_batch(bboxes1, bboxes2):
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes.
    
    :param bboxes1: Predicted bounding boxes of shape (N, 4) in format (x1, y1, x2, y2).
    :param bboxes2: Ground truth bounding boxes of shape (M, 4) in format (x1, y1, x2, y2).
    :return: Matrix of shape (N, M) containing IoU values.
    """
    ious = np.zeros((len(bboxes1), len(bboxes2)))
    if ious.size == 0:
        return ious

    # Expand dimensions for broadcasting
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Compute intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h

    # Compute union
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - intersection

    # Compute IoU
    iou = intersection / union
    return iou

def eiou_batch(bboxes1, bboxes2):
    """
    Compute Efficient IoU (EIoU) between two sets of bounding boxes.
    
    :param bboxes1: Predicted bounding boxes of shape (N, 4) in format (x1, y1, x2, y2).
    :param bboxes2: Ground truth bounding boxes of shape (M, 4) in format (x1, y1, x2, y2).
    :return: Matrix of shape (N, M) containing EIoU values.
    """
    eiou = np.zeros((len(bboxes1), len(bboxes2)))
    if eiou.size == 0:
        return eiou

    # Expand dimensions for broadcasting
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Compute intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h

    # Compute union
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - intersection

    # Compute IoU
    iou = intersection / union

    # Compute aspect ratio penalty
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    alpha = v / (1 - iou + v)

    # Compute EIoU
    eiou = iou - (w1 - w2) ** 2 / (w1 ** 2 + w2 ** 2) - (h1 - h2) ** 2 / (h1 ** 2 + h2 ** 2) - alpha * v
    return eiou

def bbsindex_batch(bboxes1, bboxes2, iou_only=False):
    """Calculates the association cost based on IoU and box similarity in batch format."""
    eps = 1e-7

    # Calculate the intersection area
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w_intersection = np.maximum(0., xx2 - xx1)
    h_intersection = np.maximum(0., yy2 - yy1)
    intersection = w_intersection * h_intersection

    # Calculate the union area
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - intersection + eps

    # Calculate IoU
    iou = intersection / union

    if iou_only:
        return 1.0 - iou

    # Calculate the DIoU using the diou_batch function
    diou = diou_batch(np.squeeze(bboxes1, axis=1), np.squeeze(bboxes2, axis=0))

    # Calculate box similarity
    delta_w = np.abs((bboxes2[..., 2] - bboxes2[..., 0]) - (bboxes1[..., 2] - bboxes1[..., 0]))
    delta_h = np.abs((bboxes2[..., 3] - bboxes2[..., 1]) - (bboxes1[..., 3] - bboxes1[..., 1]))

    sw = w_intersection / (w_intersection + delta_w + eps)
    sh = h_intersection / (h_intersection + delta_h + eps)

    # Calculate the BBSI
    bbsi = diou + sh + sw

    # Normalize the BBSI
    cost = bbsi / 3.0

    return cost

def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1 - CX2 
    dy = CY1 - CY2 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight, association_fun="iou"):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)] = 0

    if association_fun == "iou":
        iou_matrix  = iou_batch(detections, trackers)
    elif association_fun == "diou":
        iou_matrix = diou_batch(detections, trackers)
    elif association_fun == "giou":
        iou_matrix = giou_batch(detections, trackers)
    elif association_fun == "ciou":
        iou_matrix = ciou_batch(detections, trackers)
    else:
        iou_matrix = eiou_batch(detections, trackers)

    #iou_matrix = diou_matrix
    #dbiou_matrix = biou_batch(detections, trackers, 1, func=compute_iou)
    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    #iou_matrix = dbiou_matrix #0.5*iou_matrix + 0.5*diou_matrix
    #iou_matrix = 1*dbiou_matrix #+ 0.5*diou_matrix

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix+angle_diff_cost))
            #matched_indices = linear_assignment(-(iou_matrix))
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]< iou_threshold):
            
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def associate_without_speed(detections, trackers, iou_threshold, association_fun="iou"): #, velocities, previous_obs, vdc_weight):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    if association_fun == "iou":
        iou_matrix  = iou_batch(detections, trackers)
    elif association_fun == "diou":
        iou_matrix = diou_batch(detections, trackers)
    elif association_fun == "giou":
        iou_matrix = giou_batch(detections, trackers)
    elif association_fun == "ciou":
        iou_matrix = ciou_batch(detections, trackers)
    else:
        iou_matrix = eiou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix))
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def calculate_affinity_cost_matrix(bbox1, bbox2, alpha=1):
    
    # Initialize cost matrix
    num_bbox1 = len(bbox1)
    num_bbox2 = len(bbox2)
    cost_matrix = np.zeros((num_bbox1, num_bbox2))

    # Calculate cost matrix
    for i, box in enumerate(bbox1):
        for j, track in enumerate(bbox2):
            cost = 0.5*compute_bbsindex(box, track) + 0.5*compute_ciou(box, track) #compute_ciou(box, track) #+ calculate_diou(box, track) #euclidean_distance(box, track)
            cost_matrix[i, j] = cost  # Negative for cost minimization
    return cost_matrix


def compute_overlap_tracks(tracks, detection, overlap_threshold=0.6):
    for track in tracks:
        diou_value = compute_diou(track.get_state()[0], detection)
        if overlap_threshold <= diou_value:
            return True 
    return False

def buffered(box, b):
    """
    Return a new box whose width and height have been inflated by a coefficient b.
    This is useful to compute BIoU.
    
    Parameters:
    box: list or tuple of length 4
        The coordinates of the bounding box in the format (x1, y1, x2, y2).
    b: float
        The coefficient by which to inflate the box.
    
    Returns:
    list
        The coordinates of the buffered box in the format (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    width = x2 - x1
    height = y2 - y1
    return [x1 - b * width / 2, y1 - b * height / 2, x2 + b * width / 2, y2 + b * height / 2]

def compute_biou(box1, box2, b, func=None):
    """
    Computes the Buffered Intersection over Union (BIoU) at the box level.
    ref: https://arxiv.org/abs/2211.14317
    
    Parameters:
    box1, box2: list or tuple of length 4
        The coordinates of the bounding boxes in the format (x1, y1, x2, y2).
    b: float
        The coefficient by which to inflate the boxes.
    
    Returns:
    float
        The BIoU between the two bounding boxes.
    """
    if box1 is None or box2 is None:
        return 0.0
    
    buffered_box1 = buffered(box1, b)
    buffered_box2 = buffered(box2, b)
    return func(buffered_box1, buffered_box2)

def buffered_batch(boxes, b):
    """
    Return new boxes whose width and height have been inflated by a coefficient b.
    This is useful to compute BIoU.
    
    Parameters:
    boxes: numpy array of shape (N, 4)
        The coordinates of the bounding boxes in the format (x1, y1, x2, y2).
    b: float
        The coefficient by which to inflate the boxes.
    
    Returns:
    numpy array of shape (N, 4)
        The coordinates of the buffered boxes in the format (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    widths = x2 - x1
    heights = y2 - y1
    buffered_boxes = np.stack([
        x1 - b * widths / 2,
        y1 - b * heights / 2,
        x2 + b * widths / 2,
        y2 + b * heights / 2
    ], axis=1)
    return buffered_boxes

def compute_biou_batch(boxes1, boxes2, b, func=None):
    """
    Computes the Buffered Intersection over Union (BIoU) at the box level for batches of boxes.
    ref: https://arxiv.org/abs/2211.14317
    
    Parameters:
    boxes1, boxes2: numpy arrays of shape (N, 4)
        The coordinates of the bounding boxes in the format (x1, y1, x2, y2).
    b: float
        The coefficient by which to inflate the boxes.
    func: callable
        A function that computes the IoU between two bounding boxes.
    
    Returns:
    numpy array of shape (N,)
        The BIoU between the corresponding pairs of bounding boxes.
    """
    if boxes1 is None or boxes2 is None:
        return np.zeros((boxes1.shape[0],))
    
    buffered_boxes1 = buffered_batch(boxes1, b)
    buffered_boxes2 = buffered_batch(boxes2, b)
    
    biou_scores = np.zeros((boxes1.shape[0],))
    for i in range(boxes1.shape[0]):
        biou_scores[i] = func(buffered_boxes1[i], buffered_boxes2[i])
    
    return biou_scores

def biou_batch(boxes1, boxes2, b, func=None):
    """
    Builds a cost matrix based on the Buffered IoU (BIoU) between two sets of bounding boxes.
    
    Parameters:
    boxes1: numpy array of shape (M, 4)
        The first set of bounding boxes in the format (x1, y1, x2, y2).
    boxes2: numpy array of shape (N, 4)
        The second set of bounding boxes in the format (x1, y1, x2, y2).
    b: float
        The coefficient by which to inflate the boxes for BIoU computation.
    func: callable
        A function that computes the IoU between two bounding boxes.
    
    Returns:
    numpy array of shape (M, N)
        The cost matrix where each element (i, j) represents the cost (1 - BIoU) between boxes1[i] and boxes2[j].
    """
    M = boxes1.shape[0]
    N = boxes2.shape[0]
    cost_matrix = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            # Compute BIoU between boxes1[i] and boxes2[j]
            biou = compute_biou_batch(boxes1[np.newaxis, i], boxes2[np.newaxis, j], b, func)[0]
            # Cost is 1 - BIoU (since higher IoU means lower cost)
            cost_matrix[i, j] = biou
    
    return cost_matrix