"""This file defines nms functions to merge boxes"""

import time

import cv2
import numpy as np
from shapely.geometry import Polygon

def boxes_3d_to_corners(boxes_3d):
    all_corners = []
    for box_3d in boxes_3d:
        x3d, y3d, z3d, l, h, w, yaw = box_3d
        R = np.array([[np.cos(yaw),  0,  np.sin(yaw)],
                      [0,            1,  0          ],
                      [-np.sin(yaw), 0,  np.cos(yaw)]]);
        corners = np.array([[ l/2,  0.0,  w/2],  # front up right
                            [ l/2,  0.0, -w/2],  # front up left
                            [-l/2,  0.0, -w/2],  # back up left
                            [-l/2,  0.0,  w/2],  # back up right
                            [ l/2, -h,  w/2],  # front down right
                            [ l/2, -h, -w/2],  # front down left
                            [-l/2, -h, -w/2],  # back down left
                            [-l/2, -h,  w/2]]) # back down right
        r_corners = corners.dot(np.transpose(R))
        cam_points_xyz = r_corners+np.array([x3d, y3d, z3d])
        all_corners.append(cam_points_xyz)
    return np.array(all_corners)

def overlapped_boxes_3d(single_box, box_list):
    x0_max, y0_max, z0_max = np.max(single_box, axis=0)
    x0_min, y0_min, z0_min = np.min(single_box, axis=0)
    overlap = np.zeros(len(box_list))
    for i, box in enumerate(box_list):
        x_max, y_max, z_max = np.max(box, axis=0)
        x_min, y_min, z_min = np.min(box, axis=0)
        if x0_max < x_min or x0_min > x_max:
            overlap[i] = 0
            continue
        if y0_max < y_min or y0_min > y_max:
            overlap[i] = 0
            continue
        if z0_max < z_min or z0_min > z_max:
            overlap[i] = 0
            continue
        x_draw_min = min(x0_min, x_min)
        x_draw_max = max(x0_max, x_max)
        z_draw_min = min(z0_min, z_min)
        z_draw_max = max(z0_max, z_max)
        offset = np.array([x_draw_min, z_draw_min])
        buf1 = np.zeros((z_draw_max-z_draw_min, x_draw_max-x_draw_min),
            dtype=np.int32)
        buf2 = np.zeros_like(buf1)
        cv2.fillPoly(buf1, [single_box[:4, [0,2]]-offset], color=1)
        cv2.fillPoly(buf2, [box[:4, [0,2]]-offset], color=1)
        shared_area = cv2.countNonZero(buf1*buf2)
        area1 = cv2.countNonZero(buf1)
        area2 = cv2.countNonZero(buf2)
        shared_y = min(y_max, y0_max) - max(y_min, y0_min)
        intersection = shared_y * shared_area
        union = (y_max-y_min) * area2 + (y0_max-y0_min) * area1
        overlap[i] = np.float32(intersection) / (union - intersection)
    return overlap

def overlapped_boxes_3d_fast_poly(single_box, box_list):
    single_box_max_corner = np.max(single_box, axis=0)
    single_box_min_corner = np.min(single_box, axis=0)
    x0_max, y0_max, z0_max = single_box_max_corner
    x0_min, y0_min, z0_min = single_box_min_corner
    max_corner = np.max(box_list, axis=1)
    min_corner = np.min(box_list, axis=1)
    overlap = np.zeros(len(box_list))
    non_overlap_mask =  np.logical_or(single_box_max_corner < min_corner,
        single_box_min_corner > max_corner)
    non_overlap_mask = np.any(non_overlap_mask, axis=1)
    p1  = Polygon(single_box[:4, [0,2]])
    area1 = p1.area
    for i in range(len(box_list)):
        if not non_overlap_mask[i]:
            x_max, y_max, z_max = max_corner[i]
            x_min, y_min, z_min = min_corner[i]
            p2 =  Polygon(box_list[i][:4, [0,2]])
            shared_area = p1.intersection(p2).area
            area2 = p2.area
            shared_y = min(y_max, y0_max) - max(y_min, y0_min)
            intersection = shared_y * shared_area
            union = (y_max-y_min) * area2 + (y0_max-y0_min) * area1
            overlap[i] = np.float32(intersection) / (union - intersection)
    return overlap

def bboxes_sort(classes, scores, bboxes, top_k=400, attributes=None):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    idxes = np.argsort(-scores)
    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    if attributes is not None:
        attributes = attributes[idxes]
    if top_k > 0:
        if len(idxes) > top_k:
            classes = classes[:top_k]
            scores = scores[:top_k]
            bboxes = bboxes[:top_k]
            if attributes is not None:
                attributes = attributes[:top_k]
    return classes, scores, bboxes, attributes

def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45,
    overlapped_fn=overlapped_boxes_3d, appr_factor=10.0, attributes=None):
    """Apply non-maximum selection to bounding boxes.
    """
    boxes_corners = boxes_3d_to_corners(bboxes)
    # convert to pixels
    boxes_corners = np.int32(boxes_corners*appr_factor)
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = overlapped_fn(boxes_corners[i], boxes_corners[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(
                overlap <= nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(
                keep_bboxes[(i+1):], keep_overlap)##
    idxes = np.where(keep_bboxes)
    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    if attributes is not None:
        attributes = attributes[idxes]
    return classes, scores, bboxes, attributes

def bboxes_nms_uncertainty(classes, scores, bboxes, scores_threshold=0.25,
    nms_threshold=0.45, overlapped_fn=overlapped_boxes_3d, appr_factor=10.0,
    attributes=None):
    """Apply non-maximum selection to bounding boxes.
    """
    boxes_corners = boxes_3d_to_corners(bboxes)
    # boxes_corners = bboxes
    # convert to pixels
    # boxes_corners = np.int32(boxes_corners*appr_factor)
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Only compute on the rest of bboxes
            valid = keep_bboxes[(i+1):]
            # Computer overlap with bboxes which are following.
            overlap = overlapped_fn(
                boxes_corners[i], boxes_corners[(i+1):][valid])
            # Overlap threshold for keeping + checking part of the same class
            remove_overlap = np.logical_and(
                overlap > nms_threshold, classes[(i+1):][valid] == classes[i])
            overlaped_bboxes = np.concatenate(
                [bboxes[(i+1):][valid][remove_overlap], bboxes[[i]]], axis=0)
            boxes_mean = np.median(overlaped_bboxes, axis=0)
            bboxes[i][:] = boxes_mean[:]
            boxes_corners_mean = boxes_3d_to_corners(
                np.expand_dims(boxes_mean, axis=0))
            boxes_mean_overlap = overlapped_fn(boxes_corners_mean[0],
                boxes_corners[(i+1):][valid][remove_overlap])
            scores[i] += np.sum(
                scores[(i+1):][valid][remove_overlap]*boxes_mean_overlap)
            keep_bboxes[(i+1):][valid] = np.logical_not(remove_overlap)##
    idxes = np.where(keep_bboxes)
    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    if attributes is not None:
        attributes = attributes[idxes]
    return classes, scores, bboxes, attributes

def bboxes_nms_merge_only(classes, scores, bboxes, scores_threshold=0.25,
    nms_threshold=0.45, overlapped_fn=overlapped_boxes_3d, appr_factor=10.0,
    attributes=None):
    """Apply non-maximum selection to bounding boxes.
    """
    boxes_corners = boxes_3d_to_corners(bboxes)
    # convert to pixels
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Only compute on the rest of bboxes
            valid = keep_bboxes[(i+1):]
            # Computer overlap with bboxes which are following.
            overlap = overlapped_fn(boxes_corners[i],
                boxes_corners[(i+1):][valid])
            # Overlap threshold for keeping + checking part of the same class
            remove_overlap = np.logical_and(overlap > nms_threshold,
                classes[(i+1):][valid] == classes[i])
            overlaped_bboxes = np.concatenate(
                [bboxes[(i+1):][valid][remove_overlap], bboxes[[i]]], axis=0)
            boxes_mean = np.median(overlaped_bboxes, axis=0)
            bboxes[i][:] = boxes_mean[:]
            keep_bboxes[(i+1):][valid] = np.logical_not(remove_overlap)##

    idxes = np.where(keep_bboxes)
    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    if attributes is not None:
        attributes = attributes[idxes]
    return classes, scores, bboxes, attributes

def bboxes_nms_score_only(classes, scores, bboxes, scores_threshold=0.25,
    nms_threshold=0.45, overlapped_fn=overlapped_boxes_3d, appr_factor=10.0,
    attributes=None):
    """Apply non-maximum selection to bounding boxes.
    """
    boxes_corners = boxes_3d_to_corners(bboxes)
    # convert to pixels
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Only compute on the rest of bboxes
            valid = keep_bboxes[(i+1):]
            # Computer overlap with bboxes which are following.
            overlap = overlapped_fn(boxes_corners[i],
                boxes_corners[(i+1):][valid])
            # Overlap threshold for keeping + checking part of the same class
            remove_overlap = np.logical_and(overlap > nms_threshold,
                classes[(i+1):][valid] == classes[i])
            overlaped_bboxes = np.concatenate(
                [bboxes[(i+1):][valid][remove_overlap], bboxes[[i]]], axis=0)
            boxes_mean = bboxes[i][:]
            bboxes[i][:] = boxes_mean[:]
            boxes_corners_mean = boxes_3d_to_corners(
                np.expand_dims(boxes_mean, axis=0))
            boxes_mean_overlap = overlapped_fn(boxes_corners_mean[0],
                boxes_corners[(i+1):][valid][remove_overlap])
            scores[i] += np.sum(
                scores[(i+1):][valid][remove_overlap]*boxes_mean_overlap)
            keep_bboxes[(i+1):][valid] = np.logical_not(remove_overlap)##
    idxes = np.where(keep_bboxes)
    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    if attributes is not None:
        attributes = attributes[idxes]
    return classes, scores, bboxes, attributes

def nms_boxes_3d(class_labels, detection_boxes_3d, detection_scores,
    overlapped_thres=0.5, overlapped_fn=overlapped_boxes_3d, appr_factor=10.0,
    top_k=-1, attributes=None):
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_sort(
            class_labels, detection_scores, detection_boxes_3d, top_k=top_k,
            attributes=attributes)
    # nms
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_nms(
            class_labels, detection_scores, detection_boxes_3d,
            nms_threshold=overlapped_thres, overlapped_fn=overlapped_fn,
            appr_factor=appr_factor, attributes=attributes)
    return class_labels, detection_boxes_3d, detection_scores, attributes

def nms_boxes_3d_uncertainty(class_labels, detection_boxes_3d, detection_scores,
    overlapped_thres=0.5, overlapped_fn=overlapped_boxes_3d, appr_factor=10.0,
    top_k=-1, attributes=None):

    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_sort(
            class_labels, detection_scores, detection_boxes_3d, top_k=top_k,
            attributes=attributes)
    # nms
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_nms_uncertainty(
            class_labels, detection_scores, detection_boxes_3d,
            nms_threshold=overlapped_thres, overlapped_fn=overlapped_fn,
            appr_factor=appr_factor, attributes=attributes)
    return class_labels, detection_boxes_3d, detection_scores, attributes

def nms_boxes_3d_merge_only(class_labels, detection_boxes_3d, detection_scores,
    overlapped_thres=0.5, overlapped_fn=overlapped_boxes_3d, appr_factor=10.0,
    top_k=-1, attributes=None):
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_sort(
            class_labels, detection_scores, detection_boxes_3d, top_k=top_k,
            attributes=attributes)
    # nms
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_nms_merge_only(
            class_labels, detection_scores, detection_boxes_3d,
            nms_threshold=overlapped_thres, overlapped_fn=overlapped_fn,
            appr_factor=appr_factor, attributes=attributes)
    return class_labels, detection_boxes_3d, detection_scores, attributes

def nms_boxes_3d_score_only(class_labels, detection_boxes_3d, detection_scores,
    overlapped_thres=0.5, overlapped_fn=overlapped_boxes_3d, appr_factor=10.0,
    top_k=-1, attributes=None):
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_sort(
            class_labels, detection_scores, detection_boxes_3d, top_k=top_k,
            attributes=attributes)
    # nms
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_nms_score_only(
            class_labels, detection_scores, detection_boxes_3d,
            nms_threshold=overlapped_thres, overlapped_fn=overlapped_fn,
            appr_factor=appr_factor, attributes=attributes)
    return class_labels, detection_boxes_3d, detection_scores, attributes
