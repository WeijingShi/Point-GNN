"""This file implements functions to encode and decode 3boxes."""

import numpy as np

def direct_box_encoding(cls_labels, points_xyz, boxes_3d):
    return boxes_3d

def direct_box_decoding(cls_labels, points_xyz, encoded_boxes):
    return encoded_boxes

def center_box_encoding(cls_labels, points_xyz, boxes_3d):
    boxes_3d[:, 0] = boxes_3d[:, 0] - points_xyz[:, 0]
    boxes_3d[:, 1] = boxes_3d[:, 1] - points_xyz[:, 1]
    boxes_3d[:, 2] = boxes_3d[:, 2] - points_xyz[:, 2]
    return boxes_3d

def center_box_decoding(cls_labels, points_xyz, encoded_boxes):
    encoded_boxes[:, 0] = encoded_boxes[:, 0] + points_xyz[:, 0]
    encoded_boxes[:, 1] = encoded_boxes[:, 1] + points_xyz[:, 1]
    encoded_boxes[:, 2] = encoded_boxes[:, 2] + points_xyz[:, 2]
    return encoded_boxes

def voxelnet_box_encoding(cls_labels, points_xyz, boxes_3d):
    # offset
    boxes_3d[:, 0] = boxes_3d[:, 0] - points_xyz[:, 0]
    boxes_3d[:, 1] = boxes_3d[:, 1] - points_xyz[:, 1]
    boxes_3d[:, 2] = boxes_3d[:, 2] - points_xyz[:, 2]
    # Car
    mask = cls_labels[:, 0] == 2
    boxes_3d[mask, 0] = boxes_3d[mask, 0]/3.9
    boxes_3d[mask, 1] = boxes_3d[mask, 1]/1.56
    boxes_3d[mask, 2] = boxes_3d[mask, 2]/1.6
    boxes_3d[mask, 3] = np.log(boxes_3d[mask, 3]/3.9)
    boxes_3d[mask, 4] = np.log(boxes_3d[mask, 4]/1.56)
    boxes_3d[mask, 5] = np.log(boxes_3d[mask, 5]/1.6)
    # Pedestrian and Cyclist
    mask = (cls_labels[:, 0] == 1) + (cls_labels[:, 0] == 3)
    boxes_3d[mask, 0] = boxes_3d[mask, 0]/0.8
    boxes_3d[mask, 1] = boxes_3d[mask, 1]/1.73
    boxes_3d[mask, 2] = boxes_3d[mask, 2]/0.6
    boxes_3d[mask, 3] = np.log(boxes_3d[mask, 3]/0.8)
    boxes_3d[mask, 4] = np.log(boxes_3d[mask, 4]/1.73)
    boxes_3d[mask, 5] = np.log(boxes_3d[mask, 5]/0.6)
    # normalize all yaws
    boxes_3d[:, 6] = boxes_3d[:, 6]/(np.pi*0.5)
    return boxes_3d

def voxelnet_box_decoding(cls_labels, points_xyz, encoded_boxes):
    # Car
    mask = cls_labels[:, 0] == 2
    encoded_boxes[mask, 0] = encoded_boxes[mask, 0]*3.9
    encoded_boxes[mask, 1] = encoded_boxes[mask, 1]*1.56
    encoded_boxes[mask, 2] = encoded_boxes[mask, 2]*1.6
    encoded_boxes[mask, 3] = np.exp(encoded_boxes[mask, 3])*3.9
    encoded_boxes[mask, 4] = np.exp(encoded_boxes[mask, 4])*1.56
    encoded_boxes[mask, 5] = np.exp(encoded_boxes[mask, 5])*1.6
    # Pedestrian and Cyclist
    mask = (cls_labels[:, 0] == 1) + (cls_labels[:, 0] == 3)
    encoded_boxes[mask, 0] = encoded_boxes[mask, 0]*0.8
    encoded_boxes[mask, 1] = encoded_boxes[mask, 1]*1.73
    encoded_boxes[mask, 2] = encoded_boxes[mask, 2]*0.6
    encoded_boxes[mask, 3] = np.exp(encoded_boxes[mask, 3])*0.8
    encoded_boxes[mask, 4] = np.exp(encoded_boxes[mask, 4])*1.73
    encoded_boxes[mask, 5] = np.exp(encoded_boxes[mask, 5])*0.6
    # offset
    encoded_boxes[:, 0] = encoded_boxes[:, 0] + points_xyz[:, 0]
    encoded_boxes[:, 1] = encoded_boxes[:, 1] + points_xyz[:, 1]
    encoded_boxes[:, 2] = encoded_boxes[:, 2] + points_xyz[:, 2]
    # recover all yaws
    encoded_boxes[:, 6] = encoded_boxes[:, 6]*(np.pi*0.5)
    return encoded_boxes

def classaware_voxelnet_box_encoding(cls_labels, points_xyz, boxes_3d):
    """
    Args:
        boxes_3d: [None, num_classes, 7]
    """
    encoded_boxes_3d = np.zeros_like(boxes_3d)
    num_classes = boxes_3d.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    encoded_boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
    encoded_boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
    encoded_boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
    # Car horizontal
    mask = cls_labels[:, 0] == 1
    encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/3.9
    encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.56
    encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/1.6
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/3.9)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.56)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/1.6)
    encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
    # Car vertical
    mask = cls_labels[:, 0] == 2
    encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/3.9
    encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.56
    encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/1.6
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/3.9)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.56)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/1.6)
    encoded_boxes_3d[mask, 0, 6] = (boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)
    # Pedestrian horizontal
    mask = cls_labels[:, 0] == 3
    encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/0.8
    encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.73
    encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/0.6
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/0.8)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.73)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/0.6)
    encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
    # Pedestrian vertical
    mask = cls_labels[:, 0] == 4
    encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/0.8
    encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.73
    encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/0.6
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/0.8)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.73)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/0.6)
    encoded_boxes_3d[mask, 0, 6] = (boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)
    # Cyclist horizontal
    mask = cls_labels[:, 0] == 5
    encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/1.76
    encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.73
    encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/0.6
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/1.76)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.73)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/0.6)
    encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
    # Cyclist vertical
    mask = cls_labels[:, 0] == 6
    encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/1.76
    encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.73
    encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/0.6
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/1.76)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.73)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/0.6)
    encoded_boxes_3d[mask, 0, 6] = (boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)

    return encoded_boxes_3d

def classaware_voxelnet_box_decoding(cls_labels, points_xyz, encoded_boxes):
    decoded_boxes_3d = np.zeros_like(encoded_boxes)
    # Car horizontal
    mask = cls_labels[:, 0] == 1
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*3.9
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.56
    decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*1.6
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*3.9
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.56
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*1.6
    decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
    # Car vertical
    mask = cls_labels[:, 0] == 2
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*3.9
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.56
    decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*1.6
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*3.9
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.56
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*1.6
    decoded_boxes_3d[mask, 0, 6] = (
        encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
    # Pedestrian horizontal
    mask = cls_labels[:, 0] == 3
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*0.8
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.73
    decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*0.6
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*0.8
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.73
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*0.6
    decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
    # Pedestrian vertical
    mask = cls_labels[:, 0] == 4
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*0.8
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.73
    decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*0.6
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*0.8
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.73
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*0.6
    decoded_boxes_3d[mask, 0, 6] = (
        encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
    # Cyclist horizontal
    mask = cls_labels[:, 0] == 5
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*1.76
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.73
    decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*0.6
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*1.76
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.73
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*0.6
    decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
    # Cyclist vertical
    mask = cls_labels[:, 0] == 6
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*1.76
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.73
    decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*0.6
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*1.76
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.73
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*0.6
    decoded_boxes_3d[mask, 0, 6] = (
        encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
    # offset
    num_classes = encoded_boxes.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
    decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
    decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]
    return decoded_boxes_3d

median_object_size_map = {
    'Cyclist': (1.76, 1.75, 0.6),
    'Van': (4.98, 2.13, 1.88),
    'Tram': (14.66, 3.61, 2.6),
    'Car': (3.88, 1.5, 1.63),
    'Misc': (2.52, 1.65, 1.51),
    'Pedestrian': (0.88, 1.77, 0.65),
    'Truck': (10.81, 3.34, 2.63),
    'Person_sitting': (0.75, 1.26, 0.59),
    # 'DontCare': (-1.0, -1.0, -1.0)
}
# 1627 Cyclist mh=1.75; mw=0.6; ml=1.76;
# 2914 Van mh=2.13; mw=1.88; ml=4.98;
# 511 Tram mh=3.61; mw=2.6; ml=14.66;
# 28742 Car mh=1.5; mw=1.63; ml=3.88;
# 973 Misc mh=1.65; mw=1.51; ml=2.52; voxelnet
# 4487 Pedestrian mh=1.77; mw=0.65; ml=0.88;
# 1094 Truck mh=3.34; mw=2.63; ml=10.81;
# 222 Person_sitting mh=1.26; mw=0.59; ml=0.75;
# 11295 DontCare mh=-1.0; mw=-1.0; ml=-1.0;

def classaware_all_class_box_encoding(cls_labels, points_xyz, boxes_3d,
    label_map):
    encoded_boxes_3d = np.copy(boxes_3d)
    num_classes = boxes_3d.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    encoded_boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
    encoded_boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
    encoded_boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]
        mask = cls_labels[:, 0] == cls_label
        encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/l
        encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/h
        encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/w
        encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
        encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
        encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
        encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
        # vertical
        mask = cls_labels[:, 0] == (cls_label+1)
        encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/l
        encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/h
        encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/w
        encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
        encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
        encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
        encoded_boxes_3d[mask, 0, 6] = (
            boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)
    return encoded_boxes_3d

def classaware_all_class_box_decoding(cls_labels, points_xyz, encoded_boxes,
    label_map):
    decoded_boxes_3d = np.copy(encoded_boxes)
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]
        # Car horizontal
        mask = cls_labels[:, 0] == cls_label
        decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l
        decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
        decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*w
        decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
        decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
        decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
        decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
        # Car vertical
        mask = cls_labels[:, 0] == (cls_label+1)
        decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l
        decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
        decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*w
        decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
        decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
        decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
        decoded_boxes_3d[mask, 0, 6] = (
            encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
    # offset
    num_classes = encoded_boxes.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
    decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
    decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]
    return decoded_boxes_3d

def classaware_all_class_box_canonical_encoding(cls_labels, points_xyz,
    boxes_3d, label_map):
    boxes_3d = np.copy(boxes_3d)
    num_classes = boxes_3d.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
    boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
    boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
    encoded_boxes_3d = np.copy(boxes_3d)
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue

        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]
        mask = cls_labels[:, 0] == cls_label
        encoded_boxes_3d[mask, 0, 0] = (
            boxes_3d[mask, 0, 0]*np.cos(boxes_3d[mask, 0, 6]) \
            -boxes_3d[mask, 0, 2]*np.sin(boxes_3d[mask, 0, 6]))/l
        encoded_boxes_3d[mask, 0, 1] = boxes_3d[mask, 0, 1]/h
        encoded_boxes_3d[mask, 0, 2] = (
            boxes_3d[mask, 0, 0]*np.sin(boxes_3d[mask, 0, 6]) \
            +boxes_3d[mask, 0, 2]*np.cos(boxes_3d[mask, 0, 6]))/w
        encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
        encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
        encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
        encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
        # vertical
        mask = cls_labels[:, 0] == (cls_label+1)
        encoded_boxes_3d[mask, 0, 0] = (
            boxes_3d[mask, 0, 0]*np.cos(boxes_3d[mask, 0, 6]-np.pi*0.5) \
            -boxes_3d[mask, 0, 2]*np.sin(boxes_3d[mask, 0, 6]-np.pi*0.5))/w
        encoded_boxes_3d[mask, 0, 1] = boxes_3d[mask, 0, 1]/h
        encoded_boxes_3d[mask, 0, 2] = (
            boxes_3d[mask, 0, 0]*np.sin(boxes_3d[mask, 0, 6]-np.pi*0.5) \
            +boxes_3d[mask, 0, 2]*np.cos(boxes_3d[mask, 0, 6]-np.pi*0.5))/l
        encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
        encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
        encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
        encoded_boxes_3d[mask, 0, 6] = (
            boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)
    return encoded_boxes_3d

def classaware_all_class_box_canonical_decoding(cls_labels, points_xyz,
    encoded_boxes, label_map):
    decoded_boxes_3d = np.copy(encoded_boxes)
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]
        # Car horizontal
        mask = cls_labels[:, 0] == cls_label

        decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l*np.cos(
            encoded_boxes[mask, 0, 6]*(np.pi*0.25))\
            +encoded_boxes[mask, 0, 2]*w*np.sin(
            encoded_boxes[mask, 0, 6]*(np.pi*0.25))
        decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
        decoded_boxes_3d[mask, 0, 2] = -encoded_boxes[mask, 0, 0]*l*np.sin(
            encoded_boxes[mask, 0, 6]*(np.pi*0.25))\
            +encoded_boxes[mask, 0, 2]*w*np.cos(
            encoded_boxes[mask, 0, 6]*(np.pi*0.25))

        decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
        decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
        decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
        decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)

        # Car vertical
        mask = cls_labels[:, 0] == (cls_label+1)
        decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*w*np.cos(
            encoded_boxes[mask, 0, 6]*(np.pi*0.25))\
            +encoded_boxes[mask, 0, 2]*l*np.sin(
            encoded_boxes[mask, 0, 6]*(np.pi*0.25))
        decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
        decoded_boxes_3d[mask, 0, 2] = -encoded_boxes[mask, 0, 0]*w*np.sin(
            encoded_boxes[mask, 0, 6]*(np.pi*0.25))\
            +encoded_boxes[mask, 0, 2]*l*np.cos(
            encoded_boxes[mask, 0, 6]*(np.pi*0.25))

        decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
        decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
        decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
        decoded_boxes_3d[mask, 0, 6] = (
            encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
    # offset
    num_classes = encoded_boxes.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
    decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
    decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]
    return decoded_boxes_3d

def test_encode_decode():
    cls_labels = np.random.choice(5, (1000, 1))
    points_xyz = np.random.random((1000, 3))*10
    boxes_3d = np.random.random((1000, 7))*10
    boxes_3d[:, 3:6] = np.absolute(boxes_3d[:, 3:6])
    encoded_boxes = voxelnet_box_encoding(cls_labels, points_xyz, boxes_3d)
    decoded_boxes = voxelnet_box_decoding(cls_labels, points_xyz, encoded_boxes)
    assert np.isclose(decoded_boxes, boxes_3d).all()

def test_classaware_encode_decode():
    cls_labels = np.random.choice(8, (1000, 1))
    points_xyz = np.random.random((1000, 3))*10
    boxes_3d = np.random.random((1000, 1, 7))*10
    boxes_3d[:, :, 3:6] = np.absolute(boxes_3d[:, :, 3:6])
    encoded_boxes = classaware_voxelnet_box_encoding(
        cls_labels, points_xyz, boxes_3d)
    decoded_boxes = classaware_voxelnet_box_decoding(
        cls_labels, points_xyz, encoded_boxes)
    valid_box_indices = np.nonzero((cls_labels<7)*(cls_labels>0))[0]
    assert np.isclose(
        decoded_boxes[valid_box_indices], boxes_3d[valid_box_indices]).all()

def test_classaware_all_encode_decode():
    num_samples = 10000
    cls_labels = np.random.choice(
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17], (num_samples, 1))
    points_xyz = np.random.random((num_samples, 3))*10
    boxes_3d = np.random.random((num_samples, 1, 7))*10
    boxes_3d[:, :, 3:6] = np.absolute(boxes_3d[:, :, 3:6])
    label_map = {
        'Background': 0,
        'Car': 1,
        'Pedestrian': 3,
        'Cyclist': 5,
        'Van': 7,
        'Truck': 9,
        'Person_sitting': 11,
        'Tram': 13,
        'Misc': 15,
        'DontCare': 17
    }
    encoded_boxes = classaware_all_class_box_encoding(
        cls_labels, points_xyz, boxes_3d, label_map)
    decoded_boxes = classaware_all_class_box_decoding(
        cls_labels, points_xyz, encoded_boxes, label_map)
    assert np.isclose(decoded_boxes, boxes_3d).all()

def test_classaware_all_canonical_encode_decode():
    num_samples = 10000
    cls_labels = np.random.choice(
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17], (num_samples, 1))
    points_xyz = np.random.random((num_samples, 3))*10
    boxes_3d = np.random.random((num_samples, 1, 7))*10
    boxes_3d[:, :, 3:6] = np.absolute(boxes_3d[:, :, 3:6])
    label_map = {
        'Background': 0,
        'Car': 1,
        'Pedestrian': 3,
        'Cyclist': 5,
        'Van': 7,
        'Truck': 9,
        'Person_sitting': 11,
        'Tram': 13,
        'Misc': 15,
        'DontCare': 17
    }
    encoded_boxes = classaware_all_class_box_canonical_encoding(
        cls_labels, points_xyz, boxes_3d, label_map)
    decoded_boxes = classaware_all_class_box_canonical_decoding(
        cls_labels, points_xyz, encoded_boxes, label_map)
    assert np.isclose(decoded_boxes, boxes_3d).all()

def get_box_encoding_fn(encoding_method_name):
    encoding_method_dict = {
        'direct_encoding': direct_box_encoding,
        'center_box_encoding': center_box_encoding,
        'voxelnet_box_encoding': voxelnet_box_encoding,
        'classaware_voxelnet_box_encoding': classaware_voxelnet_box_encoding,
        'classaware_all_class_box_encoding':classaware_all_class_box_encoding,
        'classaware_all_class_box_canonical_encoding':
            classaware_all_class_box_canonical_encoding,
    }
    return encoding_method_dict[encoding_method_name]

def get_box_decoding_fn(encoding_method_name):
    decoding_method_dict = {
        'direct_encoding': direct_box_decoding,
        'center_box_encoding': center_box_decoding,
        'voxelnet_box_encoding': voxelnet_box_decoding,
        'classaware_voxelnet_box_encoding': classaware_voxelnet_box_decoding,
        'classaware_all_class_box_encoding': classaware_all_class_box_decoding,
        'classaware_all_class_box_canonical_encoding':
            classaware_all_class_box_canonical_decoding,
    }
    return decoding_method_dict[encoding_method_name]

def get_encoding_len(encoding_method_name):
    encoding_len_dict = {
        'direct_encoding': 7,
        'center_box_encoding': 7,
        'voxelnet_box_encoding':7,
        'classaware_voxelnet_box_encoding': 7,
        'classaware_all_class_box_encoding': 7,
        'classaware_all_class_box_canonical_encoding': 7,
    }
    return encoding_len_dict[encoding_method_name]

if __name__ == '__main__':
    test_encode_decode()
    test_classaware_encode_decode()
    test_classaware_all_encode_decode()
    test_classaware_all_canonical_encode_decode()
