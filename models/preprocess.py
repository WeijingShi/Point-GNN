"""This file defines functions to augment data from dataset. """

import numpy as np
import random
from copy import deepcopy

from dataset.kitti_dataset import Points, sel_xyz_in_box3d, \
    downsample_by_average_voxel, downsample_by_random_voxel
from models.nms import boxes_3d_to_corners, overlapped_boxes_3d

def random_jitter(cam_rgb_points, labels, xyz_std=(0.1, 0.1, 0.1)):
    xyz = cam_rgb_points.xyz
    x_delta = np.random.normal(size=(xyz.shape[0], 1), scale=xyz_std[0])
    y_delta = np.random.normal(size=(xyz.shape[0], 1), scale=xyz_std[1])
    z_delta = np.random.normal(size=(xyz.shape[0], 1), scale=xyz_std[2])
    xyz += np.hstack([x_delta, y_delta, z_delta])
    return Points(xyz=xyz, attr=cam_rgb_points.attr), labels

def random_drop(cam_rgb_points, labels, drop_prob=0.5, tier_prob=None):
    if isinstance(drop_prob, list):
        drop_prob = np.random.choice(drop_prob, p=tier_prob)
    xyz = cam_rgb_points.xyz
    mask = np.random.uniform(size=xyz.shape[0]) > drop_prob
    if np.sum(mask) == 0:
        # print("Warning: attempt to drop all points, restore to all points")
        mask = np.ones_like(mask)
    return Points(xyz=xyz[mask], attr=cam_rgb_points.attr[mask]), labels

def random_global_drop(cam_rgb_points, labels, drop_std=0.25):
    drop_prob = np.abs(np.random.normal(scale=drop_std))
    # print("drop %f "%(drop_prob))
    return random_drop(cam_rgb_points, labels, drop_prob=drop_prob)

def random_voxel_downsample(cam_rgb_points, labels, voxel_std=0.2,
    min_voxel=0.02, max_voxel=0.8):
    voxel_size = np.abs(np.random.normal(scale=voxel_std))
    voxel_size = np.minimum(voxel_size, max_voxel)
    if voxel_size < min_voxel:
        return cam_rgb_points, labels
    downsampled_points = downsample_by_random_voxel(cam_rgb_points,
        voxel_size, add_rnd3d=True)
    return downsampled_points, labels

def random_rotation_all(cam_rgb_points, labels, method_name='normal',
    yaw_std=0.3, expend_factor=(1.0, 1.1, 1.1)):
    xyz = cam_rgb_points.xyz
    if method_name == 'normal':
        delta_yaw = np.random.normal(scale=yaw_std)
    else:
        if method_name == 'uniform':
            delta_yaw = np.random.uniform(low=-yaw_std, high=yaw_std)
    R = np.array([[np.cos(delta_yaw),  0,  np.sin(delta_yaw)],
                  [0,            1,  0          ],
                  [-np.sin(delta_yaw), 0,  np.cos(delta_yaw)]]);
    xyz = xyz.dot(np.transpose(R))
    # print('rotate globally'+str(delta_yaw))
    for label in labels:
        if label['name'] != 'DontCare':
            tx = label['x3d']
            ty = label['y3d']
            tz = label['z3d']
            xyz_center = np.array([[tx, ty, tz]])
            xyz_center = xyz_center.dot(np.transpose(R))
            label['x3d'], label['y3d'], label['z3d'] = xyz_center[0]
            label['yaw'] = label['yaw']+delta_yaw
    return Points(xyz=xyz, attr=cam_rgb_points.attr), labels

def random_flip_all(cam_rgb_points, labels, flip_prob=0.5):
    xyz = cam_rgb_points.xyz
    p =  np.random.uniform()
    if p < flip_prob:
        xyz[:,0] = -xyz[:,0]
        for label in labels:
            if label['name'] != 'DontCare':
                label['x3d'] = -label['x3d']
                label['yaw'] = np.pi-label['yaw']
    return Points(xyz=xyz, attr=cam_rgb_points.attr), labels

def random_scale_all(cam_rgb_points, labels, method_name='normal',
    scale_std=0.05):
    xyz = cam_rgb_points.xyz
    if method_name == 'normal':
        scale = np.random.normal(scale=scale_std) + 1.0
    else:
        if method_name == 'uniform':
            scale = np.random.uniform(low=-scale_std, high=scale_std) + 1
    xyz *= scale
    for label in labels:
        if label['name'] != 'DontCare':
            label['x3d'] *= scale
            label['y3d'] *= scale
            label['z3d'] *= scale
            label['length'] *= scale
            label['width'] *= scale
            label['height'] *= scale
    return Points(xyz=xyz, attr=cam_rgb_points.attr), labels

def random_box_rotation(cam_rgb_points, labels, max_overlap_num_allowed=0.1,
    max_trails = 100, appr_factor=100, method_name='normal',
    yaw_std=0.3, expend_factor=(1.0, 1.1, 1.1),
    augment_list=[
    'Car',
    'Pedestrian',
    'Cyclist',
    'Van',
    'Truck',
    'Misc',
    'Tram',
    'Person_sitting',
    ]
    ):
    xyz = cam_rgb_points.xyz
    # filtering DontCare
    labels_no_dontcare = [label for label in labels
        if label['name'] != 'DontCare']
    # check existing overlap
    new_labels = []
    for i, label in enumerate(labels_no_dontcare):
        if label['name'] in augment_list:
            trial = 0
            sucess = False
            for trial in range(max_trails):
                # random rotate
                if method_name == 'normal':
                    delta_yaw = np.random.normal(scale=yaw_std)
                else:
                    if method_name == 'uniform':
                        delta_yaw = np.random.uniform(low=-yaw_std,
                            high=yaw_std)
                new_label = deepcopy(label)
                new_label['yaw'] = new_label['yaw']+delta_yaw
                # check if the new box includes more points than before
                mask = sel_xyz_in_box3d(label, xyz, expend_factor)
                more_mask = sel_xyz_in_box3d(new_label,
                    xyz[np.logical_not(mask)], expend_factor)
                if np.sum(more_mask) < max_overlap_num_allowed:
                    # valid new box, start rotation
                    mask = sel_xyz_in_box3d(label, xyz, expend_factor)
                    points_xyz = xyz[mask, :]
                    tx = label['x3d']
                    ty = label['y3d']
                    tz = label['z3d']
                    points_xyz -= np.array([tx, ty, tz])
                    R = np.array([[np.cos(delta_yaw),  0,  np.sin(delta_yaw)],
                                  [0,            1,  0          ],
                                  [-np.sin(delta_yaw), 0,  np.cos(delta_yaw)]]);
                    points_xyz = points_xyz.dot(np.transpose(R))
                    points_xyz = points_xyz+np.array([tx, ty, tz])
                    xyz[mask, :] = points_xyz
                    # update boxes and label
                    new_labels.append(new_label)
                    sucess = True
                    break;
            if not sucess:
                # if not sucess, keep the old label
                # print('Warning: fail to augment by rotation')
                new_labels.append(label)
        else:
            new_labels.append(label)

    assert len(new_labels) == len(labels_no_dontcare)
    new_labels.extend([l for l in labels if l['name'] == 'DontCare'])
    assert len(new_labels) == len(labels)
    return Points(xyz=xyz, attr=cam_rgb_points.attr), new_labels


def random_box_global_rotation(cam_rgb_points, labels,
    max_overlap_num_allowed=0.1, max_trails = 100, appr_factor=100,
    method_name='normal', yaw_std=0.3, expend_factor=(1.1, 1.1, 1.1),
    augment_list=[
    'Car',
    'Pedestrian',
    'Cyclist',
    'Van',
    'Truck',
    'Misc',
    'Tram',
    'Person_sitting',
    ]
    ):
    xyz = cam_rgb_points.xyz
    attr = cam_rgb_points.attr
    # filtering DontCare
    labels_no_dontcare = [label for label in labels
        if label['name'] != 'DontCare']
    # check existing overlap
    new_labels = []
    for i, label in enumerate(labels_no_dontcare):
        if label['name'] in augment_list:
            trial = 0
            sucess = False
            for trial in range(max_trails):
                # random rotate
                if method_name == 'normal':
                    delta_yaw = np.random.normal(scale=yaw_std)
                else:
                    if method_name == 'uniform':
                        delta_yaw = np.random.uniform(
                            low=-yaw_std, high=yaw_std)
                new_label = deepcopy(label)
                new_label['yaw'] = new_label['yaw']+delta_yaw
                tx = new_label['x3d']
                ty = new_label['y3d']
                tz = new_label['z3d']
                R = np.array([[np.cos(delta_yaw),  0,  np.sin(delta_yaw)],
                              [0,            1,  0          ],
                              [-np.sin(delta_yaw), 0,  np.cos(delta_yaw)]]);
                new_label['x3d'],new_label['y3d'],new_label['z3d'] = \
                    np.array([tx, ty, tz]).dot(np.transpose(R))
                # check if the new box includes more points than before
                mask = sel_xyz_in_box3d(label, xyz, expend_factor)
                new_mask = sel_xyz_in_box3d(new_label, xyz, expend_factor)
                more_mask = np.logical_and(new_mask, np.logical_not(mask))
                if np.sum(more_mask) < max_overlap_num_allowed:
                    # valid new box, start rotation
                    points_xyz = xyz[mask, :]
                    points_xyz = points_xyz.dot(np.transpose(R))
                    # points_xyz = points_xyz+np.array([tx, ty, tz])
                    xyz[mask, :] = points_xyz
                    xyz = xyz[np.logical_not(more_mask)]
                    attr = attr[np.logical_not(more_mask)]
                    # update boxes and label
                    new_labels.append(new_label)
                    sucess = True
                    break;
            if not sucess:
                # if not sucess, keep the old label
                # print('Warning: fail to augment by rotation')
                new_labels.append(label)
        else:
            new_labels.append(label)

    assert len(new_labels) == len(labels_no_dontcare)
    new_labels.extend([l for l in labels if l['name'] == 'DontCare'])
    assert len(new_labels) == len(labels)
    return Points(xyz=xyz, attr=attr), new_labels


def random_box_shift(cam_rgb_points, labels, max_overlap_num_allowed=0.1,
    max_overlap_rate=None, max_trails = 100, appr_factor=100,
    method_name='normal', xyz_std=(1,0,1), expend_factor=(1.0, 1.1, 1.1),
    augment_list=[
    'Car',
    'Pedestrian',
    'Cyclist',
    'Van',
    'Truck',
    'Misc',
    'Tram',
    'Person_sitting',
    ],
    shuffle=False):
    xyz = cam_rgb_points.xyz
    # filtering DontCare
    labels_no_dontcare = [label for label in labels
        if label['name'] != 'DontCare']
    if shuffle:
        random.shuffle(labels_no_dontcare)
    # check existing overlap
    new_labels = []
    label_boxes_corners = None
    for i, label in enumerate(labels_no_dontcare):
        if label['name'] in augment_list:
            trial = 0
            sucess = False
            for trial in range(max_trails):
                # random rotate
                if method_name == 'normal':
                    delta_x, delta_y, delta_z = np.random.normal(scale=xyz_std)
                else:
                    if method_name == 'uniform':
                        delta_x, delta_y, delta_z = np.random.uniform(
                            low=-xyz_std, high=xyz_std)
                new_label = deepcopy(label)
                new_label['x3d'] = new_label['x3d']+delta_x
                new_label['y3d'] = new_label['y3d']+delta_y
                new_label['z3d'] = new_label['z3d']+delta_z
                # check if the new box includes more points than before
                below_overlap = True
                mask = sel_xyz_in_box3d(label, xyz, expend_factor)
                more_mask = sel_xyz_in_box3d(new_label,
                    xyz[np.logical_not(mask)], expend_factor)
                below_overlap *= np.sum(more_mask) < max_overlap_num_allowed
                if max_overlap_rate is not None:
                    new_boxes = np.array([
                        [new_label['x3d'],
                         new_label['y3d'],
                         new_label['z3d'],
                         new_label['length'],
                         new_label['height'],
                         new_label['width'],
                         new_label['yaw']]
                         ])
                    new_boxes_corners = np.int32(
                        appr_factor*boxes_3d_to_corners(new_boxes))
                    label_boxes = np.array([
                        [l['x3d'], l['y3d'], l['z3d'],
                        l['length'], l['height'], l['width'], l['yaw']]
                            for l in new_labels])
                    label_boxes_corners = np.int32(
                        appr_factor*boxes_3d_to_corners(label_boxes))
                    below_overlap_rate = np.all(overlapped_boxes_3d(
                        new_boxes_corners[0],
                        label_boxes_corners) < max_overlap_rate)
                    below_overlap *= below_overlap_rate
                if below_overlap:
                    # valid new box, start rotation
                    mask = sel_xyz_in_box3d(label, xyz, expend_factor)
                    points_xyz = xyz[mask, :]
                    points_xyz = points_xyz+np.array(
                        [delta_x, delta_y, delta_z])
                    xyz[mask, :] = points_xyz
                    # update boxes and label
                    new_labels.append(new_label)
                    sucess = True
                    break;
            if not sucess:
                # if not sucess, keep the old label
                # print('Warning: fail to augment by shifting')
                new_labels.append(label)
        else:
            new_labels.append(label)
    assert len(new_labels) == len(labels_no_dontcare)
    new_labels.extend([l for l in labels if l['name'] == 'DontCare'])
    assert len(new_labels) == len(labels)
    return Points(xyz=xyz, attr=cam_rgb_points.attr), new_labels

def dilute_background(cam_rgb_points, labels, dilute_voxel_base=0.4,
    expend_factor=(4.0, 4.0, 4.0),
    keep_list=[
    # 'Background',
    'Car',
    'Pedestrian',
    'Cyclist',
    'Van',
    'Truck',
    'Misc',
    # 'Tram',
    'Person_sitting',
    # 'DontCare'
    ],
    ):
    xyz = cam_rgb_points.xyz
    mask = np.zeros(xyz.shape[0], dtype=np.bool)

    labels_no_dontcare = []
    for label in labels:
        if label['name'] in keep_list:
            labels_no_dontcare.append(label)

    # if no object then keep some objects
    if len(labels_no_dontcare) < 1:
        for label in labels:
            if label['name'] != 'DontCare':
                labels_no_dontcare.append(label)


    selected_labels = deepcopy(labels_no_dontcare)
    for label in selected_labels:
        mask += sel_xyz_in_box3d(label, xyz, expend_factor)

    #assert mask.any()
    if not mask.any():
        # keep two point
        mask[0] = True

    background_xyz = xyz[np.logical_not(mask)]
    background_attr = cam_rgb_points.attr[np.logical_not(mask)]
    background_points = Points(xyz=background_xyz, attr=background_attr)
    front_xyz = xyz[mask]
    front_attr = cam_rgb_points.attr[mask]
    diluted_background_points = downsample_by_random_voxel(
        background_points, dilute_voxel_base, add_rnd3d=True)

    return Points(
        xyz=np.concatenate([front_xyz, diluted_background_points.xyz], axis=0),
        attr=np.concatenate([front_attr,
            diluted_background_points.attr], axis=0)), labels_no_dontcare

def remove_background(cam_rgb_points, labels, expend_factor=(4.0, 4.0, 4.0),
    keep_list=[
    # 'Background',
    'Car',
    'Pedestrian',
    'Cyclist',
    'Van',
    'Truck',
    'Misc',
    # 'Tram',
    'Person_sitting',
    # 'DontCare'
    ],
    num_object=-1,
    mask_random_rotation_std = 0,
    mask_random_jitter_stds = (0., 0., 0., 0., 0., 0.)
    ):
    xyz = cam_rgb_points.xyz
    mask = np.zeros(xyz.shape[0], dtype=np.bool)

    labels_no_dontcare = []
    for label in labels:
        if label['name'] in keep_list:
            labels_no_dontcare.append(label)

    # if no object then keep some objects
    if len(labels_no_dontcare) < 1:
        for label in labels:
            if label['name'] != 'DontCare':
                labels_no_dontcare.append(label)

    selected_labels = []
    if num_object > 0:
        sample_idx = np.random.choice(len(labels_no_dontcare), num_object)
        for i in sample_idx:
            selected_labels.append(labels_no_dontcare[i])
    else:
        selected_labels = labels_no_dontcare

    selected_labels = deepcopy(selected_labels)
    for label in selected_labels:
        mask += sel_xyz_in_box3d(label, xyz, expend_factor)

    #assert mask.any()
    if not mask.any():
        # keep two point
        mask[0] = True
    return Points(xyz=xyz[mask],
        attr=cam_rgb_points.attr[mask]), labels_no_dontcare

def random_transition(cam_rgb_points, labels, xyz_std=(0.1, 0.1, 0.1)):
    xyz = cam_rgb_points.xyz
    x_delta = np.random.normal(scale=xyz_std[0])
    y_delta = np.random.normal(scale=xyz_std[1])
    z_delta = np.random.normal(scale=xyz_std[2])
    xyz += np.hstack([x_delta, y_delta, z_delta])
    for label in labels:
        label['x3d'] += x_delta
        label['y3d'] += y_delta
        label['z3d'] += z_delta
    return Points(xyz=xyz, attr=cam_rgb_points.attr), labels


def empty(cam_rgb_points, labels):
    return cam_rgb_points, labels

aug_method_map = {
    'random_jitter': random_jitter,
    'random_box_rotation': random_box_rotation,
    'random_box_shift': random_box_shift,
    'random_transition': random_transition,
    'remove_background': remove_background,
    'random_rotation_all': random_rotation_all,
    'random_flip_all': random_flip_all,
    'random_drop': random_drop,
    'random_global_drop':random_global_drop,
    'random_voxel_downsample': random_voxel_downsample,
    'random_scale_all': random_scale_all,
    'random_box_global_rotation': random_box_global_rotation,
    'dilute_background':dilute_background,
}
def get_data_aug(aug_configs=[]):
    if len(aug_configs)==0:
        return empty
    def multiple_aug(cam_rgb_points, labels):
        for aug_config in aug_configs:
            aug_method = aug_method_map[aug_config['method_name']]
            cam_rgb_points, labels = aug_method(
                cam_rgb_points, labels, **aug_config['method_kwargs'])
        return cam_rgb_points, labels
    return multiple_aug
