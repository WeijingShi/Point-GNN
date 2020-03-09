"""This file implement augmentation by cropping and parsing ground truth boxes"""

import os
import json

import numpy as np
import open3d
from copy import deepcopy
from tqdm import tqdm

from dataset.kitti_dataset import KittiDataset, sel_xyz_in_box3d, \
    sel_xyz_in_box2d, Points
from models.nms import boxes_3d_to_corners, overlapped_boxes_3d, \
    overlapped_boxes_3d_fast_poly
from models import preprocess

def save_cropped_boxes(dataset, filename, expand_factor=[1.1, 1.1, 1.1],
    minimum_points=10, backlist=[]):
    cropped_labels = {}
    cropped_cam_points = {}
    for frame_idx in tqdm(range(dataset.num_files)):
        labels = dataset.get_label(frame_idx)
        cam_points = dataset.get_cam_points_in_image_with_rgb(frame_idx)
        for label in labels:
            if label['name'] != "DontCare":
                if label['name'] not in backlist:
                    mask = sel_xyz_in_box3d(label, cam_points.xyz,
                        expand_factor)
                    if np.sum(mask) > minimum_points:
                        if label['name'] in cropped_labels:
                            cropped_labels[label['name']].append(label)
                            cropped_cam_points[label['name']].append(
                                [cam_points.xyz[mask].tolist(),
                                cam_points.attr[mask].tolist()])
                        else:
                            cropped_labels[label['name']] = [label]
                            cropped_cam_points[label['name']] = [
                                [cam_points.xyz[mask].tolist(),
                                cam_points.attr[mask].tolist()]]

    with open(filename, 'w') as outfile:
        json.dump((cropped_labels,cropped_cam_points), outfile)

def load_cropped_boxes(filename):
    with open(filename, 'r') as infile:
        cropped_labels, cropped_cam_points = json.load(infile)
    for key in cropped_cam_points:
        print("Load %d %s" % (len(cropped_cam_points[key]), key))
        for i, cam_points in enumerate(cropped_cam_points[key]):
            cropped_cam_points[key][i] = Points(xyz=np.array(cam_points[0]),
                                            attr=np.array(cam_points[1]))
    return cropped_labels, cropped_cam_points

def vis_cropped_boxes(cropped_labels, cropped_cam_points, dataset):
    for key in cropped_cam_points:
        if key == 'Pedestrian':
            for i, cam_points in enumerate(cropped_cam_points[key]):
                label = cropped_labels[key][i]
                print(label['name'])
                pcd = open3d.PointCloud()
                pcd.points = open3d.Vector3dVector(cam_points.xyz)
                pcd.colors = open3d.Vector3dVector(cam_points.attr[:, 1:])
                def custom_draw_geometry_load_option(geometry_list):
                    vis = open3d.Visualizer()
                    vis.create_window()
                    for geometry in geometry_list:
                        vis.add_geometry(geometry)
                    ctr = vis.get_view_control()
                    ctr.rotate(0.0, 3141.0, 0)
                    vis.run()
                    vis.destroy_window()
                custom_draw_geometry_load_option(
                    [pcd] + dataset.draw_open3D_box(label))

def parser_without_collision(cam_rgb_points, labels,
    sample_cam_points, sample_labels,
    overlap_mode = 'box',
    auto_box_height = False,
    max_overlap_rate = 0.01,
    appr_factor = 100,
    max_overlap_num_allowed=1, max_trails=1, method_name='normal',
    yaw_std=0.3, expand_factor=(1.1, 1.1, 1.1),
    must_have_ground=False):
    xyz = cam_rgb_points.xyz
    attr = cam_rgb_points.attr
    if overlap_mode == 'box' or overlap_mode == 'box_and_point':
        label_boxes = np.array([
            [l['x3d'], l['y3d'], l['z3d'], l['length'],
            l['height'], l['width'], l['yaw']]
                for l in labels ])
        label_boxes_corners = np.int32(
            appr_factor*boxes_3d_to_corners(label_boxes))
    for i, label in enumerate(sample_labels):
        trial = 0
        sucess = False
        for trial in range(max_trails):
            # random rotate
            if method_name == 'normal':
                delta_yaw = np.random.normal(scale=yaw_std)
            else:
                if method_name == 'uniform':
                    delta_yaw = np.random.uniform(low=-yaw_std, high=yaw_std)
            new_label = deepcopy(label)
            R = np.array([[np.cos(delta_yaw),  0,  np.sin(delta_yaw)],
                          [0,            1,  0          ],
                          [-np.sin(delta_yaw), 0,  np.cos(delta_yaw)]]);
            tx = new_label['x3d']
            ty = new_label['y3d']
            tz = new_label['z3d']
            xyz_center = np.array([[tx, ty, tz]])
            xyz_center = xyz_center.dot(np.transpose(R))
            new_label['x3d'], new_label['y3d'], new_label['z3d'] = xyz_center[0]
            new_label['yaw'] = new_label['yaw']+delta_yaw
            if auto_box_height:
                original_height = new_label['height']
                mask_2d = sel_xyz_in_box2d(new_label, xyz, expand_factor)
                if np.sum(mask_2d) > 0:
                    ground_height = np.amax(xyz[mask_2d][:,1])
                    y3d_adjust = ground_height - new_label['y3d']
                else:
                    if must_have_ground:
                        continue;
                    y3d_adjust = 0
                # if np.abs(y3d_adjust) > 1:
                #     y3d_adjust = 0
                new_label['y3d'] += y3d_adjust
                new_label['height'] = original_height
            mask = sel_xyz_in_box3d(new_label, xyz, expand_factor)
            # check if the new box includes more points than before
            below_overlap = False
            if overlap_mode == 'box':
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
                below_overlap = np.all(overlapped_boxes_3d_fast_poly(
                    new_boxes_corners[0],
                    label_boxes_corners) < max_overlap_rate)
            if overlap_mode == 'point':
                below_overlap = np.sum(mask) < max_overlap_num_allowed
            if overlap_mode == 'box_and_point':
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
                below_overlap = np.all(
                    overlapped_boxes_3d_fast_poly(new_boxes_corners[0],
                    label_boxes_corners) < max_overlap_rate)
                below_overlap = np.logical_and(below_overlap,
                    (np.sum(mask) < max_overlap_num_allowed))
            if below_overlap:

                points_xyz = sample_cam_points[i].xyz
                points_attr = sample_cam_points[i].attr
                points_xyz = points_xyz.dot(np.transpose(R))
                if auto_box_height:
                    points_xyz[:,1] = points_xyz[:,1] + y3d_adjust
                xyz = xyz[np.logical_not(mask)]
                xyz =  np.concatenate([points_xyz, xyz], axis=0)
                attr = attr[np.logical_not(mask)]
                attr =  np.concatenate([points_attr, attr], axis=0)
                # update boxes and label
                labels.append(new_label)
                if overlap_mode == 'box' or overlap_mode == 'box_and_point':
                    label_boxes_corners = np.append(label_boxes_corners,
                        new_boxes_corners,axis=0)
                sucess = True
                break;
        # if not sucess:
            # if not sucess, keep the old label
            # print('Warning: fail to parse cropped box')
    return Points(xyz=xyz, attr=attr), labels

class CropAugSampler():
    """ A class to sample from cropped objects and parse it to a frame """
    def __init__(self, crop_filename):
        self._cropped_labels, self._cropped_cam_points = load_cropped_boxes(\
            crop_filename)
    def crop_aug(self, cam_rgb_points, labels,
        sample_rate={"Car":1, "Pedestrian":1, "Cyclist":1},
        parser_kwargs={}):
        sample_labels = []
        sample_cam_points = []
        for key in sample_rate:
            sample_indices = np.random.choice(len(self._cropped_labels[key]),
                size=sample_rate[key], replace=False)
            sample_labels.extend(
                deepcopy([self._cropped_labels[key][idx]
                    for idx in sample_indices]))
            sample_cam_points.extend(
                deepcopy([self._cropped_cam_points[key][idx]
                    for idx in sample_indices]))
        return parser_without_collision(cam_rgb_points, labels,
            sample_cam_points, sample_labels,
            **parser_kwargs)

def vis_crop_aug_sampler(crop_filename, dataset):
    sampler = CropAugSampler(crop_filename)
    for frame_idx in range(10):
        labels = dataset.get_label(frame_idx)
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx)
        cam_rgb_points, labels = sampler.crop_aug(cam_rgb_points, labels,
            sample_rate={"Car":2, "Pedestrian":10, "Cyclist":10},
            parser_kwargs={
                'max_overlap_num_allowed': 50,
                'max_trails':100,
                'method_name':'normal',
                'yaw_std':np.pi/16,
                'expand_factor':(1.1, 1.1, 1.1),
                'auto_box_height': True,
                'overlap_mode':'box_and_point',
                'max_overlap_rate': 1e-6,
                'appr_factor': 100,
                'must_have_ground': True,
                })
        aug_configs = [
            {'method_name': 'random_box_global_rotation',
             'method_kwargs': { 'max_overlap_num_allowed':100,
                                'max_trails': 100,
                                'appr_factor':100,
                                'method_name':'normal',
                                'yaw_std':np.pi/8,
                                'expend_factor':(1.1, 1.1, 1.1)
                                }
            }
        ]
        aug_fn = preprocess.get_data_aug(aug_configs)
        cam_rgb_points, labels = aug_fn(cam_rgb_points, labels)
        dataset.vis_points(cam_rgb_points, labels, expend_factor=(1.1, 1.1,1.1))


    # # Example of usage
    # print('generate training split: ')
    # kitti_train = KittiDataset(
    #     '../dataset/kitti/image/training/image_2',
    #     '../dataset/kitti/velodyne/training/velodyne/',
    #     '../dataset/kitti/calib/training/calib/',
    #     '../dataset/kitti/labels/training/label_2/',
    #     '../dataset/kitti/3DOP_splits/train.txt',)
    # save_cropped_boxes(kitti_train, "../dataset/kitti/cropped/car_person_cyclist_train.json",
    #     expand_factor = (1.1, 1.1, 1.1), minimum_points=10,
    #     backlist=['Van', 'Truck', 'Misc', 'Tram', 'Person_sitting'])
    # print("generate val split: ")
    # kitti_val = KittiDataset(
    #     '../dataset/kitti/image/training/image_2',
    #     '../dataset/kitti/velodyne/training/velodyne/',
    #     '../dataset/kitti/calib/training/calib/',
    #     '../dataset/kitti/labels/training/label_2/',
    #     '../dataset/kitti/3DOP_splits/val.txt',)
    # save_cropped_boxes(kitti_val, "../dataset/kitti/cropped/car_person_cyclist_val.json",
    #     expand_factor = (1.1, 1.1, 1.1), minimum_points=10,
    #     backlist=['Van', 'Truck', 'Misc', 'Tram', 'Person_sitting'])
    # print("generate trainval: ")
    # kitti_trainval = KittiDataset(
    #     '../dataset/kitti/image/training/image_2',
    #     '../dataset/kitti/velodyne/training/velodyne/',
    #     '../dataset/kitti/calib/training/calib/',
    #     '../dataset/kitti/labels/training/label_2/',
    #     '../dataset/kitti/3DOP_splits/trainval.txt',)
    # save_cropped_boxes(kitti_trainval, "../dataset/kitti/cropped/car_person_cyclist_trainval.json",
    #     expand_factor = (1.1, 1.1, 1.1), minimum_points=10,
    #     backlist=['Van', 'Truck', 'Misc', 'Tram', 'Person_sitting'])
    # cropped_labels, cropped_cam_points = load_cropped_boxes(
    #     "../dataset/kitti/cropped/car_person_cyclist_train.json")
    # vis_cropped_boxes(cropped_labels, cropped_cam_points, kitti_train)
    # cropped_labels, cropped_cam_points = load_cropped_boxes(
    #     "../dataset/kitti/cropped/car_person_cyclist_val.json")
    # vis_cropped_boxes(cropped_labels, cropped_cam_points, kitti_val)
    # cropped_labels, cropped_cam_points = load_cropped_boxes(
    #     "../dataset/kitti/cropped/car_person_cyclist_trainval.json")
    # vis_cropped_boxes(cropped_labels, cropped_cam_points, kitti_trainval)
    # vis_crop_aug_sampler("../dataset/kitti/cropped/car_person_cyclist_val.json", kitti_val)
