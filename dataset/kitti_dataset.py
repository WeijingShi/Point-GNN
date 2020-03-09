"""This file defines a class to interact with KITTI dataset. """


import os
import time
from os.path import isfile, join
import random
from collections import namedtuple, defaultdict

import numpy as np
import open3d
import cv2

Points = namedtuple('Points', ['xyz', 'attr'])

def downsample_by_average_voxel(points, voxel_size):
    """Voxel downsampling using average function.

    points: a Points namedtuple containing "xyz" and "attr".
    voxel_size: the size of voxel cells used for downsampling.
    """
    # create voxel grid
    xmax, ymax, zmax = np.amax(points.xyz, axis=0)
    xmin, ymin, zmin = np.amin(points.xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    xyz_idx = (points.xyz - xyz_offset) // voxel_size
    xyz_idx = xyz_idx.astype(np.int32)
    dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
    keys = xyz_idx[:, 0] + xyz_idx[:, 1]*dim_x + xyz_idx[:, 2]*dim_y*dim_x
    order = np.argsort(keys)
    keys = keys[order]
    points_xyz = points.xyz[order]
    unique_keys, lens = np.unique(keys, return_counts=True)
    indices = np.hstack([[0], lens[:-1]]).cumsum()
    downsampled_xyz = np.add.reduceat(
        points_xyz, indices, axis=0)/lens[:,np.newaxis]
    include_attr = points.attr is not None
    if include_attr:
        attr = points.attr[order]
        downsampled_attr = np.add.reduceat(
            attr, indices, axis=0)/lens[:,np.newaxis]
    if include_attr:
        return Points(xyz=downsampled_xyz,
                attr=downsampled_attr)
    else:
        return Points(xyz=downsampled_xyz,
                attr=None)

def downsample_by_random_voxel(points, voxel_size, add_rnd3d=False):
    """Downsample the points using base_voxel_size at different scales"""
    xmax, ymax, zmax = np.amax(points.xyz, axis=0)
    xmin, ymin, zmin = np.amin(points.xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)

    if not add_rnd3d:
        xyz_idx = (points.xyz - xyz_offset) // voxel_size
    else:
        xyz_idx = (points.xyz - xyz_offset +
            voxel_size*np.random.random((1,3))) // voxel_size
    dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
    keys = xyz_idx[:, 0] + xyz_idx[:, 1]*dim_x + xyz_idx[:, 2]*dim_y*dim_x
    num_points = xyz_idx.shape[0]

    voxels_idx = {}
    for pidx in range(len(points.xyz)):
        key = keys[pidx]
        if key in voxels_idx:
            voxels_idx[key].append(pidx)
        else:
            voxels_idx[key] = [pidx]

    downsampled_xyz = []
    downsampled_attr = []
    for key in voxels_idx:
        center_idx = random.choice(voxels_idx[key])
        downsampled_xyz.append(points.xyz[center_idx])
        downsampled_attr.append(points.attr[center_idx])

    return Points(xyz=np.array(downsampled_xyz),
        attr=np.array(downsampled_attr))


def box3d_to_cam_points(label, expend_factor=(1.0, 1.0, 1.0)):
    """Project 3D box into camera coordinates.
    Args:
        label: a dictionary containing "x3d", "y3d", "z3d", "yaw", "height"
            "width", "length".

    Returns: a numpy array [8, 3] representing the corners of the 3d box in
        camera coordinates.
    """

    yaw = label['yaw']
    R = np.array([[np.cos(yaw),  0,  np.sin(yaw)],
                  [0,            1,  0          ],
                  [-np.sin(yaw), 0,  np.cos(yaw)]]);
    h = label['height']
    delta_h = h*(expend_factor[0]-1)
    w = label['width']*expend_factor[1]
    l = label['length']*expend_factor[2]
    corners = np.array([[ l/2,  delta_h/2,  w/2],  # front up right
                        [ l/2,  delta_h/2, -w/2],  # front up left
                        [-l/2,  delta_h/2, -w/2],  # back up left
                        [-l/2,  delta_h/2,  w/2],  # back up right
                        [ l/2, -h-delta_h/2,  w/2],  # front down right
                        [ l/2, -h-delta_h/2, -w/2],  # front down left
                        [-l/2, -h-delta_h/2, -w/2],  # back down left
                        [-l/2, -h-delta_h/2,  w/2]]) # back down right
    r_corners = corners.dot(np.transpose(R))
    tx = label['x3d']
    ty = label['y3d']
    tz = label['z3d']
    cam_points_xyz = r_corners+np.array([tx, ty, tz])
    return Points(xyz = cam_points_xyz, attr = None)

def box3d_to_normals(label, expend_factor=(1.0, 1.0, 1.0)):
    """Project a 3D box into camera coordinates, compute the center
    of the box and normals.

    Args:
        label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
        "height", "width", "lenth".

    Returns: a numpy array [3, 3] containing [wx, wy, wz]^T, a [3] lower
        bound and a [3] upper bound.
    """
    box3d_points = box3d_to_cam_points(label, expend_factor)
    box3d_points_xyz = box3d_points.xyz
    wx = box3d_points_xyz[[0], :] - box3d_points_xyz[[4], :]
    lx = np.matmul(wx, box3d_points_xyz[4, :])
    ux = np.matmul(wx, box3d_points_xyz[0, :])
    wy = box3d_points_xyz[[0], :] - box3d_points_xyz[[1], :]
    ly = np.matmul(wy, box3d_points_xyz[1, :])
    uy = np.matmul(wy, box3d_points_xyz[0, :])
    wz = box3d_points_xyz[[0], :] - box3d_points_xyz[[3], :]
    lz = np.matmul(wz, box3d_points_xyz[3, :])
    uz = np.matmul(wz, box3d_points_xyz[0, :])
    return(np.concatenate([wx, wy, wz], axis=0),
        np.concatenate([lx, ly, lz]), np.concatenate([ux, uy, uz]))

def sel_xyz_in_box3d(label, xyz, expend_factor=(1.0, 1.0, 1.0)):
    """Select points in a 3D box.

    Args:
        label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
        "height", "width", "lenth".

    Returns: a bool mask indicating points inside a 3D box.
    """

    normals, lower, upper = box3d_to_normals(label, expend_factor)
    projected = np.matmul(xyz, np.transpose(normals))
    points_in_x = np.logical_and(projected[:, 0] > lower[0],
        projected[:, 0] < upper[0])
    points_in_y = np.logical_and(projected[:, 1] > lower[1],
        projected[:, 1] < upper[1])
    points_in_z = np.logical_and(projected[:, 2] > lower[2],
        projected[:, 2] < upper[2])
    mask = np.logical_and.reduce((points_in_x, points_in_y, points_in_z))
    return mask

def sel_xyz_in_box2d(label, xyz, expend_factor=(1.0, 1.0, 1.0)):
    """Select points in a 3D box.

    Args:
        label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
        "height", "width", "lenth".

    Returns: a bool mask indicating points inside a 3D box.
    """

    normals, lower, upper = box3d_to_normals(label, expend_factor)
    normals, lower, upper = normals[1:], lower[1:], upper[1:]
    projected = np.matmul(xyz, np.transpose(normals))
    points_in_y = np.logical_and(projected[:, 0] > lower[0],
        projected[:, 0] < upper[0])
    points_in_z = np.logical_and(projected[:, 1] > lower[1],
        projected[:, 1] < upper[1])
    mask = np.logical_and.reduce((points_in_y, points_in_z))
    return mask

class KittiDataset(object):
    """A class to interact with KITTI dataset."""

    def __init__(self, image_dir, point_dir, calib_dir, label_dir,
        index_filename=None, is_training=True, is_raw=False, difficulty=-100,
        num_classes=8):
        """
        Args:
            image_dir: a string of the path to image folder.
            point_dir: a string of the path to point cloud data folder.
            calib_dir: a string of the path to the calibration matrices.
            label_dir: a string of the path to the label folder.
            index_filename: a string containing a path an index file.
        """

        self._image_dir = image_dir
        self._point_dir = point_dir
        self._calib_dir = calib_dir
        self._label_dir = label_dir
        self._index_filename = index_filename
        if index_filename:
            self._file_list = self._read_index_file(index_filename)
        else:
            self._file_list = self._get_file_list(self._image_dir)
        self._verify_file_list(
            self._image_dir, self._point_dir, self._label_dir, self._calib_dir,
            self._file_list, is_training, is_raw)
        self._is_training = is_training
        self._is_raw = is_raw
        self.num_classes = num_classes
        self.difficulty = difficulty
        self._max_image_height = 376
        self._max_image_width = 1242

    def __str__(self):
        """Generate a string summary of the dataset"""
        summary_string = ('Dataset Summary:\n'
            +'image_dir=%s\n' % self._image_dir
            +'point_dir=%s\n' % self._point_dir
            +'calib_dir=%s\n' % self._calib_dir
            +'label_dir=%s\n' % self._label_dir
            +'index_filename=%s\n' % self._index_filename
            +'Total number of sampels: %d\n' % self.num_files)
        statics = self.get_statics()
        return summary_string + statics

    def get_statics(self):
        import matplotlib.pyplot as plt
        """Get statics of objects inside the dataset"""
        x_dict = defaultdict(list)
        y_dict = defaultdict(list)
        z_dict = defaultdict(list)
        h_dict = defaultdict(list)
        w_dict = defaultdict(list)
        l_dict = defaultdict(list)
        view_angle_dict = defaultdict(list)
        yaw_dict = defaultdict(list)
        for frame_idx in range(self.num_files):
            labels = self.get_label(frame_idx)
            for label in labels:
                if label['ymin'] > 0:
                    if label['ymax'] - label['ymin'] > 25:
                        object_name = label['name']
                        h_dict[object_name].append(label['height'])
                        w_dict[object_name].append(label['width'])
                        l_dict[object_name].append(label['length'])
                        x_dict[object_name].append(label['x3d'])
                        y_dict[object_name].append(label['y3d'])
                        z_dict[object_name].append(label['z3d'])
                        view_angle_dict[object_name].append(
                            np.arctan(label['x3d']/label['z3d']))
                        yaw_dict[object_name].append(label['yaw'])
        plt.scatter(z_dict['Pedestrian'], np.array(l_dict['Pedestrian']))
        plt.title('Scatter plot pythonspot.com')
        plt.show()
        # compute ingore statics
        import nms
        truncation_rates = []
        no_truncation_rates = []
        image_height = []
        image_width = []
        for frame_idx in range(self.num_files):
            labels = self.get_label(frame_idx)
            calib = self.get_calib(frame_idx)
            image = self.get_image(frame_idx)
            image_height.append(image.shape[0])
            image_width.append(image.shape[1])
            for label in labels:
                if label['name'] == 'Car':
                    # too small
                    if label['ymax'] - label['ymin'] < 25:
                        object_name = label['name']
                        h_dict['ignored_by_height'].append(label['height'])
                        w_dict['ignored_by_height'].append(label['width'])
                        l_dict['ignored_by_height'].append(label['length'])
                        x_dict['ignored_by_height'].append(label['x3d'])
                        y_dict['ignored_by_height'].append(label['y3d'])
                        z_dict['ignored_by_height'].append(label['z3d'])
                        view_angle_dict['ignored_by_height'].append(
                            np.arctan(label['x3d']/label['z3d']))
                        yaw_dict['ignored_by_height'].append(label['yaw'])
                    if label['truncation'] > 0.5:
                        h_dict['ignored_by_truncation'].append(label['height'])
                        w_dict['ignored_by_truncation'].append(label['width'])
                        l_dict['ignored_by_truncation'].append(label['length'])
                        x_dict['ignored_by_truncation'].append(label['x3d'])
                        y_dict['ignored_by_truncation'].append(label['y3d'])
                        z_dict['ignored_by_truncation'].append(label['z3d'])
                        view_angle_dict['ignored_by_truncation'].append(
                            np.arctan(label['x3d']/label['z3d']))
                        yaw_dict['ignored_by_truncation'].append(label['yaw'])
                    detection_boxes_3d = np.array(
                        [[label['x3d'], label['y3d'], label['z3d'],
                        label['length'], label['height'], label['width'],
                        label['yaw']]])
                    detection_boxes_3d_corners = nms.boxes_3d_to_corners(
                        detection_boxes_3d)
                    corners_cam_points = Points(
                        xyz=detection_boxes_3d_corners[0], attr=None)
                    corners_img_points = self.cam_points_to_image(
                        corners_cam_points, calib)
                    corners_xy = corners_img_points.xyz[:, :2]
                    xmin, ymin = np.amin(corners_xy, axis=0)
                    xmax, ymax = np.amax(corners_xy, axis=0)
                    clip_xmin = max(xmin, 0.0)
                    clip_ymin = max(ymin, 0.0)
                    clip_xmax = min(xmax, 1242.0)
                    clip_ymax = min(ymax, 375.0)
                    height = clip_ymax - clip_ymin
                    truncation_rate = 1.0 - \
                        (clip_ymax - clip_ymin)*(clip_xmax - clip_xmin)\
                        /((ymax - ymin)*(xmax - xmin))
                    if label['truncation'] > 0.5:
                        truncation_rates.append(truncation_rate)
                    else:
                        no_truncation_rates.append(truncation_rate)
                    if label['occlusion'] > 2:
                        h_dict['ignored_by_occlusion'].append(label['height'])
                        w_dict['ignored_by_occlusion'].append(label['width'])
                        l_dict['ignored_by_occlusion'].append(label['length'])
                        x_dict['ignored_by_occlusion'].append(label['x3d'])
                        y_dict['ignored_by_occlusion'].append(label['y3d'])
                        z_dict['ignored_by_occlusion'].append(label['z3d'])
                        view_angle_dict['ignored_by_occlusion'].append(
                            np.arctan(label['x3d']/label['z3d']))
                        yaw_dict['ignored_by_occlusion'].append(label['yaw'])
        statics = ""
        for object_name in h_dict:
            print(object_name+"l="+str(
                np.histogram(l_dict[object_name], 10, density=True)))
            if len(h_dict[object_name]) == 0:
                continue
            statics += (str(len(h_dict[object_name]))+ " "+ str(object_name)
                    + " "
                       + "mh=" + str(np.min(h_dict[object_name])) + " "
                               + str(np.median(h_dict[object_name])) + " "
                               + str(np.max(h_dict[object_name])) + "; "
                       + "mw=" + str(np.min(w_dict[object_name])) + " "
                               + str(np.median(w_dict[object_name])) + " "
                               + str(np.max(w_dict[object_name])) + "; "
                       + "ml=" + str(np.min(l_dict[object_name])) + " "
                               + str(np.median(l_dict[object_name])) + " "
                               + str(np.max(l_dict[object_name])) + "; "
                       + "mx=" + str(np.min(x_dict[object_name])) + " "
                               + str(np.median(x_dict[object_name])) + " "
                               + str(np.max(x_dict[object_name])) + "; "
                       + "my=" + str(np.min(y_dict[object_name])) + " "
                               + str(np.median(y_dict[object_name])) + " "
                               + str(np.max(y_dict[object_name])) + "; "
                       + "mz=" + str(np.min(z_dict[object_name])) + " "
                               + str(np.median(z_dict[object_name])) + " "
                               + str(np.max(z_dict[object_name])) + "; "
                       + "mA=" + str(np.min(view_angle_dict[object_name]))
                       + " "
                              + str(np.median(view_angle_dict[object_name]))
                       + " "
                              + str(np.max(view_angle_dict[object_name])) + "; "
                       + "mY=" + str(np.min(yaw_dict[object_name])) + " "
                               + str(np.median(yaw_dict[object_name])) + " "
                               + str(np.max(yaw_dict[object_name])) + "; "
                       + "image_height" + str(np.min(image_height)) + " "
                       + str(np.max(image_height)) +" "
                       + "image_width" + str(np.min(image_width)) + " "
                       + str(np.max(image_width)) + ";"
                       "\n")

        return statics

    @property
    def num_files(self):
        return len(self._file_list)

    def _read_index_file(self, index_filename):
        """Read an index file containing the filenames.

        Args:
            index_filename: a string containing the path to an index file.

        Returns: a list of filenames.
        """

        file_list = []
        with open(index_filename, 'r') as f:
            for line in f:
                file_list.append(line.rstrip('\n').split('.')[0])
        return file_list

    def _get_file_list(self, image_dir):
        """Load all filenames from image_dir.

        Args:
            image_dir: a string of path to the image folder.

        Returns: a list of filenames.
        """

        file_list = [f.split('.')[0]
            for f in os.listdir(image_dir) if isfile(join(image_dir, f))]
        file_list.sort()
        return file_list

    def _verify_file_list(
        self, image_dir, point_dir, label_dir, calib_dir, file_list,
        is_training, is_raw):
        """Varify the files in file_list exist.

        Args:
            image_dir: a string of the path to image folder.
            point_dir: a string of the path to point cloud data folder.
            label_dir: a string of the path to the label folder.
            calib_dir: a string of the path to the calibration folder.
            file_list: a list of filenames.
            is_training: if False, label_dir is not verified.

        Raise: assertion error when file in file_list is not complete.
        """

        for f in file_list:
            image_file = join(image_dir, f)+'.png'
            point_file = join(point_dir, f)+'.bin'
            label_file = join(label_dir, f)+'.txt'
            calib_file = join(calib_dir, f)+'.txt'
            assert isfile(image_file), "Image %s does not exist" % image_file
            assert isfile(point_file), "Point %s does not exist" % point_file
            if not is_raw:
                assert isfile(calib_file), \
                    "Calib %s does not exist" % calib_file
            if is_training:
                assert isfile(label_file), \
                    "Label %s does not exist" % label_file

    def downsample_by_voxel(self, points, voxel_size, method='AVERAGE'):
        """Downsample point cloud by voxel.

        points: a Points namedtuple containing "xyz" and "attr".
        voxel_size: the size of voxel cells used for downsampling.
        method: 'AVERAGE', all points inside a voxel cell are averaged
        including xyz and attr.
        """
        # create voxel grid
        xmax, ymax, zmax = np.amax(points.xyz, axis=0)
        xmin, ymin, zmin = np.amin(points.xyz, axis=0)
        dim_x = int((xmax - xmin) / voxel_size + 1)
        dim_y = int((ymax - ymin) / voxel_size + 1)
        dim_z = int((zmax - zmin) / voxel_size + 1)
        voxel_account = {}
        xyz_idx = np.int32(
            (points.xyz - np.asarray([[xmin, ymin, zmin]])) / voxel_size)
        for pidx in range(xyz_idx.shape[0]):
            x_idx = xyz_idx[pidx, 0]
            y_idx = xyz_idx[pidx, 1]
            z_idx = xyz_idx[pidx, 2]
            # TODO check bug impact
            key = x_idx + y_idx*dim_x + z_idx*dim_y*dim_x
            if key in voxel_account:
                voxel_account[key].append(pidx)
            else:
                voxel_account[key] = [pidx]
        # compute voxel points
        downsampled_xyz_list = []
        if points.attr is not None:
            downsampled_attr_list = []
        if method == 'AVERAGE':
            for idx, pidx_list in voxel_account.iteritems():
                if len(pidx_list) > 0:
                    downsampled_xyz_list.append(
                        np.mean(points.xyz[pidx_list, :],
                            axis=0, keepdims=True))
                    if points.attr is not None:
                        downsampled_attr_list.append(
                            np.mean(points.attr[pidx_list, :],
                                axis=0, keepdims=True))
        if points.attr is not None:
            return Points(xyz=np.vstack(downsampled_xyz_list),
                attr=np.vstack(downsampled_attr_list))
        else:
            return Points(xyz=np.vstack(downsampled_xyz_list),
                attr=None)

    def get_calib(self, frame_idx):
        """Load calibration matrices and compute calibrations.

        Args:
            frame_idx: the index of the frame to read.

        Returns: a dictionary of calibrations.
        """

        calib_file = join(self._calib_dir, self._file_list[frame_idx])+'.txt'
        with open(calib_file, 'r') as f:
            calib = {}
            for line in f:
                fields = line.split(' ')
                matrix_name = fields[0].rstrip(':')
                matrix = np.array(fields[1:], dtype=np.float32)
                calib[matrix_name] = matrix
        calib['P2'] = calib['P2'].reshape(3, 4)
        calib['R0_rect'] = calib['R0_rect'].reshape(3,3)
        calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3,4)
        R0_rect = np.eye(4)
        R0_rect[:3, :3] = calib['R0_rect']
        calib['velo_to_rect'] = np.vstack([calib['Tr_velo_to_cam'],[0,0,0,1]])
        calib['cam_to_image'] = np.hstack([calib['P2'][:, 0:3], [[0],[0],[0]]])
        calib['rect_to_cam'] = np.hstack([
            calib['R0_rect'],
            np.matmul(
                np.linalg.inv(calib['P2'][:, 0:3]), calib['P2'][:, [3]])])
        calib['rect_to_cam'] = np.vstack([calib['rect_to_cam'],
            [0,0,0,1]])
        calib['velo_to_cam'] = np.matmul(calib['rect_to_cam'],
            calib['velo_to_rect'])
        calib['cam_to_velo'] = np.linalg.inv(calib['velo_to_cam'])
        # senity check
        calib['velo_to_image'] = np.matmul(calib['cam_to_image'],
            calib['velo_to_cam'])
        assert np.isclose(calib['velo_to_image'],
            np.matmul(np.matmul(calib['P2'], R0_rect),
            calib['velo_to_rect'])).all()
        return calib

    def get_raw_calib(self, calib_velo_to_cam_path, calib_cam_to_cam_path):
        """Read calibrations in kitti raw dataset."""
        with open(calib_cam_to_cam_path, 'r') as f:
            calib = {}
            for line in f:
                line = line.rstrip('\n')
                fields = line.split(':')
                calib[fields[0]] = fields[1]
        calib['corner_dist'] = np.array(
            calib['corner_dist'], dtype=np.float32)
        for i in range(4):
            calib['S_0%d'%i] = np.array(
                calib['S_0%d'%i].split(' ')[1:], dtype=np.float32).reshape(1,2)
            calib['K_0%d'%i] = np.array(
                calib['K_0%d'%i].split(' ')[1:], dtype=np.float32).reshape(3,3)
            calib['D_0%d'%i] = np.array(
                calib['D_0%d'%i].split(' ')[1:], dtype=np.float32).reshape(1,5)
            calib['R_0%d'%i] = np.array(
                calib['R_0%d'%i].split(' ')[1:], dtype=np.float32).reshape(3,3)
            calib['T_0%d'%i] = np.array(
                calib['T_0%d'%i].split(' ')[1:], dtype=np.float32).reshape(3,1)
            calib['S_rect_0%d'%i] = np.array(
                calib['S_rect_0%d'%i].split(' ')[1:],
                    dtype=np.float32).reshape(1,2)
            calib['R_rect_0%d'%i] = np.array(
                calib['R_rect_0%d'%i].split(' ')[1:],
                    dtype=np.float32).reshape(3,3)
            calib['P_rect_0%d'%i] = np.array(
                calib['P_rect_0%d'%i].split(' ')[1:],
                    dtype=np.float32).reshape(3,4)
        with open(calib_velo_to_cam_path, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                fields = line.split(':')
                calib[fields[0]] = fields[1]
        calib['R'] = np.array(
            calib['R'].split(' ')[1:], dtype=np.float32).reshape(3,3)
        calib['T'] = np.array(
            calib['T'].split(' ')[1:], dtype=np.float32).reshape(3,1)
        calib['Tr_velo_to_cam'] = np.vstack(
            [np.hstack([calib['R'], calib['T']]),[0,0,0,1]])

        R0_rect = np.eye(4)
        R0_rect[:3, :3] = calib['R_rect_00']
        T2 = np.eye(4)
        T2[0, 3] = calib['P_rect_02'][0, 3]/calib['P_rect_02'][0, 0]
        calib['velo_to_cam'] = T2.dot(R0_rect.dot(calib['Tr_velo_to_cam']))
        calib['cam_to_image'] = np.hstack(
            [calib['P_rect_02'][:, 0:3], [[0],[0],[0]]])
        calib['velo_to_image'] = np.matmul(calib['cam_to_image'],
            calib['velo_to_cam'])
        return calib

    def get_filename(self, frame_idx):
        """Get the filename based on frame_idx.

        Args:
            frame_idx: the index of the frame to get.

        Returns: a string containing the filename.
        """
        return self._file_list[frame_idx]

    def get_velo_points(self, frame_idx, xyz_range=None):
        """Load velo points from frame_idx.

        Args:
            frame_idx: the index of the frame to read.

        Returns: Points.
        """

        point_file = join(self._point_dir, self._file_list[frame_idx])+'.bin'
        velo_data = np.fromfile(point_file, dtype=np.float32).reshape(-1, 4)
        velo_points = velo_data[:,:3]
        reflections = velo_data[:,[3]]
        if xyz_range is not None:
            x_range, y_range, z_range = xyz_range
            mask =(
                velo_points[:, 0] > x_range[0])*(velo_points[:, 0] < x_range[1])
            mask *=(
                velo_points[:, 1] > y_range[0])*(velo_points[:, 1] < y_range[1])
            mask *=(
                velo_points[:, 2] > z_range[0])*(velo_points[:, 2] < z_range[1])
            return Points(xyz = velo_points[mask], attr = reflections[mask])
        return Points(xyz = velo_points, attr = reflections)

    def get_cam_points(self, frame_idx,
        downsample_voxel_size=None, calib=None, xyz_range=None):
        """Load velo points and convert them to camera coordinates.

        Args:
            frame_idx: the index of the frame to read.

        Returns: Points.
        """
        velo_points = self.get_velo_points(frame_idx, xyz_range=xyz_range)
        if calib is None:
            calib = self.get_calib(frame_idx)
        cam_points = self.velo_points_to_cam(velo_points, calib)
        if downsample_voxel_size is not None:
            cam_points = downsample_by_average_voxel(cam_points,
                downsample_voxel_size)
        return cam_points


    def calc_distances(self, p0, points):
        return ((p0 - points)**2).sum(axis=1)

    def farthest_first(self, pts, K):
        farthest_pts = np.zeros((K, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, K):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances,
                self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

    def get_cam_points_in_image(self, frame_idx, downsample_voxel_size=None,
        calib=None, xyz_range=None):
        """Load velo points and remove points that are not observed by camera.
        """
        if calib is None:
            calib = self.get_calib(frame_idx)
        cam_points = self.get_cam_points(frame_idx, downsample_voxel_size,
            calib=calib, xyz_range=xyz_range)
        image = self.get_image(frame_idx)
        height = image.shape[0]
        width = image.shape[1]
        front_cam_points_idx = cam_points.xyz[:,2] > 0.1
        front_cam_points = Points(cam_points.xyz[front_cam_points_idx, :],
            cam_points.attr[front_cam_points_idx, :])
        img_points = self.cam_points_to_image(front_cam_points, calib)
        img_points_in_image_idx = np.logical_and.reduce(
            [img_points.xyz[:,0]>0, img_points.xyz[:,0]<width,
             img_points.xyz[:,1]>0, img_points.xyz[:,1]<height])
        cam_points_in_img = Points(
            xyz = front_cam_points.xyz[img_points_in_image_idx,:],
            attr = front_cam_points.attr[img_points_in_image_idx,:])
        return cam_points_in_img

    def get_cam_points_in_image_with_rgb(self, frame_idx,
        downsample_voxel_size=None, calib=None, xyz_range=None):
        """Get camera points that are visible in image and append image color
        to the points as attributes."""
        if calib is None:
            calib = self.get_calib(frame_idx)
        cam_points = self.get_cam_points(frame_idx, downsample_voxel_size,
            calib = calib, xyz_range=xyz_range)
        front_cam_points_idx = cam_points.xyz[:,2] > 0.1
        front_cam_points = Points(cam_points.xyz[front_cam_points_idx, :],
            cam_points.attr[front_cam_points_idx, :])
        image = self.get_image(frame_idx)
        height = image.shape[0]
        width = image.shape[1]
        img_points = self.cam_points_to_image(front_cam_points, calib)
        img_points_in_image_idx = np.logical_and.reduce(
            [img_points.xyz[:,0]>0, img_points.xyz[:,0]<width,
             img_points.xyz[:,1]>0, img_points.xyz[:,1]<height])
        cam_points_in_img = Points(
            xyz = front_cam_points.xyz[img_points_in_image_idx,:],
            attr = front_cam_points.attr[img_points_in_image_idx,:])
        cam_points_in_img_with_rgb = self.rgb_to_cam_points(cam_points_in_img,
            image, calib)
        return cam_points_in_img_with_rgb

    def get_image(self, frame_idx):
        """Load the image from frame_idx.

        Args:
            frame_idx: the index of the frame to read.

        Returns: cv2.matrix
        """

        image_file = join(self._image_dir, self._file_list[frame_idx])+'.png'
        return cv2.imread(image_file)

    def get_label(self, frame_idx, no_orientation=False):
        """Load bbox labels from frame_idx frame.

        Args:
            frame_idx: the index of the frame to read.

        Returns: a list of object label dictionaries.
        """

        MIN_HEIGHT = [40, 25, 25]
        MAX_OCCLUSION = [0, 1, 2]
        MAX_TRUNCATION = [0.15, 0.3, 0.5]
        label_file = join(self._label_dir, self._file_list[frame_idx])+'.txt'
        label_list = []
        with open(label_file, 'r') as f:
            for line in f:
                label={}
                line = line.strip()
                if line == '':
                    continue
                fields = line.split(' ')
                label['name'] = fields[0]
                # 0=visible 1=partly occluded, 2=fully occluded, 3=unknown
                label['truncation'] = float(fields[1])
                label['occlusion'] = int(fields[2])
                label['alpha'] =  float(fields[3])
                label['xmin'] =  float(fields[4])
                label['ymin'] =  float(fields[5])
                label['xmax'] =  float(fields[6])
                label['ymax'] =  float(fields[7])
                label['height'] =  float(fields[8])
                label['width'] =  float(fields[9])
                label['length'] =  float(fields[10])
                label['x3d'] =  float(fields[11])
                label['y3d'] =  float(fields[12])
                label['z3d'] =  float(fields[13])
                label['yaw'] =  float(fields[14])
                if len(fields) > 15:
                    label['score'] =  float(fields[15])
                if self.difficulty > -1:
                    if label['truncation'] > MAX_TRUNCATION[self.difficulty]:
                        continue
                    if label['occlusion'] > MAX_OCCLUSION[self.difficulty]:
                        continue
                    if (label['ymax'] - label['ymin']
                        ) < MIN_HEIGHT[self.difficulty]:
                        continue
                label_list.append(label)
        return label_list

    def box3d_to_cam_points(self, label, expend_factor=(1.0, 1.0, 1.0)):
        """Project 3D box into camera coordinates.
        Args:
            label: a dictionary containing "x3d", "y3d", "z3d", "yaw", "height"
                "width", "length".

        Returns: a numpy array [8, 3] representing the corners of the 3d box in
            camera coordinates.
        """

        yaw = label['yaw']
        R = np.array([[np.cos(yaw),  0,  np.sin(yaw)],
                      [0,            1,  0          ],
                      [-np.sin(yaw), 0,  np.cos(yaw)]]);
        h = label['height']
        delta_h = h*(expend_factor[0]-1)
        w = label['width']*expend_factor[1]
        l = label['length']*expend_factor[2]
        corners = np.array([[ l/2,  delta_h/2,  w/2],  # front up right
                            [ l/2,  delta_h/2, -w/2],  # front up left
                            [-l/2,  delta_h/2, -w/2],  # back up left
                            [-l/2,  delta_h/2,  w/2],  # back up right
                            [ l/2, -h-delta_h/2,  w/2],  # front down right
                            [ l/2, -h-delta_h/2, -w/2],  # front down left
                            [-l/2, -h-delta_h/2, -w/2],  # back down left
                            [-l/2, -h-delta_h/2,  w/2]]) # back down right
        r_corners = corners.dot(np.transpose(R))
        tx = label['x3d']
        ty = label['y3d']
        tz = label['z3d']
        cam_points_xyz = r_corners+np.array([tx, ty, tz])
        return Points(xyz = cam_points_xyz, attr = None)

    def boxes_3d_to_line_set(self, boxes_3d, boxes_color=None):
        points = []
        edges = []
        colors = []
        for i, box_3d in enumerate(boxes_3d):
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
            points.append(cam_points_xyz)
            edges.append(
               np.array([[0, 1], [0, 4], [0, 3],
                        [1, 2], [1, 5], [2, 3],
                        [2, 6], [3, 7], [4, 5],
                        [4, 7], [5, 6], [6, 7]])+i*8)
            if boxes_color is None:
                colors.append(np.tile([[1.0, 0.0, 0.0]], [12, 1]))
            else:
                colors.append(np.tile(boxes_color[[i], :], [12, 1]))
        if len(points) == 0:
            return None, None, None
        return np.vstack(points), np.vstack(edges), np.vstack(colors)

    def draw_open3D_box(self, label, expend_factor=(1.0, 1.0, 1.0)):
        """Draw a 3d box using open3d.

        Args:
            label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
            "height", "width", "lenth".

        returns: a open3d mesh object.
        """
        yaw = label['yaw']
        R = np.array([[np.cos(yaw),  0,  np.sin(yaw)],
                      [0,            1,  0          ],
                      [-np.sin(yaw), 0,  np.cos(yaw)]]);
        Rh = np.array([ [1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]])

        Rl = np.array([ [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]])

        h = label['height']
        delta_h = h*(expend_factor[0]-1)
        w = label['width']*expend_factor[1]
        l = label['length']*expend_factor[2]
        print((l, w, h))
        tx = label['x3d']
        ty = label['y3d']
        tz = label['z3d']

        box_offset = np.array([ [ l/2,  -h/2-delta_h/2,  w/2],
                                [ l/2,  -h/2-delta_h/2, -w/2],
                                [-l/2,  -h/2-delta_h/2, -w/2],
                                [-l/2,  -h/2-delta_h/2,  w/2],

                                [ l/2, delta_h/2, 0],
                                [ -l/2, delta_h/2, 0],
                                [l/2, -h-delta_h/2, 0],
                                [-l/2, -h-delta_h/2, 0],

                                [0, delta_h/2, w/2],
                                [0, delta_h/2, -w/2],
                                [0, -h-delta_h/2, w/2],
                                [0, -h-delta_h/2, -w/2]])

        transform = np.matmul(R, np.transpose(box_offset))
        transform = transform + np.array([[tx], [ty], [tz]])
        transform = np.vstack((transform, np.ones((1, 12))))
        hrotation = np.vstack((R.dot(Rh), np.zeros((1,3))))
        lrotation = np.vstack((R.dot(Rl), np.zeros((1,3))))
        wrotation = np.vstack((R, np.zeros((1,3))))

        h1_cylinder = open3d.create_mesh_cylinder(radius = h/100, height = h)
        h1_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
        h1_cylinder.transform(np.hstack((hrotation, transform[:, [0]])))

        h2_cylinder = open3d.create_mesh_cylinder(radius = h/100, height = h)
        h2_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
        h2_cylinder.transform(np.hstack((hrotation, transform[:, [1]])))

        h3_cylinder = open3d.create_mesh_cylinder(radius = h/100, height = h)
        h3_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
        h3_cylinder.transform(np.hstack((hrotation, transform[:, [2]])))

        h4_cylinder = open3d.create_mesh_cylinder(radius = h/100, height = h)
        h4_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
        h4_cylinder.transform(np.hstack((hrotation, transform[:, [3]])))

        w1_cylinder = open3d.create_mesh_cylinder(radius = w/100, height = w)
        w1_cylinder.paint_uniform_color([0.9, 0.1, 0.1])
        w1_cylinder.transform(np.hstack((wrotation, transform[:, [4]])))

        w2_cylinder = open3d.create_mesh_cylinder(radius = w/100, height = w)
        w2_cylinder.paint_uniform_color([0.9, 0.1, 0.1])
        w2_cylinder.transform(np.hstack((wrotation, transform[:, [5]])))

        w3_cylinder = open3d.create_mesh_cylinder(radius = w/100, height = w)
        w3_cylinder.paint_uniform_color([0.9, 0.1, 0.1])
        w3_cylinder.transform(np.hstack((wrotation, transform[:, [6]])))

        w4_cylinder = open3d.create_mesh_cylinder(radius = w/100, height = w)
        w4_cylinder.paint_uniform_color([0.9, 0.1, 0.1])
        w4_cylinder.transform(np.hstack((wrotation, transform[:, [7]])))

        l1_cylinder = open3d.create_mesh_cylinder(radius = l/100, height = l)
        l1_cylinder.paint_uniform_color([0.1, 0.1, 0.9])
        l1_cylinder.transform(np.hstack((lrotation, transform[:, [8]])))

        l2_cylinder = open3d.create_mesh_cylinder(radius = l/100, height = l)
        l2_cylinder.paint_uniform_color([0.1, 0.1, 0.9])
        l2_cylinder.transform(np.hstack((lrotation, transform[:, [9]])))

        l3_cylinder = open3d.create_mesh_cylinder(radius = l/100, height = l)
        l3_cylinder.paint_uniform_color([0.1, 0.1, 0.9])
        l3_cylinder.transform(np.hstack((lrotation, transform[:, [10]])))

        l4_cylinder = open3d.create_mesh_cylinder(radius = l/100, height = l)
        l4_cylinder.paint_uniform_color([0.1, 0.1, 0.9])
        l4_cylinder.transform(np.hstack((lrotation, transform[:, [11]])))

        return [h1_cylinder, h2_cylinder, h3_cylinder, h4_cylinder,
                w1_cylinder, w2_cylinder, w3_cylinder, w4_cylinder,
                l1_cylinder, l2_cylinder, l3_cylinder, l4_cylinder]

    def box3d_to_normals(self, label, expend_factor=(1.0, 1.0, 1.0)):
        """Project a 3D box into camera coordinates, compute the center
        of the box and normals.

        Args:
            label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
            "height", "width", "lenth".

        Returns: a numpy array [3, 3] containing [wx, wy, wz]^T, a [3] lower
            bound and a [3] upper bound.
        """
        box3d_points = self.box3d_to_cam_points(label, expend_factor)
        box3d_points_xyz = box3d_points.xyz
        wx = box3d_points_xyz[[0], :] - box3d_points_xyz[[4], :]
        lx = np.matmul(wx, box3d_points_xyz[4, :])
        ux = np.matmul(wx, box3d_points_xyz[0, :])
        wy = box3d_points_xyz[[0], :] - box3d_points_xyz[[1], :]
        ly = np.matmul(wy, box3d_points_xyz[1, :])
        uy = np.matmul(wy, box3d_points_xyz[0, :])
        wz = box3d_points_xyz[[0], :] - box3d_points_xyz[[3], :]
        lz = np.matmul(wz, box3d_points_xyz[3, :])
        uz = np.matmul(wz, box3d_points_xyz[0, :])
        return(np.concatenate([wx, wy, wz], axis=0),
            np.concatenate([lx, ly, lz]), np.concatenate([ux, uy, uz]))

    def sel_points_in_box3d(self, label, points, expend_factor=(1.0, 1.0, 1.0)):
        """Select points in a 3D box.

        Args:
            label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
            "height", "width", "lenth".

        Returns: a bool mask indicating points inside a 3D box.
        """

        normals, lower, upper = self.box3d_to_normals(label, expend_factor)
        projected = np.matmul(points.xyz, np.transpose(normals))
        points_in_x = np.logical_and(projected[:, 0] > lower[0],
            projected[:, 0] < upper[0])
        points_in_y = np.logical_and(projected[:, 1] > lower[1],
            projected[:, 1] < upper[1])
        points_in_z = np.logical_and(projected[:, 2] > lower[2],
            projected[:, 2] < upper[2])
        mask = np.logical_and.reduce((points_in_x, points_in_y, points_in_z))
        return mask

    def sel_xyz_in_box3d(self, label, xyz, expend_factor=(1.0, 1.0, 1.0)):
        """Select points in a 3D box.

        Args:
            label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
            "height", "width", "lenth".

        Returns: a bool mask indicating points inside a 3D box.
        """

        normals, lower, upper = self.box3d_to_normals(label, expend_factor)
        projected = np.matmul(xyz, np.transpose(normals))
        points_in_x = np.logical_and(projected[:, 0] > lower[0],
            projected[:, 0] < upper[0])
        points_in_y = np.logical_and(projected[:, 1] > lower[1],
            projected[:, 1] < upper[1])
        points_in_z = np.logical_and(projected[:, 2] > lower[2],
            projected[:, 2] < upper[2])
        mask = np.logical_and.reduce((points_in_x, points_in_y, points_in_z))
        return mask

    def rgb_to_cam_points(self, points, image, calib):
        """Append rgb info to camera points"""

        img_points = self.cam_points_to_image(points, calib)
        rgb = image[np.int32(img_points.xyz[:,1]),
            np.int32(img_points.xyz[:,0]),::-1].astype(np.float32)/255
        return Points(points.xyz, np.hstack([points.attr, rgb]))

    def velo_points_to_cam(self, points, calib):
        """Convert points in velodyne coordinates to camera coordinates.

        """
        cam_xyz = np.matmul(points.xyz,
            np.transpose(calib['velo_to_cam'])[:3,:3].astype(np.float32))
        cam_xyz += np.transpose(
            calib['velo_to_cam'])[[3], :3].astype(np.float32)
        return Points(xyz = cam_xyz, attr = points.attr)

    def velo_to_cam(self, points_xyz, calib):
        """Convert points in velodyne coordinates to camera coordinates.

        """

        velo_xyz1 = np.hstack([points_xyz, np.ones([points_xyz.shape[0],1])])
        cam_xyz = np.transpose(
            np.matmul(calib['velo_to_cam'], np.transpose(velo_xyz1))[:3, :])
        return cam_xyz

    def cam_points_to_velo(self, points, calib):
        """Convert points from camera coordinates to velodyne coordinates.

        Args:
            points: a [N, 3] float32 numpy array.

        Returns: a [N, 3] float32 numpy array.
        """

        cam_xyz1 = np.hstack([points.xyz, np.ones([points.xyz.shape[0],1])])
        velo_xyz = np.matmul(cam_xyz1, np.transpose(calib['cam_to_velo']))[:,:3]
        return Points(xyz = velo_xyz, attr = points.attr)

    def cam_to_velo(self, points_xyz, calib):
        cam_xyz1 = np.hstack([points_xyz, np.ones([points_xyz.shape[0],1])])
        velo_xyz = np.matmul(cam_xyz1, np.transpose(calib['cam_to_velo']))[:,:3]
        return velo_xyz

    def cam_points_to_image(self, points, calib):
        """Convert camera points to image plane.

        Args:
            points: a [N, 3] float32 numpy array.

        Returns: points on image plane: a [M, 2] float32 numpy array,
                  a mask indicating points: a [N, 1] boolean numpy array.
        """

        cam_points_xyz1 = np.hstack(
            [points.xyz, np.ones([points.xyz.shape[0],1])])
        img_points_xyz = np.matmul(
            cam_points_xyz1, np.transpose(calib['cam_to_image']))
        img_points_xy1 = img_points_xyz/img_points_xyz[:,[2]]
        img_points = Points(img_points_xy1, points.attr)
        return img_points

    def velo_points_to_image(self, points, calib):
        """Convert points from velodyne coordinates to image coordinates. Points
        that behind the camera is removed.

        Args:
            points: a [N, 3] float32 numpy array.

        Returns: points on image plane: a [M, 2] float32 numpy array,
                 a mask indicating points: a [N, 1] boolean numpy array.
        """

        cam_points = self.velo_points_to_cam(points, calib)
        img_points = self.cam_points_to_image(cam_points, calib)
        return img_points

    def vis_draw_2d_box(self, image, label_list):
        """Draw 2D bounding boxes on the image.
        """
        color_list = [(0, 128, 0), (0, 255, 255), (0, 0, 128), (255, 255, 255)]
        for label in label_list:
            if label['name'] == 'DontCare':
                color = (255,191,0)
            else:
                color = color_list[label['occlusion']]
            xmin = int(label['xmin'])
            ymin = int(label['ymin'])
            xmax = int(label['xmax'])
            ymax = int(label['ymax'])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, '{:s}'.format(label['name']),
                (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)

    def vis_draw_3d_box(self, image, label_list, calib, color_map):
        """Draw 3D bounding boxes on the image.
        """
        for label in label_list:
            cam_points = self.box3d_to_cam_points(label)
            # if any(cam_points.xyz[:, 2]<0.1):
            #     # only draw 3D bounding box for objects in front of the camera
            #     continue
            img_points = self.cam_points_to_image(cam_points, calib)
            img_points_xy = img_points.xyz[:, 0:2].astype(np.int)
            color = color_map[label['name']][::-1]
            cv2.line(image, tuple(img_points_xy[0,:]),
                tuple(img_points_xy[1,:]),color,2)
            cv2.line(image, tuple(img_points_xy[1,:]),
                tuple(img_points_xy[5,:]),color,2)
            cv2.line(image, tuple(img_points_xy[5,:]),
                tuple(img_points_xy[4,:]),color,2)
            cv2.line(image, tuple(img_points_xy[4,:]),
                tuple(img_points_xy[0,:]),color,2)
            cv2.line(image, tuple(img_points_xy[1,:]),
                tuple(img_points_xy[2,:]),color,2)
            cv2.line(image, tuple(img_points_xy[2,:]),
                tuple(img_points_xy[6,:]),color,2)
            cv2.line(image, tuple(img_points_xy[6,:]),
                tuple(img_points_xy[5,:]),color,2)
            cv2.line(image, tuple(img_points_xy[2,:]),
                tuple(img_points_xy[3,:]),color,2)
            cv2.line(image, tuple(img_points_xy[3,:]),
                tuple(img_points_xy[7,:]),color,2)
            cv2.line(image, tuple(img_points_xy[7,:]),
                tuple(img_points_xy[6,:]),color,2)
            cv2.line(image, tuple(img_points_xy[3,:]),
                tuple(img_points_xy[0,:]),color,2)
            cv2.line(image, tuple(img_points_xy[4,:]),
                tuple(img_points_xy[7,:]),color,2)

    def inspect_points(self, frame_idx, downsample_voxel_size=None, calib=None,
        expend_factor=(1.0, 1.0, 1.0), no_orientation=False):
        """Inspect points inside dataset"""
        cam_points_in_img_with_rgb = self.get_cam_points_in_image_with_rgb(
            frame_idx, downsample_voxel_size=downsample_voxel_size, calib=calib)
        print("#(points)="+str(cam_points_in_img_with_rgb.xyz.shape))
        label_list = self.get_label(frame_idx, no_orientation=no_orientation)
        self.vis_points(cam_points_in_img_with_rgb,
            label_list, expend_factor=expend_factor)

    def assign_classaware_label_to_points(self, labels, xyz, expend_factor):
        """Assign class label and bounding boxes to xyz points. """
        # print("Yes, I am Here !!!!!!!!!!!!!!!!!!!!!")
        assert self.num_classes == 8
        num_points = xyz.shape[0]
        assert num_points > 0, "No point No prediction"
        assert xyz.shape[1] == 3
        # define label map
        label_map = {
            'Background': 0,
            'Car': 1,
            'Pedestrian': 3,
            'Cyclist': 5,
            'DontCare': 7
            }
        # by default, all points are assigned with background label 0.
        cls_labels = np.zeros((num_points, 1), dtype=np.int64)
        # 3d boxes for each point
        boxes_3d = np.zeros((num_points, 1, 7))
        valid_boxes = np.zeros((num_points, 1, 1), dtype=np.float32)
        # add label for each object
        for label in labels:
            obj_cls_string = label['name']
            obj_cls = label_map.get(obj_cls_string, 7)
            if obj_cls >= 1 and obj_cls <= 6:
                mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
                yaw = label['yaw']
                while yaw < -0.25*np.pi:
                    yaw += np.pi
                while yaw > 0.75*np.pi:
                    yaw -= np.pi
                if yaw < 0.25*np.pi:
                    # horizontal
                    cls_labels[mask, :] = obj_cls
                    boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
                        label['z3d'], label['length'], label['height'],
                        label['width'], yaw)
                    valid_boxes[mask, 0, :] = 1
                else:
                    # vertical
                    cls_labels[mask, :] = obj_cls+1
                    boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
                        label['z3d'], label['length'], label['height'],
                        label['width'], yaw)
                    valid_boxes[mask, 0, :] = 1
            else:
                if obj_cls_string != 'DontCare':
                    mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
                    cls_labels[mask, :] = obj_cls
                    valid_boxes[mask, 0, :] = 0.0
        return cls_labels, boxes_3d, valid_boxes, label_map

    def assign_classaware_car_label_to_points(self, labels, xyz, expend_factor):
        """Assign class label and bounding boxes to xyz points. """
        assert self.num_classes == 4
        num_points = xyz.shape[0]
        assert num_points > 0, "No point No prediction"
        assert xyz.shape[1] == 3
        # define label map
        label_map = {
            'Background': 0,
            'Car': 1,
            'DontCare': 3
            }
        # by default, all points are assigned with background label 0.
        cls_labels = np.zeros((num_points, 1), dtype=np.int64)
        # 3d boxes for each point
        boxes_3d = np.zeros((num_points, 1, 7))
        valid_boxes = np.zeros((num_points, 1, 1), dtype=np.float32)
        # add label for each object
        for label in labels:
            obj_cls_string = label['name']
            obj_cls = label_map.get(obj_cls_string, 3)
            if obj_cls >= 1 and obj_cls <= 2:
                mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
                yaw = label['yaw']
                while yaw < -0.25*np.pi:
                    yaw += np.pi
                while yaw > 0.75*np.pi:
                    yaw -= np.pi
                if yaw < 0.25*np.pi:
                    # horizontal
                    cls_labels[mask, :] = obj_cls
                    boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
                        label['z3d'], label['length'], label['height'],
                        label['width'], yaw)
                    valid_boxes[mask, 0, :] = 1
                else:
                    # vertical
                    cls_labels[mask, :] = obj_cls+1
                    boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
                        label['z3d'], label['length'], label['height'],
                        label['width'], yaw)
                    valid_boxes[mask, 0, :] = 1
            else:
                if obj_cls_string != 'DontCare':
                    mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
                    cls_labels[mask, :] = obj_cls
                    valid_boxes[mask, 0, :] = 0.0

        return cls_labels, boxes_3d, valid_boxes, label_map

    def assign_classaware_ped_and_cyc_label_to_points(self, labels, xyz,
        expend_factor):
        """Assign class label and bounding boxes to xyz points. """
        assert self.num_classes == 6
        num_points = xyz.shape[0]
        assert num_points > 0, "No point No prediction"
        assert xyz.shape[1] == 3
        # define label map
        label_map = {
            'Background': 0,
            'Pedestrian': 1,
            'Cyclist':3,
            'DontCare': 5
            }
        # by default, all points are assigned with background label 0.
        cls_labels = np.zeros((num_points, 1), dtype=np.int64)
        # 3d boxes for each point
        boxes_3d = np.zeros((num_points, 1, 7))
        valid_boxes = np.zeros((num_points, 1, 1), dtype=np.float32)
        # add label for each object
        for label in labels:
            obj_cls_string = label['name']
            obj_cls = label_map.get(obj_cls_string, 5)
            if obj_cls >= 1 and obj_cls <= 4:
                mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
                yaw = label['yaw']
                while yaw < -0.25*np.pi:
                    yaw += np.pi
                while yaw > 0.75*np.pi:
                    yaw -= np.pi
                if yaw < 0.25*np.pi:
                    # horizontal
                    cls_labels[mask, :] = obj_cls
                    boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
                        label['z3d'], label['length'], label['height'],
                        label['width'], yaw)
                    valid_boxes[mask, 0, :] = 1
                else:
                    # vertical
                    cls_labels[mask, :] = obj_cls+1
                    boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
                        label['z3d'], label['length'], label['height'],
                        label['width'], yaw)
                    valid_boxes[mask, 0, :] = 1
            else:
                if obj_cls_string != 'DontCare':
                    mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
                    cls_labels[mask, :] = obj_cls
                    valid_boxes[mask, 0, :] = 0.0

        return cls_labels, boxes_3d, valid_boxes, label_map

    def vis_points(self, cam_points_in_img_with_rgb,
        label_list=None, expend_factor=(1.0, 1.0, 1.0)):
        color_map = {
            'Pedestrian': ["DeepPink",(255,20,147)],
            'Person_sitting': ["DeepPink",(255,255,147)],
            'Car': ['Red', (255, 0, 0)],
            'Van': ['Red', (255, 255, 0)],
            'Cyclist': ["Salmon",(250,128,114)],
            'DontCare': ["Blue",(0,0,255)],
        }
        mesh_list = []
        if label_list is not None:
            for label in label_list:
                print(label['name'])
                point_mask = self.sel_points_in_box3d(label,
                    cam_points_in_img_with_rgb, expend_factor=expend_factor)
                color = np.array(
                    color_map.get(label['name'], ["Olive",(0,128,0)])[1])/255.0
                cam_points_in_img_with_rgb.attr[point_mask, 1:] = color
                mesh_list = mesh_list + self.draw_open3D_box(
                    label, expend_factor=expend_factor)
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(cam_points_in_img_with_rgb.xyz)
        pcd.colors = open3d.Vector3dVector(
            cam_points_in_img_with_rgb.attr[:,1:4])
        def custom_draw_geometry_load_option(geometry_list):
            vis = open3d.Visualizer()
            vis.create_window()
            for geometry in geometry_list:
                vis.add_geometry(geometry)
            ctr = vis.get_view_control()
            ctr.rotate(0.0, 3141.0, 0)
            vis.run()
            vis.destroy_window()
        custom_draw_geometry_load_option(mesh_list + [pcd])

    def vis_graph(self, points, A):
        """Visualize a 3D graph.

        Args:
            points: a Point objects containing vertices.
            A: the adjacency matrix.

        """
        xyz = points.xyz
        d_idx = np.tile(
            np.expand_dims(np.arange(A.shape[0]), 1), [1, A.shape[1]])
        d_idx = d_idx.reshape([-1, 1])
        s_idx = A.reshape([-1, 1])
        lines = np.hstack([d_idx, s_idx])
        line_set = open3d.LineSet()
        line_set.points = open3d.Vector3dVector(xyz)
        line_set.lines = open3d.Vector2iVector(lines)
        line_set.colors = open3d.Vector3dVector(
            [[1,0,0] for i in range(lines.shape[0])])
        open3d.draw_geometries([line_set])

    def vis_point_graph(self, cam_points_in_img_with_rgb, A, labels=None,
        edge_color=None):
        """Visualize a 3D graph with points.

        Args:
            points: a Point objects containing vertices.
            A: the adjacency matrix.

        """
        mesh_list = []
        if labels is not None:
            # if labels are provided, add 3D bounding boxes.
            for label in labels:
                # point_mask = kitti.sel_points_in_box3d(label,
                #     cam_points_in_img_with_rgb)
                # cam_points_in_img_with_rgb.attr[point_mask, :]
                # = (0, 255, 0, 0)
                mesh_list = mesh_list + self.draw_open3D_box(label)
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(cam_points_in_img_with_rgb.xyz)
        pcd.colors = open3d.Vector3dVector(
            cam_points_in_img_with_rgb.attr[:,1:4])
        # if downsampleing_size is not None:
        #     pcd = open3d.voxel_down_sample(pcd, voxel_size=downsampleing_size)
        xyz = cam_points_in_img_with_rgb.xyz
        colors = cam_points_in_img_with_rgb.attr[:,1:4]
        if edge_color is not None:
            colors[:,:] = edge_color
        # xyz = cam_points_in_img_with_rgb.xyz
        d_idx = np.tile(
            np.expand_dims(np.arange(A.shape[0]), 1), [1, A.shape[1]])
        colors = np.tile(
            np.expand_dims(colors[0:A.shape[0], :], 1),
            [1, A.shape[1], 1])
        colors = colors.reshape([-1, 3])
        d_idx = d_idx.reshape([-1, 1])
        s_idx = A.reshape([-1, 1])
        lines = np.hstack([d_idx, s_idx])
        line_set = open3d.LineSet()
        line_set.points = open3d.Vector3dVector(xyz)
        line_set.lines = open3d.Vector2iVector(lines)
        line_set.colors = open3d.Vector3dVector(
            colors)
        def custom_draw_geometry_load_option(geometry_list):
            vis = open3d.Visualizer()
            vis.create_window()
            for geometry in geometry_list:
                vis.add_geometry(geometry)
            ctr = vis.get_view_control()
            ctr.rotate(0.0, 3141.0, 0)
            vis.run()
            vis.destroy_window()
        custom_draw_geometry_load_option(mesh_list + [pcd] + [line_set])
