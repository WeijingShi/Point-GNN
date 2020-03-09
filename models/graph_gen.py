"""The file defines functions to generate graphs."""

import time
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d
import tensorflow as tf

def multi_layer_downsampling(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False,):
    """Downsample the points using base_voxel_size at different scales"""
    xmax, ymax, zmax = np.amax(points_xyz, axis=0)
    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    downsampled_list = [points_xyz]
    last_level = 0
    for level in levels:
        if np.isclose(last_level, level):
            downsampled_list.append(np.copy(downsampled_list[-1]))
        else:
            if add_rnd3d:
                xyz_idx = (points_xyz-xyz_offset+
                    base_voxel_size*level*np.random.random((1,3)))//\
                        (base_voxel_size*level)
                xyz_idx = xyz_idx.astype(np.int32)
                dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
                keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+\
                    xyz_idx[:, 2]*dim_y*dim_x
                sorted_order = np.argsort(keys)
                sorted_keys = keys[sorted_order]
                sorted_points_xyz = points_xyz[sorted_order]
                _, lens = np.unique(sorted_keys, return_counts=True)
                indices = np.hstack([[0], lens[:-1]]).cumsum()
                downsampled_xyz = np.add.reduceat(
                    sorted_points_xyz, indices, axis=0)/lens[:,np.newaxis]
                downsampled_list.append(np.array(downsampled_xyz))
            else:
                pcd = open3d.PointCloud()
                pcd.points = open3d.Vector3dVector(points_xyz)
                downsampled_xyz = np.asarray(open3d.voxel_down_sample(
                    pcd, voxel_size = base_voxel_size*level).points)
                downsampled_list.append(downsampled_xyz)
        last_level = level
    return downsampled_list

def multi_layer_downsampling_select(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
    """Downsample the points at different scales and match the downsampled
    points to original points by a nearest neighbor search.

    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.

    returns: vertex_coord_list, keypoint_indices_list
    """
    # Voxel downsampling
    vertex_coord_list = multi_layer_downsampling(
        points_xyz, base_voxel_size, levels=levels, add_rnd3d=add_rnd3d)
    num_levels = len(vertex_coord_list)
    assert num_levels == len(levels) + 1
    # Match downsampled vertices to original by a nearest neighbor search.
    keypoint_indices_list = []
    last_level = 0
    for i in range(1, num_levels):
        current_level = levels[i-1]
        base_points = vertex_coord_list[i-1]
        current_points = vertex_coord_list[i]
        if np.isclose(current_level, last_level):
            # same downsample scale (gnn layer),
            # just copy it, no need to search.
            vertex_coord_list[i] = base_points
            keypoint_indices_list.append(
                np.expand_dims(np.arange(base_points.shape[0]),axis=1))
        else:
            # different scale (pooling layer), search original points.
            nbrs = NearestNeighbors(n_neighbors=1,
                algorithm='kd_tree', n_jobs=1).fit(base_points)
            indices = nbrs.kneighbors(current_points, return_distance=False)
            vertex_coord_list[i] = base_points[indices[:, 0], :]
            keypoint_indices_list.append(indices)
        last_level = current_level
    return vertex_coord_list, keypoint_indices_list

def multi_layer_downsampling_random(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
    """Downsample the points at different scales by randomly select a point
    within a voxel cell.

    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.

    returns: vertex_coord_list, keypoint_indices_list
    """
    xmax, ymax, zmax = np.amax(points_xyz, axis=0)
    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    vertex_coord_list = [points_xyz]
    keypoint_indices_list = []
    last_level = 0
    for level in levels:
        last_points_xyz = vertex_coord_list[-1]
        if np.isclose(last_level, level):
            # same downsample scale (gnn layer), just copy it
            vertex_coord_list.append(np.copy(last_points_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.arange(len(last_points_xyz)), axis=1))
        else:
            if not add_rnd3d:
                xyz_idx = (last_points_xyz - xyz_offset) \
                    // (base_voxel_size*level)
            else:
                xyz_idx = (last_points_xyz - xyz_offset +
                    base_voxel_size*level*np.random.random((1,3))) \
                        // (base_voxel_size*level)
            xyz_idx = xyz_idx.astype(np.int32)
            dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
            keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+xyz_idx[:, 2]*dim_y*dim_x
            num_points = xyz_idx.shape[0]

            voxels_idx = {}
            for pidx in range(len(last_points_xyz)):
                key = keys[pidx]
                if key in voxels_idx:
                    voxels_idx[key].append(pidx)
                else:
                    voxels_idx[key] = [pidx]

            downsampled_xyz = []
            downsampled_xyz_idx = []
            for key in voxels_idx:
                center_idx = random.choice(voxels_idx[key])
                downsampled_xyz.append(last_points_xyz[center_idx])
                downsampled_xyz_idx.append(center_idx)
            vertex_coord_list.append(np.array(downsampled_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.array(downsampled_xyz_idx),axis=1))
        last_level = level

    return vertex_coord_list, keypoint_indices_list

def gen_multi_level_local_graph_v3(
    points_xyz, base_voxel_size, level_configs, add_rnd3d=False,
    downsample_method='center'):
    """Generating graphs at multiple scale. This function enforce output
    vertices of a graph matches the input vertices of next graph so that
    gnn layers can be applied sequentially.

    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.
        downsample_method: string, the name of downsampling method.
    returns: vertex_coord_list, keypoint_indices_list, edges_list
    """
    if isinstance(base_voxel_size, list):
        base_voxel_size = np.array(base_voxel_size)
    # Gather the downsample scale for each graph
    scales = [config['graph_scale'] for config in level_configs]
    # Generate vertex coordinates
    if downsample_method=='center':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_select(
                points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
    if downsample_method=='random':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_random(
                points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
    # Create edges
    edges_list = []
    for config in level_configs:
        graph_level = config['graph_level']
        gen_graph_fn = get_graph_generate_fn(config['graph_gen_method'])
        method_kwarg = config['graph_gen_kwargs']
        points_xyz = vertex_coord_list[graph_level]
        center_xyz = vertex_coord_list[graph_level+1]
        vertices = gen_graph_fn(points_xyz, center_xyz, **method_kwarg)
        edges_list.append(vertices)
    return vertex_coord_list, keypoint_indices_list, edges_list

def gen_disjointed_rnn_local_graph_v3(
    points_xyz, center_xyz, radius, num_neighbors,
    neighbors_downsample_method='random',
    scale=None):
    """Generate a local graph by radius neighbors.
    """
    if scale is not None:
        scale = np.array(scale)
        points_xyz = points_xyz/scale
        center_xyz = center_xyz/scale
    nbrs = NearestNeighbors(
        radius=radius,algorithm='ball_tree', n_jobs=1, ).fit(points_xyz)
    indices = nbrs.radius_neighbors(center_xyz, return_distance=False)
    if num_neighbors > 0:
        if neighbors_downsample_method == 'random':
            indices = [neighbors if neighbors.size <= num_neighbors else
                np.random.choice(neighbors, num_neighbors, replace=False)
                for neighbors in indices]
    vertices_v = np.concatenate(indices)
    vertices_i = np.concatenate(
        [i*np.ones(neighbors.size, dtype=np.int32)
            for i, neighbors in enumerate(indices)])
    vertices = np.array([vertices_v, vertices_i]).transpose()
    return vertices

def get_graph_generate_fn(method_name):
    method_map = {
        'disjointed_rnn_local_graph_v3':gen_disjointed_rnn_local_graph_v3,
        'multi_level_local_graph_v3': gen_multi_level_local_graph_v3,
    }
    return method_map[method_name]
