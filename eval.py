"""This file defines the evaluation process of Point-GNN object detection."""

import os
import time
import argparse

import numpy as np
import tensorflow as tf

from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, \
                          get_encoding_len
from models import preprocess
from util.config_util import load_config, load_train_config
from util.summary_util import write_summary_scale

parser = argparse.ArgumentParser(description='Repeated evaluation of PointGNN.')
parser.add_argument('eval_config_path', type=str,
                   help='Path to train_config')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                   help='Path to KITTI dataset. Default="../dataset/kitti/"')
parser.add_argument('--dataset_split_file', type=str,
                    default='',
                   help='Path to KITTI dataset split file.'
                   'Default="DATASET_ROOT_DIR/3DOP_splits'
                   '/eval_config["eval_dataset"]"')
args = parser.parse_args()
eval_config = load_train_config(args.eval_config_path)
DATASET_DIR = args.dataset_root_dir
if args.dataset_split_file == '':
    DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
        './3DOP_splits/'+eval_config['eval_dataset'])
else:
    DATASET_SPLIT_FILE = args.dataset_split_file

config_path = os.path.join(eval_config['train_dir'], eval_config['config_path'])
while not os.path.isfile(config_path):
    print('No config file found in %s, waiting' %  config_path)
    time.sleep(eval_config['eval_every_second'])
config = load_config(config_path)
if 'eval' in config:
    config = config['eval']
dataset = KittiDataset(
    os.path.join(DATASET_DIR, 'image/training/image_2'),
    os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
    os.path.join(DATASET_DIR, 'calib/training/calib/'),
    os.path.join(DATASET_DIR, 'labels/training/label_2'),
    DATASET_SPLIT_FILE,
    num_classes=config['num_classes'])
NUM_CLASSES = dataset.num_classes

if 'NUM_TEST_SAMPLE' not in eval_config:
    NUM_TEST_SAMPLE = dataset.num_files
else:
    if eval_config['NUM_TEST_SAMPLE'] < 0:
        NUM_TEST_SAMPLE = dataset.num_files
    else:
        NUM_TEST_SAMPLE = eval_config['NUM_TEST_SAMPLE']

print(NUM_TEST_SAMPLE)
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])

aug_fn = preprocess.get_data_aug(eval_config['data_aug_configs'])
def fetch_data(frame_idx):
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx,
        config['downsample_by_voxel_size'])
    box_label_list = dataset.get_label(frame_idx)
    cam_rgb_points, box_label_list = aug_fn(cam_rgb_points, box_label_list)
    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(
        cam_rgb_points.xyz, **config['graph_gen_kwargs'])
    if config['input_features'] == 'irgb':
        input_v = cam_rgb_points.attr
    elif config['input_features'] == '0rgb':
        input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
            cam_rgb_points.attr[:, 1:]])
    elif config['input_features'] == '0000':
        input_v = np.zeros_like(cam_rgb_points.attr)
    elif config['input_features'] == 'i000':
        input_v = np.hstack([cam_rgb_points.attr[:, [0]],
            np.zeros((cam_rgb_points.attr.shape[0], 3))])
    elif config['input_features'] == 'i':
        input_v = cam_rgb_points.attr[:, [0]]
    elif config['input_features'] == '0':
        input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))
    last_layer_graph_level = config['model_kwargs'][
        'layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
    if config['label_method'] == 'yaw':
        (cls_labels, boxes_3d, valid_boxes, label_map) =\
            dataset.assign_classaware_label_to_points(box_label_list,
                last_layer_points_xyz, expend_factor=(1.0, 1.0, 1.0))
    if config['label_method'] == 'Car':
        cls_labels, boxes_3d, valid_boxes, label_map =\
            dataset.assign_classaware_car_label_to_points(box_label_list,
                last_layer_points_xyz,  expend_factor=(1.0, 1.0, 1.0))
    if config['label_method'] == 'Pedestrian_and_Cyclist':
        cls_labels, boxes_3d, valid_boxes, label_map =\
            dataset.assign_classaware_ped_and_cyc_label_to_points(
                box_label_list,
                last_layer_points_xyz,  expend_factor=(1.0, 1.0, 1.0))
    encoded_boxes = box_encoding_fn(
        cls_labels, last_layer_points_xyz, boxes_3d, label_map)
    # reducing memory usage by casting to 32bits
    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]
    cls_labels = cls_labels.astype(np.int32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)
    return(input_v, vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, encoded_boxes, valid_boxes)

# model =======================================================================
if config['input_features'] == 'irgb':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 3])
elif config['input_features'] == '0000':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'i000':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'i':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 1])
elif config['input_features'] == '0':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 1])

t_vertex_coord_list = [tf.placeholder(dtype=tf.float32, shape=[None, 3])]
for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
    t_vertex_coord_list.append(
        tf.placeholder(dtype=tf.float32, shape=[None, 3]))

t_edges_list = []
for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
    t_edges_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 2]))

t_keypoint_indices_list = []
for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
    t_keypoint_indices_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 1]))

t_class_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])
t_encoded_gt_boxes = tf.placeholder(dtype=tf.float32,
    shape=[None, 1, BOX_ENCODING_LEN])
t_valid_gt_boxes = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1])

t_is_training = tf.placeholder(dtype=tf.bool, shape=[])

model = get_model(config['model_name'])(num_classes=NUM_CLASSES,
    box_encoding_len=BOX_ENCODING_LEN, mode='eval', **config['model_kwargs'])
t_logits, t_pred_box = model.predict(
    t_initial_vertex_features, t_vertex_coord_list, t_keypoint_indices_list,
    t_edges_list,
    t_is_training)
t_probs = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)
t_loss_dict = model.loss(t_logits, t_class_labels, t_pred_box,
    t_encoded_gt_boxes, t_valid_gt_boxes, **config['loss'])
t_cls_loss = t_loss_dict['cls_loss']
t_loc_loss = t_loss_dict['loc_loss']
t_reg_loss = t_loss_dict['reg_loss']
t_classwise_loc_loss = t_loss_dict['classwise_loc_loss']
t_total_loss = t_cls_loss + t_loc_loss + t_reg_loss

t_classwise_loc_loss_update_ops = {}
for class_idx in range(NUM_CLASSES):
    for bi in range(BOX_ENCODING_LEN):
        classwise_loc_loss_ind =t_classwise_loc_loss[class_idx][bi]
        t_mean_loss, t_mean_loss_op = tf.metrics.mean(
            classwise_loc_loss_ind,
            name=('loc_loss_cls_%d_box_%d'%(class_idx, bi)))
        t_classwise_loc_loss_update_ops[
            ('loc_loss_cls_%d_box_%d'%(class_idx, bi))] = t_mean_loss_op
    classwise_loc_loss =t_classwise_loc_loss[class_idx]
    t_mean_loss, t_mean_loss_op = tf.metrics.mean(
        classwise_loc_loss,
        name=('loc_loss_cls_%d'%class_idx))
    t_classwise_loc_loss_update_ops[
        ('loc_loss_cls_%d'%class_idx)] = t_mean_loss_op

# metrics
t_recall_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_recall, t_recall_update_op = tf.metrics.recall(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        tf.equal(t_predictions, tf.constant(class_idx, tf.int32)),
        name=('recall_%d'%class_idx))
    t_recall_update_ops[('recall_%d'%class_idx)] = t_recall_update_op

t_precision_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_precision, t_precision_update_op = tf.metrics.precision(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        tf.equal(t_predictions, tf.constant(class_idx, tf.int32)),
        name=('precision_%d'%class_idx))
    t_precision_update_ops[('precision_%d'%class_idx)] = t_precision_update_op

t_mAP_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_mAP, t_mAP_update_op = tf.metrics.auc(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        t_probs[:, class_idx],
        num_thresholds=200,
        curve='PR',
        name=('mAP_%d'%class_idx),
        summation_method='careful_interpolation')
    t_mAP_update_ops[('mAP_%d'%class_idx)] = t_mAP_update_op

t_mean_cls_loss, t_mean_cls_loss_op = tf.metrics.mean(
    t_cls_loss,
    name='mean_cls_loss')
t_mean_loc_loss, t_mean_loc_loss_op = tf.metrics.mean(
    t_loc_loss,
    name='mean_loc_loss')
t_mean_reg_loss, t_mean_reg_loss_op = tf.metrics.mean(
    t_reg_loss,
    name='mean_reg_loss')
t_mean_total_loss, t_mean_total_loss_op = tf.metrics.mean(
    t_total_loss,
    name='mean_total_loss')

metrics_update_ops = {
    'cls_loss': t_mean_cls_loss_op,
    'loc_loss': t_mean_loc_loss_op,
    'reg_loss': t_mean_reg_loss_op,
    'total_loss': t_mean_total_loss_op,}
metrics_update_ops.update(t_recall_update_ops)
metrics_update_ops.update(t_precision_update_ops)
metrics_update_ops.update(t_mAP_update_ops)
metrics_update_ops.update(t_classwise_loc_loss_update_ops)

# optimizers ================================================================
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
fetches = {
    'step': global_step,
    'predictions': t_predictions,
    'pred_box': t_pred_box

}
fetches.update(metrics_update_ops)

# preprocessing data ========================================================
class DataProvider(object):
    """This class provides input data to training.
    It has option to load dataset in memory so that preprocessing does not
    repeat every time.
    Note, if there is randomness inside graph creation, samples should be
    reloaded for the randomness to take effect.
    """
    def __init__(self, fetch_data, load_dataset_to_mem=True,
        load_dataset_every_N_time=1, capacity=1):
        self._fetch_data = fetch_data
        self._loaded_data_dic = {}
        self._loaded_data_ctr_dic = {}
        self._load_dataset_to_mem = load_dataset_to_mem
        self._load_every_N_time = load_dataset_every_N_time
        self._capacity = capacity
    def provide(self, frame_idx):
        extend_frame_idx = frame_idx+np.random.choice(
            self._capacity)*NUM_TEST_SAMPLE
        if self._load_dataset_to_mem:
            if extend_frame_idx in self._loaded_data_ctr_dic:
                ctr = self._loaded_data_ctr_dic[extend_frame_idx]
                if ctr >= self._load_every_N_time:
                    del self._loaded_data_ctr_dic[extend_frame_idx]
                    del self._loaded_data_dic[extend_frame_idx]
            if frame_idx not in self._loaded_data_dic:
                self._loaded_data_dic[extend_frame_idx] = self._fetch_data(
                    frame_idx)
                self._loaded_data_ctr_dic[extend_frame_idx] = 0
            self._loaded_data_ctr_dic[extend_frame_idx] += 1
            return self._loaded_data_dic[extend_frame_idx]
        else:
            return self._fetch_data(frame_idx)

data_provider = DataProvider(fetch_data, load_dataset_to_mem=False)
saver = tf.train.Saver()
graph = tf.get_default_graph()
if eval_config['gpu_memusage'] < 0:
    gpu_options = tf.GPUOptions(allow_growth=True)
else:
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=eval_config['gpu_memusage'])

def eval_once(graph, gpu_options, saver, checkpoint_path):
    """Evaluate the model once. """
    with tf.Session(graph=graph,
        config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.variables_initializer(tf.global_variables()))
        sess.run(tf.variables_initializer(tf.local_variables()))
        print('Restore from checkpoint %s' % checkpoint_path)
        saver.restore(sess, checkpoint_path)
        previous_step = sess.run(global_step)
        print('Global step = %d' % previous_step)
        start_time = time.time()
        for frame_idx in range(NUM_TEST_SAMPLE):
            (input_v, vertex_coord_list, keypoint_indices_list, edges_list,
            cls_labels, encoded_boxes, valid_boxes)\
                = data_provider.provide(frame_idx)
            feed_dict = {
                t_initial_vertex_features: input_v,
                t_class_labels: cls_labels,
                t_encoded_gt_boxes: encoded_boxes,
                t_valid_gt_boxes: valid_boxes,
                t_is_training: config['eval_is_training'],
            }
            feed_dict.update(dict(zip(t_edges_list, edges_list)))
            feed_dict.update(
                dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
            feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))
            results = sess.run(fetches, feed_dict=feed_dict)

            if NUM_TEST_SAMPLE >= 10:
                if (frame_idx + 1) % (NUM_TEST_SAMPLE // 10) == 0:
                    print('@frame %d' % frame_idx)
                    print('cls:%f, loc:%f, reg:%f, loss: %f'
                        % (results['cls_loss'], results['loc_loss'],
                        results['reg_loss'], results['total_loss']))
                    for class_idx in range(NUM_CLASSES):
                        print('Class_%d: recall=%f, prec=%f, mAP=%f, loc=%f'
                            % (class_idx,
                            results['recall_%d'%class_idx],
                            results['precision_%d'%class_idx],
                            results['mAP_%d'%class_idx],
                            results['loc_loss_cls_%d'%class_idx]))
                        print('         '+\
                            'x=%.4f y=%.4f z=%.4f l=%.4f h=%.4f w=%.4f y=%.4f'
                            %(
                            results['loc_loss_cls_%d_box_%d'%(class_idx, 0)],
                            results['loc_loss_cls_%d_box_%d'%(class_idx, 1)],
                            results['loc_loss_cls_%d_box_%d'%(class_idx, 2)],
                            results['loc_loss_cls_%d_box_%d'%(class_idx, 3)],
                            results['loc_loss_cls_%d_box_%d'%(class_idx, 4)],
                            results['loc_loss_cls_%d_box_%d'%(class_idx, 5)],
                            results['loc_loss_cls_%d_box_%d'%(class_idx, 6)]),
                            )
        print('STEP: %d, time cost: %f'
            % (results['step'], time.time()-start_time))
        print('cls:%f, loc:%f, reg:%f, loss: %f'
            % (results['cls_loss'], results['loc_loss'], results['reg_loss'],
            results['total_loss']))
        for class_idx in range(NUM_CLASSES):
            print('Class_%d: recall=%f, prec=%f, mAP=%f, loc=%f'
                % (class_idx,
                results['recall_%d'%class_idx],
                results['precision_%d'%class_idx],
                results['mAP_%d'%class_idx],
                results['loc_loss_cls_%d'%class_idx]))
            print("         x=%.4f y=%.4f z=%.4f l=%.4f h=%.4f w=%.4f y=%.4f"
            %(
            results['loc_loss_cls_%d_box_%d'%(class_idx, 0)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 1)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 2)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 3)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 4)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 5)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 6)]),
            )
        # add summaries ====================================================
        for key in metrics_update_ops:
            write_summary_scale(key, results[key], results['step'],
                eval_config['eval_dir'])
        return results['step']

def eval_repeat():
    last_evaluated_model_path = None
    while True:
        previous_time = time.time()
        model_path = tf.train.latest_checkpoint(eval_config['train_dir'])
        if not model_path:
            print('No checkpoint found in %s, wait for %f seconds'
                % (eval_config['train_dir'], eval_config['eval_every_second']))
        if last_evaluated_model_path == model_path:
            print(
                'Checkpoint %s has been evaluated already, wait for %f seconds'
                % (model_path, eval_config['eval_every_second']))
        else:
            last_evaluated_model_path = model_path
            current_step = eval_once(graph, gpu_options, saver, model_path)
        if current_step >= eval_config['max_step']:
            break
        time_to_next_eval = (
            previous_time + eval_config['eval_every_second'] - time.time())
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)

if __name__ == '__main__':
    eval_repeat()
