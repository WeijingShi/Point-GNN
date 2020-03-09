"""This file implements models for object detection. """

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.loss import focal_loss_sigmoid, focal_loss_softmax
from models import gnn

regularizer_dict = {
    'l2': slim.l2_regularizer,
    'l1': slim.l1_regularizer,
    'l1_l2': slim.l1_l2_regularizer,
}
keras_regularizer_dict = {
    'l2': tf.keras.regularizers.l1_l2,
    'l1': tf.keras.regularizers.l1,
    'l1_l2': tf.keras.regularizers.l1_l2,
}

class MultiLayerFastLocalGraphModelV2(object):
    """General multiple layer GNN model. The graphs are generated outside this
    model and then feeded into this model. This model applies a list of layers
    sequentially, while each layer chooses the graph they operate on.
    """

    def __init__(self, num_classes, box_encoding_len, regularizer_type=None,
        regularizer_kwargs=None, layer_configs=None, mode=None):
        """
        Args:
            num_classes: int, the number of object classes.
            box_encoding_len: int, the length of encoded bounding box.
            regularizer_type: string, one of 'l2','l1', 'l1_l2'.
            regularizer_kwargs: dict, keyword args to the regularizer.
            layer_config: A list of layer configurations.
            mode: string, one of 'train', 'eval', 'test'.
        """
        self.num_classes = num_classes
        self.box_encoding_len = box_encoding_len
        if regularizer_type is None:
            assert regularizer_kwargs is None, 'No regularizer no kwargs'
            self._regularizer = None
        else:
            self._regularizer = regularizer_dict[regularizer_type](
                **regularizer_kwargs)
        self._layer_configs = layer_configs
        self._default_layers_type = {
            'scatter_max_point_set_pooling': gnn.PointSetPooling(
                point_feature_fn=gnn.multi_layer_neural_network_fn,
                aggregation_fn=gnn.graph_scatter_max_fn,
                output_fn=gnn.multi_layer_neural_network_fn
                ),
            'scatter_max_graph_auto_center_net': gnn.GraphNetAutoCenter(
                edge_feature_fn=gnn.multi_layer_neural_network_fn,
                aggregation_fn=gnn.graph_scatter_max_fn,
                update_fn=gnn.multi_layer_neural_network_fn,
                auto_offset_fn=gnn.multi_layer_neural_network_fn,
                ),
            'classaware_predictor': gnn.ClassAwarePredictor(
                cls_fn=partial(gnn.multi_layer_fc_fn, Ks=(64,), num_layer=2),
                loc_fn=partial(gnn.multi_layer_fc_fn,
                    Ks=(64, 64,), num_layer=3)
                ),
            'classaware_predictor_128': gnn.ClassAwarePredictor(
                cls_fn=partial(gnn.multi_layer_fc_fn, Ks=(128,), num_layer=2),
                loc_fn=partial(gnn.multi_layer_fc_fn,
                    Ks=(128, 128), num_layer=3)
                ),
            'classaware_separated_predictor': gnn.ClassAwareSeparatedPredictor(
                cls_fn=partial(gnn.multi_layer_fc_fn, Ks=(64,), num_layer=2),
                loc_fn=partial(gnn.multi_layer_fc_fn,
                    Ks=(64, 64,), num_layer=3)
                ),
        }
        assert mode in ['train', 'eval', 'test'], 'Unsupported mode'
        self._mode = mode

    def predict(self,
        t_initial_vertex_features,
        t_vertex_coord_list,
        t_keypoint_indices_list,
        t_edges_list,
        is_training,
        ):
        """
        Predict the objects with initial vertex features and a list of graphs.
        The model applies layers sequentially while each layer choose the graph
        that they operates. For example, a layer can choose the i-th graph,
        which is composed of t_vertex_coord_list[i], t_edges_list[i], and
        optionally t_keypoint_indices_list[i]. It operates on the graph and
        output the updated vertex_features. Then the next layer takes the
        vertex_features and also choose a graph to further update the features.

        Args:
            t_initial_vertex_features: a [N, M] tensor, the initial features of
            N vertices. For example, the intensity value of lidar reflection.
            t_vertex_coord_list: a list of [Ni, 3] tensors, the coordinates of
            a list of graph vertices.
            t_keypoint_indices_list: a list of [Nj, 1] tensors or None. For a
            pooling layer, it outputs a reduced number of vertices, aka. the
            keypoints. t_keypoint_indices_list[i] is the indices of those
            keypoints. For a gnn layer, it does not reduce the vertex number,
            thus t_keypoint_indices_list[i] should be set to 'None'.
            t_edges_list: a list of [Ki, 2] tensors. t_edges_list[i] are edges
            for the i-th graph. it contains Ki pair of (source, destination)
            vertex indices.
            is_training: boolean, whether in training mode or not.
        returns: [N_output, num_classes] logits tensor for classification,
        [N_output, num_classes, box_encoding_len] box_encodings tensor for
        localization.
        """
        with slim.arg_scope([slim.batch_norm], is_training=is_training), \
            slim.arg_scope([slim.fully_connected],
                weights_regularizer=self._regularizer):
                tfeatures_list = []
                tfeatures = t_initial_vertex_features
                tfeatures_list.append(tfeatures)
                for idx in range(len(self._layer_configs)-1):
                    layer_config = self._layer_configs[idx]
                    layer_scope = layer_config['scope']
                    layer_type = layer_config['type']
                    layer_kwargs = layer_config['kwargs']
                    graph_level = layer_config['graph_level']
                    t_vertex_coordinates = t_vertex_coord_list[graph_level]
                    t_keypoint_indices = t_keypoint_indices_list[graph_level]
                    t_edges = t_edges_list[graph_level]
                    with tf.variable_scope(layer_scope, reuse=tf.AUTO_REUSE):
                        flgn = self._default_layers_type[layer_type]
                        print('@ level %d Graph, Add layer: %s, type: %s'%
                            (graph_level, layer_scope, layer_type))
                        if 'device' in layer_config:
                            with tf.device(layer_config['device']):
                                tfeatures = flgn.apply_regular(
                                    tfeatures,
                                    t_vertex_coordinates,
                                    t_keypoint_indices,
                                    t_edges,
                                    **layer_kwargs)
                        else:
                            tfeatures = flgn.apply_regular(
                                tfeatures,
                                t_vertex_coordinates,
                                t_keypoint_indices,
                                t_edges,
                                **layer_kwargs)

                        tfeatures_list.append(tfeatures)
                        print('Feature Dim:' + str(tfeatures.shape[-1]))
                predictor_config = self._layer_configs[-1]
                assert (predictor_config['type']=='classaware_predictor' or
                    predictor_config['type']=='classaware_predictor_128' or
                    predictor_config['type']=='classaware_separated_predictor')
                predictor = self._default_layers_type[predictor_config['type']]
                print('Final Feature Dim:'+str(tfeatures.shape[-1]))
                with tf.variable_scope(predictor_config['scope'],
                reuse=tf.AUTO_REUSE):
                    logits, box_encodings =  predictor.apply_regular(tfeatures,
                        num_classes=self.num_classes,
                        box_encoding_len=self.box_encoding_len,
                        **predictor_config['kwargs'])
                    print("Prediction %d classes" % self.num_classes)
        return logits, box_encodings

    def postprocess(self, logits):
        """Output predictions. """
        prob = tf.nn.softmax(logits, axis=-1)
        return prob

    def loss(self, logits, labels, pred_box, gt_box, valid_box,
             cls_loss_type='focal_sigmoid', cls_loss_kwargs={},
             loc_loss_type='huber_loss', loc_loss_kwargs={},
             loc_loss_weight=1.0,
             cls_loss_weight=1.0):
        """Output loss value.

        Args:
            logits: [N, num_classes] tensor. The classification logits from
            predict method.
            labels: [N] tensor. The one hot class labels.
            pred_box: [N, num_classes, box_encoding_len] tensor. The encoded
            bounding boxes from the predict method.
            gt_box: [N, 1, box_encoding_len] tensor. The ground truth encoded
            bounding boxes.
            valid_box: [N] tensor. An indicator of whether the vertex is from
            an object of interest (whether it has a valid bounding box).
            cls_loss_type: string, the type of classification loss function.
            cls_loss_kwargs: dict, keyword args to the classifcation loss.
            loc_loss_type: string, the type of localization loss function.
            loc_loss_kwargs: dict, keyword args to the localization loss.
            loc_loss_weight: scalar, weight on localization loss.
            cls_loss_weight: scalar, weight on the classifcation loss.
        returns: a dict of cls_loss, loc_loss, reg_loss, num_endpoint,
        num_valid_endpoint. num_endpoint is the number of output vertices.
        num_valid_endpoint is the number of output vertices that have a valid
        bounding box. Those numbers are useful for weighting during batching.
        """
        if isinstance(loc_loss_weight, dict):
            loc_loss_weight = loc_loss_weight[self._mode]
        if isinstance(cls_loss_weight, dict):
            cls_loss_weight = cls_loss_weight[self._mode]
        if isinstance(cls_loss_type, dict):
            cls_loss_type = cls_loss_type[self._mode]
            cls_loss_kwargs = cls_loss_kwargs[self._mode]
        if isinstance(loc_loss_type, dict):
            loc_loss_type = loc_loss_type[self._mode]
            loc_loss_kwargs = loc_loss_kwargs[self._mode]

        loss_dict = {}
        assert cls_loss_type in ['softmax', 'top_k_softmax',
            'focal_sigmoid', 'focal_softmax']
        if cls_loss_type == 'softmax':
            point_loss =tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.squeeze(labels, axis=1), logits=logits)
            num_endpoint = tf.shape(point_loss)[0]
        if cls_loss_type == 'focal_sigmoid':
            point_loss = focal_loss_sigmoid(labels, logits, **cls_loss_kwargs)
            num_endpoint = tf.shape(point_loss)[0]
        if cls_loss_type == 'focal_softmax':
            point_loss = focal_loss_softmax(labels, logits, **cls_loss_kwargs)
            num_endpoint = tf.shape(point_loss)[0]
        if cls_loss_type == 'top_k_softmax':
            point_loss =tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.squeeze(labels, axis=1), logits=logits)
            num_endpoint = tf.shape(point_loss)[0]
            k = cls_loss_kwargs['k']
            top_k_cls_loss, _ = tf.math.top_k(point_loss, k=k, sorted=True)
            point_loss = top_k_cls_loss
        cls_loss = cls_loss_weight* tf.reduce_mean(point_loss)
        batch_idx = tf.range(tf.shape(pred_box)[0])
        batch_idx = tf.expand_dims(batch_idx, axis=1)
        batch_idx = tf.concat([batch_idx, labels], axis=1)
        pred_box = tf.gather_nd(pred_box, batch_idx)
        pred_box = tf.expand_dims(pred_box, axis=1)
        #pred_box = tf.batch_gather(pred_box, labels)
        if loc_loss_type == 'huber_loss':
            all_loc_loss = loc_loss_weight*tf.losses.huber_loss(
                gt_box,
                pred_box,
                delta=1.0,
                weights=valid_box,
                reduction=tf.losses.Reduction.NONE)
            print(all_loc_loss.shape)
            all_loc_loss = tf.squeeze(all_loc_loss, axis=1)
            if 'classwise_loc_loss_weight' in loc_loss_kwargs and\
            self._mode == 'train':
                classwise_loc_loss_weight = loc_loss_kwargs[
                    'classwise_loc_loss_weight']
                classwise_loc_loss_weight = tf.gather(
                    classwise_loc_loss_weight, labels)
                all_loc_loss = all_loc_loss*classwise_loc_loss_weight
            num_valid_endpoint = tf.reduce_sum(valid_box)
            mean_loc_loss = tf.reduce_mean(all_loc_loss, axis=1)
            loc_loss = tf.div_no_nan(tf.reduce_sum(mean_loc_loss),
                num_valid_endpoint)
            classwise_loc_loss = []
            for class_idx in range(self.num_classes):
                class_mask = tf.where(tf.equal(tf.squeeze(labels, axis=1),
                    tf.constant(class_idx, tf.int32)))
                l = tf.reduce_sum(tf.gather(all_loc_loss, class_mask), axis=0)
                l = tf.squeeze(l, axis=0)
                is_nan_mask = tf.is_nan(l)
                l = tf.where(is_nan_mask, tf.zeros_like(l),l)
                classwise_loc_loss.append(l)
            loss_dict['classwise_loc_loss'] = classwise_loc_loss
        if loc_loss_type == 'top_k_huber_loss':
            k = loc_loss_kwargs['k']
            all_loc_loss = loc_loss_weight*tf.losses.huber_loss(
                gt_box,
                pred_box,
                delta=1.0,
                weights=valid_box,
                reduction=tf.losses.Reduction.NONE)
            all_loc_loss = tf.squeeze(all_loc_loss, axis=1)
            if 'classwise_loc_loss_weight' in loc_loss_kwargs \
                and self._mode == 'train':
                classwise_loc_loss_weight = loc_loss_kwargs[
                    'classwise_loc_loss_weight']
                classwise_loc_loss_weight = tf.gather(
                    classwise_loc_loss_weight, labels)
                all_loc_loss = all_loc_loss*classwise_loc_loss_weight
            loc_loss = tf.reduce_mean(all_loc_loss, axis=1)
            top_k_loc_loss, top_k_indices = tf.math.top_k(loc_loss,
                k=k, sorted=True)
            valid_box = tf.squeeze(valid_box, axis=2)
            valid_box = tf.squeeze(valid_box, axis=1)
            top_k_valid_box = tf.gather(valid_box, top_k_indices)
            num_valid_endpoint = tf.reduce_sum(top_k_valid_box)
            loc_loss = tf.div_no_nan(tf.reduce_sum(top_k_loc_loss),
                num_valid_endpoint)
            top_k_labels = tf.gather(labels, top_k_indices)
            all_top_k_loc_loss = tf.gather(all_loc_loss, top_k_indices)
            classwise_loc_loss = []
            for class_idx in range(self.num_classes):
                class_mask = tf.where(tf.equal(tf.squeeze(top_k_labels),
                    tf.constant(class_idx, tf.int32)))
                l = tf.reduce_sum(tf.gather(all_top_k_loc_loss, class_mask),
                    axis=0)
                l = tf.squeeze(l, axis=0)
                is_nan_mask = tf.is_nan(l)
                l = tf.where(is_nan_mask, tf.zeros_like(l),l)
                classwise_loc_loss.append(l)
            loss_dict['classwise_loc_loss'] = classwise_loc_loss

        with tf.control_dependencies([tf.assert_equal(tf.is_nan(loc_loss),
        False)]):
            reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        loss_dict.update({'cls_loss': cls_loss, 'loc_loss': loc_loss,
            'reg_loss': reg_loss, 'num_endpoint': num_endpoint,
            'num_valid_endpoint':num_valid_endpoint})
        return loss_dict

def get_model(model_name):
    """Fetch a model class."""
    model_map = {
        'multi_layer_fast_local_graph_model_v2':
            MultiLayerFastLocalGraphModelV2,
    }
    return model_map[model_name]
