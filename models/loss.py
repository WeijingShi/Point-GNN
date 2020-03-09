"""Implements popular losses. """

import tensorflow as tf

def focal_loss_sigmoid(labels, logits, alpha=0.5, gamma=2):
    """
    github.com/tensorflow/models/blob/master/\
        research/object_detection/core/losses.py
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter.
      If positive samples number > negtive samples number,
      alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    prob = tf.sigmoid(logits)
    labels = tf.one_hot(labels,depth=prob.shape[1])
    labels = tf.squeeze(labels, axis=1)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    prob_t= (labels*prob) + (1-labels)*(1-prob)
    modulating = tf.pow(1-prob_t, gamma)
    alpha_weight = (labels*alpha) + (1-labels)*(1-alpha)
    focal_cross_entropy = (modulating * alpha_weight * cross_ent)
    return focal_cross_entropy

def focal_loss_softmax(labels, logits, gamma=2):
    """
    https://github.com/fudannlp16/focal-loss/blob/master/focal_loss.py
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits,axis=-1) # [batch_size,num_classes]
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.squeeze(labels, axis=1), logits=logits)
    cross_ent = tf.expand_dims(cross_ent, 1)
    labels = tf.cast(labels, tf.int32)
    L=((1.0-tf.batch_gather(y_pred, labels))**gamma)*cross_ent
    return L

def test_focal_loss():
    labels = tf.constant([[1], [2]], dtype=tf.int32)
    logits = tf.constant([[-100, -100, -100], [-20, -20, -40.0]],
        dtype=tf.float32)
    focal_sigmoid = focal_loss_sigmoid(labels, logits)
    focal_softmax = focal_loss_softmax(labels, logits)
    with tf.Session() as sess:
        print(sess.run(focal_sigmoid))
        print(sess.run(focal_softmax))
if __name__ == '__main__':
    test_focal_loss()
