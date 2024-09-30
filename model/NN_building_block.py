""" Building blocks for NN. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
from tensorflow.python.framework import ops
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10

FLAGS = flags.FLAGS


"""
network building blocks

"""

# --------------------------------------------- Keras blocks ------------------------------------------------


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

# --------------------------------------------- tf blocks ----------------------------------------------


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def res_block(inp, cweight, bweight, reuse, scope, train_flag=True, activation=tf.nn.relu, norm_type='batch_norm',
              max_pool_kernel_size=2, max_pool_pad='VALID', ave_pool=False, output_activate=True):
    """ a block for resnet """
    stride = 2
    # short cut
    outp1 = tf.nn.conv2d(inp, cweight[0], 1, 'SAME') + bweight[0]
    outp2 = inp
    for itr1 in range(1, 4):
        outp2 = tf.nn.conv2d(outp2, cweight[itr1], 1, 'SAME') + bweight[itr1]
        if itr1 < 3:
            outp2 = normalize(outp2, activation, reuse, scope+'/bn'+str(itr1-1), train_flag, norm_type=norm_type)
    outp = outp1 + outp2
    if output_activate:
        outp = normalize(outp, activation, reuse, scope+'/bn2', train_flag, norm_type=norm_type)
    outp = tf.nn.max_pool(outp, max_pool_kernel_size, stride, max_pool_pad)
    if ave_pool:
        outp = tf.reduce_mean(outp, [1, 2])
    return outp


def fc_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu,
             nob_flag=False, train_flag=True, norm_type='batch_norm'):
    """ Perform, fc, batch norm, activation """
    if nob_flag:
        fc_output = tf.matmul(inp, cweight)
    else:
        fc_output = tf.matmul(inp, cweight) + bweight
    normed = normalize(fc_output, activation, reuse, scope, train_flag, norm_type=norm_type)
    return normed


def normalize(inp, activation, reuse, scope, train_flag, norm_type='batch_norm', scale=True):
    if norm_type == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope,
                                    is_training=train_flag, scale=scale, epsilon=1e-5, decay=0.9)
    elif norm_type == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm_type == 'None':
        if activation is not None:
            if activation == tf.nn.relu:
                return activation(inp)
            elif activation == tf.nn.leaky_relu:
                return activation(inp, alpha=0.1)
        else:
            return inp
    else:
        raise NameError('unknown normalization type')


"""""""""""""""""""""""
Loss functions

"""""""""""""""""""""""


def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))


def min_pairdist(pred1, pred2):
    pred1 = tf.reshape(pred1, [-1])
    pred2 = tf.reshape(pred2, [-1])
    return tf.minimum(100.0, tf.reduce_mean(tf.square(pred1-pred2)))


def max_pairdist(pred1, pred2):
    pred1 = tf.reshape(pred1, [-1])
    pred2 = tf.reshape(pred2, [-1])
    loss = tf.minimum(-0.0, tf.maximum(FLAGS.loss_lowerbound, -tf.square(pred1-pred2)))
    return tf.reduce_mean(loss)


def single_min_pairdist(pred1, pred2):
    loss = tf.square(pred1-pred2)
    return tf.reduce_mean(loss, axis=1)


def single_max_pairdist(pred1, pred2):
    loss = -tf.square(pred1-pred2)
    return tf.reduce_mean(loss, axis=1)


def WARPloss(pred1, pred2, N, Y):
    K = tf.cast(tf.floor((Y-1)/N), tf.int32)
    alpha = tf.divide(tf.ones(K), tf.cast(tf.range(1, K+1), tf.float32))
    L = tf.reduce_sum(alpha)
    loss = L * tf.maximum(0.0, 0.1 - pred1 + pred2)

    return tf.reduce_mean(loss)


def xent_without_softmax(prob, label):
    # prob and label must have the same shape
    return -tf.reduce_sum(tf.multiply(label, tf.log(prob+1e-5)), axis=-1)

def xent_without_softmax_binary(prob, label):
    # prob and label must have the same shape
    label = tf.cast(label, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=prob)
    return loss

def sparse_xent(pred, label):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)

def xent(pred, label, update_batch_size):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives

    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / update_batch_size


def xent_single(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives

    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)


"""
for triplet loss calculation

"""


def pairwise_distances(task_mu, spec_mu, distance_type='inner product', stop=False):
    """Compute the 2D matrix of distances between model & task embeddings.
    Args:
        task_mu: tensor of shape [meta_batch_size_task, embed_dim]
        task_logvar: tensor of shape [meta_batch_size_task, embed_dim]
        spec_mu: tensor of shape [meta_batch_size_spec, embed_dim]
        spec_logvar: tensor of shape [meta_batch_size_spec, embed_dim]

        distance_type: string. Type of distances.

    Returns:
        pairwise_distances: tensor of shape [true_meta_batch_size_task, true_meta_batch_size_spec]
    """
    dim_z = tf.shape(task_mu)[-1]
    if distance_type == 'kl divergence':
        batch_size_task = tf.shape(task_mu)[0]
        batch_size_spec = tf.shape(spec_mu)[0]
        dup_task_mu = tf.reshape(tf.tile(tf.expand_dims(task_mu, axis=1), [1, batch_size_spec, 1]), [-1, dim_z])
        dup_spec_mu = tf.reshape(tf.tile(tf.expand_dims(spec_mu, axis=0), [batch_size_task, 1, 1]), [-1, dim_z])
        if not stop:
            dup_kl_divergence = 0.5 * tf.reduce_sum((dup_task_mu-dup_spec_mu) ** 2, axis=1)
        else:
            dup_kl_divergence = 0.5 * tf.reduce_sum((tf.stop_gradient(dup_task_mu)-dup_spec_mu) ** 2, axis=1)
        kl_divergence = tf.reshape(dup_kl_divergence, [batch_size_task, batch_size_spec])

        return kl_divergence
    else:
        task_embeddings = task_mu
        spec_embeddings = spec_mu
        dot_product = tf.matmul(task_embeddings, tf.transpose(spec_embeddings))

        if distance_type == 'inner product':
            norm_a = tf.sqrt(tf.reduce_sum(task_embeddings ** 2, axis=1))
            norm_b = tf.sqrt(tf.reduce_sum(spec_embeddings ** 2, axis=1))
            norm = tf.tensordot(norm_a, norm_b, axes=0)
            return -dot_product/(norm + 1e-5)
        elif distance_type == 'l2_nonsquared' or distance_type == 'l2_squared':

            square_norm_task = tf.diag_part(tf.matmul(task_embeddings, tf.transpose(task_embeddings)))
            square_norm_spec = tf.diag_part(tf.matmul(spec_embeddings, tf.transpose(spec_embeddings)))
            distances = tf.expand_dims(square_norm_task, 1) - 2.0 * dot_product + tf.expand_dims(square_norm_spec, 0)
            distances = tf.maximum(distances, 0.0)

            if distance_type == 'l2_nonsquared':
                mask = tf.to_float(tf.equal(distances, 0.0))
                distances = distances + mask * 1e-16
                distances = tf.sqrt(distances)
                distances = distances * (1.0 - mask)

            return distances

        else:
            raise NameError('unrecognized distance type')


def classwise_pairwise_distances(task_mu, distance_type='kl divergence'):
    """Compute the 2D matrix of distances between model & task embeddings.
    Args:
        task_mu: tensor of shape [meta_batch_size_task, embed_dim]
        distance_type: string. Type of distances.
    Returns:
        pairwise_distances: tensor of shape [meta_batch_size_task, meta_batch_size_task]
    """

    if distance_type == 'kl divergence':
        batch_size_task = tf.shape(task_mu)[0]
        dim_z = tf.shape(task_mu)[-1]
        dup_task_mu1 = tf.reshape(tf.tile(tf.expand_dims(task_mu, axis=1), [1, batch_size_task, 1]), [-1, dim_z])
        dup_task_mu2 = tf.reshape(tf.tile(tf.expand_dims(task_mu, axis=0), [batch_size_task, 1, 1]), [-1, dim_z])

        dup_kl_divergence = 0.5 * tf.reduce_sum((dup_task_mu1-dup_task_mu2) ** 2, axis=1)
        kl_divergence = tf.reshape(dup_kl_divergence, [batch_size_task, batch_size_task])

        return kl_divergence
    else:
        task_embeddings = task_mu

        dot_product = tf.matmul(task_embeddings, tf.transpose(task_embeddings))

        if distance_type == 'inner product':
            return dot_product
        elif distance_type == 'l2_nonsquared' or distance_type == 'l2_squared':

            square_norm_task = tf.diag_part(tf.matmul(task_embeddings, tf.transpose(task_embeddings)))
            distances = tf.expand_dims(square_norm_task, 1) - 2.0 * dot_product + tf.expand_dims(square_norm_task, 0)
            distances = tf.maximum(distances, 0.0)

            if distance_type == 'l2_nonsquared':
                mask = tf.to_float(tf.equal(distances, 0.0))
                distances = distances + mask * 1e-16

                distances = tf.sqrt(distances)

                distances = distances * (1.0 - mask)

            return distances

        else:
            raise NameError('unrecognized distance type')


def get_classwise_triplet_mask(input_mask, task_mask):
    """
    A triplet (i, j, k) is valid if:
        - input_mask(i, j) == 1 and input_mask(i, k) == 0 and task_mask(i, j) == 1
    Args:
        input_mask: [batch_size_task, batch_size_task] vector, whether the two tasks belong to the same class
        task_mask: [batch_size_task, batch_size_task] vector, whether the two tasks belong to the same party
    Returns:
        Return a [batch_size_task, batch_size_task, batch_size_task] 3D mask
         where mask[#task, #task, #task] is True iff the triplet (i, j, k) is valid.

    """

    batch_size_task = tf.shape(input_mask)[0]

    full_task_mask = tf.tile(tf.expand_dims(task_mask, 1), [1, batch_size_task, 1])
    label_not_equal_mask = tf.not_equal(tf.tile(tf.expand_dims(input_mask, 1), [1, batch_size_task, 1]),
                                        tf.tile(tf.expand_dims(input_mask, 2), [1, 1, batch_size_task]))

    positive_mask = tf.tile(tf.expand_dims(input_mask, axis=-1), [1, 1, batch_size_task])

    negative_mask = tf.not_equal(tf.tile(tf.expand_dims(input_mask, axis=1), [1, batch_size_task, 1]),
                                 -tf.ones(tf.shape(positive_mask)))

    mask = tf.logical_and(label_not_equal_mask,
                          tf.logical_and(tf.logical_and(tf.cast(positive_mask, tf.bool), negative_mask),
                          tf.cast(full_task_mask, tf.bool)))
    return mask


def get_triplet_mask(input_mask):
    """
    A triplet (i, j, k) is valid if:
        - mask(i, j) == 1 and mask(i, k) == 0
    Args:
        input_mask: [batch_size_task, batch_size_spec] 2-D tensor
    Returns:
        mask: a [batch_size_task, batch_size_spec, batch_size_spec] 3D mask
             where mask[#task, #model, #model] is True iff the triplet (a, p, n) is valid.
    """
    num_task, num_spec = tf.shape(input_mask)[0], tf.shape(input_mask)[1]
    pos_mask = expand_and_repeat(tf.reshape(input_mask, [-1]), 1, num_task*num_spec)
    neg_mask = expand_and_repeat(tf.reshape(input_mask, [-1]), 0, num_task*num_spec)
    effective_pos_mask = tf.greater_equal(pos_mask, tf.ones(tf.shape(pos_mask))*1)
    plain_pos_mask = tf.less(pos_mask, tf.ones(tf.shape(pos_mask))*1)
    mask = tf.greater(pos_mask, neg_mask)
    effective_mask = tf.logical_and(mask, effective_pos_mask)
    plain_mask = tf.logical_and(mask, plain_pos_mask)
    return effective_mask, plain_mask


def get_hard_easy_mask(input_mask):
    """
    A triplet (i, j, k) is valid if:
        - mask(i, j) == 1 and mask(i, k) == 0
    Args:
        input_mask: [batch_size_task, batch_size_spec] 2-D tensor
    Returns:
        hard_mask: a [batch_size_task, batch_size_spec, batch_size_spec] 3D mask
        where mask[#task, #model, #model] is True iff the triplet (a, p, n) is p is useful and n is ok.
        easy_mask: a [batch_size_task, batch_size_spec, batch_size_spec] 3D mask
        where mask[#task, #model, #model] is True iff the triplet (a, p, n) is p is useful and n is bad.
    """

    batch_size_task, batch_size_spec = tf.shape(input_mask)[0], tf.shape(input_mask)[1]

    useful_mask = tf.equal(tf.tile(tf.expand_dims(input_mask, axis=-1), [1, 1, batch_size_spec]),
                           tf.ones([batch_size_task, batch_size_spec, batch_size_spec]))
    weak_useful_mask = tf.equal(tf.tile(tf.expand_dims(input_mask, axis=-1), [1, 1, batch_size_spec]),
                                tf.zeros([batch_size_task, batch_size_spec, batch_size_spec]))
    ok_mask = tf.equal(tf.tile(tf.expand_dims(input_mask, axis=-2), [1, batch_size_spec, 1]),
                       tf.zeros([batch_size_task, batch_size_spec, batch_size_spec]))
    bad_mask = tf.equal(tf.tile(tf.expand_dims(input_mask, axis=-2), [1, batch_size_spec, 1]),
                        -tf.ones([batch_size_task, batch_size_spec, batch_size_spec]))
    useful_easy_mask = tf.logical_and(useful_mask, bad_mask)
    useful_hard_mask = tf.logical_and(useful_mask, ok_mask)
    ok_mask = tf.logical_and(weak_useful_mask, bad_mask)
    return useful_easy_mask, useful_hard_mask, ok_mask


def get_loss(mask, ori_triplet_loss):
    show_triplet_loss = tf.boolean_mask(tf.reshape(ori_triplet_loss, [-1]),
                                        tf.reshape(tf.cast(mask, tf.bool), [-1]))
    triplet_loss = tf.reshape(show_triplet_loss, [-1])
    num_triplets = tf.to_float(tf.shape(triplet_loss)[0])
    num_valid_triplets = tf.reduce_sum(tf.to_float(tf.greater(triplet_loss, 0.0)))
    learn_loss = tf.reduce_sum(triplet_loss)
    return learn_loss, num_triplets, num_valid_triplets


def batch_triplet_loss(input_mask, pairwise_dist=None, task_mu=None, spec_mu=None,
                       gamma=0.5, distance_type='inner product'):

    """Build the triplet loss over a batch of embeddings using the squared triplet loss.
    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        input_mask: the input mask
        task_mu: tensor of shape [batch_size_task, embed_dim]
        spec_mu: tensor of shape [batch_size_spec, embed_dim]
        distance_type: type of distance for the embedding space
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    num_task, num_spec = tf.shape(input_mask)[0], tf.shape(input_mask)[1]
    # Get the pairwise distance matrix
    if pairwise_dist is None:
        pairwise_dist = pairwise_distances(task_mu, spec_mu, distance_type=distance_type)
    else:
        pairwise_dist = pairwise_dist
    margin = gamma # - 0.1  # 40.0
    # ========================================= pairwise & triplet loss ============================================
    anchor_positive_dist = expand_and_repeat(tf.reshape(pairwise_dist, [-1]), 1, num_task*num_spec)
    anchor_negative_dist = expand_and_repeat(tf.reshape(pairwise_dist, [-1]), 0, num_task*num_spec)
    ori_triplet_loss = tf.maximum(anchor_positive_dist - anchor_negative_dist + margin, 0.0)
    # ============================================ masking and count ===============================================
    masks = get_triplet_mask(input_mask)
    loss, num_triplets, num_valid_triplets = [], [], []
    for mask in masks:
        curr_loss, curr_num_triplets, curr_num_valid_triplets = get_loss(mask, ori_triplet_loss)
        loss.append(curr_loss)
        num_triplets.append(curr_num_triplets)
        num_valid_triplets.append(curr_num_valid_triplets)
    show_loss = ori_triplet_loss
    return show_loss, masks, loss, num_triplets, num_valid_triplets, pairwise_dist


def batch_classwise_triplet_loss(input_mask, task_mask, task_mu, c_gamma=5.0, distance_type='inner product'):
    """Build the triplet loss over a batch of embeddings using the squared triplet loss.
    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        input_mask: the input mask, has shape [batch_size_task, batch_size_task]
        task_mask: the task mask, has shape
        task_mu: tensor of shape [batch_size_task, embed_dim]
        distance_type: type of distance for the embedding space
    Returns:
        central_loss: scalar tensor containing intra-class distance minimization loss
        loss: scalar tensor containing the triplet loss
        fraction_valid_triplets: fraction of useful triplet loss
        pairwise_dist: the pairwise distance matrix

    """
    # Get the pairwise distance matrix
    pairwise_dist = classwise_pairwise_distances(task_mu, distance_type=distance_type)
    # =========================================== central loss ==============================================
    margin = c_gamma # 40
    # ======================================= triplet loss ==============================================
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)  # shape [meta_batch_size_task, meta_batch_size_task, 1]
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)  # shape [meta_batch_size_task, 1, meta_batch_size_task]
    ori_triplet_loss = tf.maximum(anchor_positive_dist - anchor_negative_dist + margin, 0.0)
    # ========================================== mask and count =============================================
    mask = get_classwise_triplet_mask(input_mask, task_mask)
    learn_loss, num_triplets, num_valid_triplets = get_loss(mask, ori_triplet_loss)
    show_loss = ori_triplet_loss
    frac_valid_triplets = num_valid_triplets / (num_triplets + 1e-5)
    return show_loss, mask, learn_loss/(num_valid_triplets+1e-5), frac_valid_triplets, pairwise_dist


"""
Gradient Flipper. Useful for DANN.

"""


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()


# ---------------------------------------------- networks -----------------------------------------------
conv_initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
xavier_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
zero_initializer = tf.keras.initializers.Zeros(dtype=tf.float32)
normal_initializer = tf.initializers.truncated_normal(dtype=tf.float32)


class Conv:
    def __init__(self, scope, train_flag, dim_input, dim_output, kernel_size, stride, activation=tf.nn.relu,
                 padding='SAME', use_bias=True, norm_type='batch_norm', output_activate=True, reuse=False):
        self.train_flag = train_flag
        self.scope = scope
        self.reuse = reuse
        self.use_bias = use_bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.padding = padding
        self.norm_type = norm_type
        self.output_activate = output_activate
        self.dim_input, self.dim_output = dim_input, dim_output
        self.weights = self._get_weight()

    def _get_weight(self):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            weights = {}
            weights['w'] = tf.get_variable('w', [self.kernel_size, self.kernel_size, self.dim_input, self.dim_output],
                                           initializer=conv_initializer, dtype=tf.float32)
            if self.use_bias:
                weights['b'] = tf.get_variable('b', [self.dim_output], initializer=zero_initializer, dtype=tf.float32)
        return weights

    def forward(self, inp, reuse=False):
        with tf.variable_scope(self.scope):
            outp = tf.nn.conv2d(inp, self.weights['w'], self.stride, self.padding)
            if self.use_bias:
                outp += self.weights['b']
            outp = normalize(outp, self.activation, reuse, 'bn', self.train_flag, norm_type=self.norm_type)
        return outp

class FC:

    def __init__(self, scope, train_flag, dim_input, dim_output, use_bias=False, activation=None,
                 norm_type='None', reuse=False):
        self.train_flag = train_flag
        self.scope = scope
        self.reuse = reuse
        self.use_bias = use_bias
        self.activation = activation
        self.norm_type = norm_type
        self.dim_input, self.dim_output = dim_input, dim_output
        self.weights = self._get_weight()

    def _get_weight(self):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            weights = {}
            weights['w'] = tf.get_variable('w', [self.dim_input, self.dim_output],
                                           initializer=xavier_initializer, dtype=tf.float32)
            if self.use_bias:
                weights['b'] = tf.get_variable('b', [self.dim_output], initializer=zero_initializer, dtype=tf.float32)
        return weights

    def forward(self, inp, reuse=False):
        with tf.variable_scope(self.scope):
            outp = tf.matmul(inp, self.weights['w'])
            if self.use_bias:
                outp += self.weights['b']
            outp = normalize(outp, self.activation, reuse, 'bn', self.train_flag, norm_type=self.norm_type)
        return outp


class NormalResBlock:
    """ a res block for resnet <= 34 """

    def __init__(self, scope, train_flag, dim_input, dim_middle, dim_output, activation=tf.nn.relu,
                 norm_type='batch_norm', output_activate=True, init_stride=2, reuse=False):
        self.train_flag = train_flag
        self.scope = scope
        self.reuse = reuse
        self.activation = activation
        self.norm_type = norm_type
        self.output_activate = output_activate
        self.dim_input, self.dim_output = dim_input, dim_output
        self.init_stride = init_stride
        self.model = self._build_model()

    def _build_model(self):
        model = {}
        with tf.variable_scope(self.scope):
            if self.init_stride == 2:
                model['conv_short'] = Conv('conv_short', self.train_flag, self.dim_input, self.dim_output, kernel_size=1,
                                           stride=self.init_stride, reuse=self.reuse, activation=None, norm_type='None')
            model['conv0'] = Conv('conv0', self.train_flag, self.dim_input, self.dim_output,
                                  kernel_size=3, stride=self.init_stride, reuse=self.reuse)
            model['conv1'] = Conv('conv1', self.train_flag, self.dim_output, self.dim_output,
                                  kernel_size=3, stride=1, reuse=self.reuse, activation=None, norm_type='None')
        return model

    def forward(self, inp, dropblock=None, reuse=False):
        with tf.variable_scope(self.scope):
            if self.init_stride == 2:
                outp1 = self.model['conv_short'].forward(inp, reuse=reuse)
            else:
                outp1 = inp
            outp2 = self.model['conv0'].forward(inp, reuse=reuse)
            outp2 = self.model['conv1'].forward(outp2, reuse=reuse)
            outp = outp1 + outp2
            if self.output_activate:
                outp = normalize(outp, self.activation, reuse, 'bn_out', self.train_flag, norm_type=self.norm_type)
        return outp


class BottleneckResBlock:
    """ a res block for resnet >= 50 """

    def __init__(self, scope, train_flag, dim_input, dim_middle, dim_output, downsample=True,
                 activation=tf.nn.relu, norm_type='batch_norm', output_activate=True, init_stride=2, reuse=False):

        self.train_flag = train_flag
        self.scope = scope
        self.reuse = reuse
        self.activation = activation
        self.norm_type = norm_type
        self.output_activate = output_activate
        self.dim_input, self.dim_middle, self.dim_output = dim_input, dim_middle, dim_output
        self.init_stride = init_stride
        self.model = self._build_model()

    def _build_model(self):
        model = {}
        with tf.variable_scope(self.scope):
            model['conv_short'] = Conv('conv_short', self.train_flag, self.dim_input, self.dim_output, kernel_size=1,
                                       stride=self.init_stride, reuse=self.reuse, activation=None, norm_type='None')
            model['conv0'] = Conv('conv0', self.train_flag, self.dim_input, self.dim_middle,
                                  kernel_size=1, stride=1, reuse=self.reuse)
            model['conv1'] = Conv('conv1', self.train_flag, self.dim_middle, self.dim_middle,
                                  kernel_size=3, stride=self.init_stride, reuse=self.reuse)
            model['conv2'] = Conv('conv2', self.train_flag, self.dim_middle, self.dim_output,
                                  kernel_size=1, stride=1, reuse=self.reuse, activation=None, norm_type='None')
        return model

    def forward(self, inp, dropblock=None, reuse=False):
        with tf.variable_scope(self.scope):
            outp1 = self.model['conv_short'].forward(inp, reuse=reuse)
            outp2 = self.model['conv0'].forward(inp, reuse=reuse)
            outp2 = self.model['conv1'].forward(outp2, reuse=reuse)
            outp2 = self.model['conv2'].forward(outp2, reuse=reuse)
            outp = outp1 + outp2
            if self.output_activate:
                outp = normalize(outp, self.activation, reuse, 'bn_out', self.train_flag, norm_type=self.norm_type)
        return outp


class FewShotResBlock:
    """ a res block for resnet-12 """

    def __init__(self, scope, train_flag, dim_input, dim_middle, dim_output, downsample=True,
                 activation=tf.nn.leaky_relu, norm_type='batch_norm', output_activate=True,
                 init_stride=2, dropout=False, reuse=False):

        self.train_flag = train_flag
        self.scope = scope
        self.reuse = reuse
        self.activation = activation
        self.norm_type = norm_type
        self.output_activate = output_activate
        self.dim_input, self.dim_middle, self.dim_output = dim_input, dim_middle, dim_output
        self.init_stride = init_stride
        self.model = self._build_model()
        if self.train_flag and dropout:
            self.use_dropout = True
        else:
            self.use_dropout = False

    def _build_model(self):
        model = {}
        with tf.variable_scope(self.scope):
            model['conv_short'] = Conv('conv_short', self.train_flag, self.dim_input, self.dim_output, kernel_size=1,
                                       stride=1, reuse=self.reuse, activation=None, norm_type='batch_norm', use_bias=False)
            model['conv0'] = Conv('conv0', self.train_flag, self.dim_input, self.dim_middle, kernel_size=3, stride=1,
                                  reuse=self.reuse, activation=self.activation, norm_type=self.norm_type)
            model['conv1'] = Conv('conv1', self.train_flag, self.dim_middle, self.dim_middle, kernel_size=3, stride=1,
                                  reuse=self.reuse, activation=self.activation, norm_type=self.norm_type)
            model['conv2'] = Conv('conv2', self.train_flag, self.dim_middle, self.dim_output,
                                  kernel_size=3, stride=1, reuse=self.reuse, activation=None, norm_type='batch_norm')
        return model

    def forward(self, inp, dropblock=None, reuse=False):
        with tf.variable_scope(self.scope):
            outp1 = self.model['conv_short'].forward(inp, reuse=reuse)
            outp2 = self.model['conv0'].forward(inp, reuse=reuse)
            outp2 = self.model['conv1'].forward(outp2, reuse=reuse)
            outp2 = self.model['conv2'].forward(outp2, reuse=reuse)
            outp = outp1 + outp2
            if self.output_activate:
                outp = normalize(outp, self.activation, reuse, 'bn_out', self.train_flag, norm_type='None')
            outp = tf.nn.max_pool(outp, ksize=2, strides=2, padding='VALID')
            if self.use_dropout:
                if dropblock is not None:
                    outp = outp * dropblock
        return outp


class ResNetStage:
    """ a res stage """

    def __init__(self, scope, train_flag, stage_idx, num_blocks, init_channel=3, network_type=18,
                 activation=tf.nn.relu, norm_type='batch_norm', output_activate=True, reuse=False):

        self.train_flag = train_flag
        self.scope = scope
        self.reuse = reuse
        self.activation = activation
        self.norm_type = norm_type
        self.output_activate = output_activate
        self.stage_idx = stage_idx
        self.num_blocks = num_blocks
        self.network_type = network_type
        if self.network_type == 12:
            self.block = FewShotResBlock
            self.num_filters_input = [init_channel, 64, 160, 320]
            self.num_filters_middle = [64, 160, 320, 640]
            self.num_filters_output = [64, 160, 320, 640]

        elif self.network_type <= 34:
            self.block = NormalResBlock
            self.num_filters_input = [init_channel, 64, 128, 256]
            self.num_filters_middle = [64, 128, 256, 512]
            self.num_filters_output = self.num_filters_middle
        else:
            self.num_filters_input = [init_channel, 256, 512, 1024]
            self.num_filters_middle = [64, 128, 256, 512]
            self.num_filters_output = [256, 512, 1024, 2048]
            self.block = BottleneckResBlock
#        self.downsample = downsample
        self.model = self._build_model()

    def _build_model(self):
        model = []
        with tf.variable_scope(self.scope):
            for block_idx in range(self.num_blocks):
                if block_idx == 0:
                    dim_input = self.num_filters_input[self.stage_idx]
                    if self.stage_idx > 0 or self.network_type == 12:
                        init_stride = 2
                    else:
                        init_stride = 1
                else:
                    dim_input = self.num_filters_output[self.stage_idx]
                    init_stride = 1
                curr_block = self.block('block'+str(block_idx), self.train_flag, dim_input=dim_input,
                                        dim_middle=self.num_filters_middle[self.stage_idx],
                                        dim_output=self.num_filters_output[self.stage_idx],
                                        init_stride=init_stride, reuse=self.reuse)
                model.append(curr_block)
        return model

    def forward(self, inp, reuse=False, dropblock=None):
        outp = inp
        with tf.variable_scope(self.scope):
            for block_idx in range(self.num_blocks):
                outp = self.model[block_idx].forward(outp, dropblock=dropblock, reuse=reuse)
        return outp


class ResNetFeatureBackBone:
    """
    resnet feature backbone
    """
    def __init__(self, img_size, network_type=18, reuse=True, train_flag=False, scope='feature_backbone'):
        assert network_type in (10, 12, 18, 20, 34, 50, 101, 152)
        res_block_config = {'10': (1, 1, 1, 1), '12': (1, 1, 1, 1), '18': (2, 2, 2, 2), '20': (2, 2, 2),
                            '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3)}
        all_init_conv_config = {'large': (7, 2, True), 'middle': (3, 2, False), 'small': (3, 1, False)}

        self.network_type = network_type
        self.scope = scope
        self.reuse = reuse
        self.train_flag = train_flag
        self.img_size = img_size

        self.num_blocks = res_block_config[str(self.network_type)]
        self.model = {}
        with tf.variable_scope(self.scope, reuse=self.reuse):
            assert self.img_size[0] == self.img_size[1]
            if self.network_type != 12:
                if self.img_size[0] > 128:
                    self.init_conv_config = all_init_conv_config['large']
                elif self.img_size[0] > 84:
                    self.init_conv_config = all_init_conv_config['middle']
                else:
                    self.init_conv_config = all_init_conv_config['small']

                self.model['conv_init'] = Conv('conv_init', self.train_flag, self.img_size[2], 64,
                                               kernel_size=self.init_conv_config[0],
                                               stride=self.init_conv_config[1], reuse=self.reuse)
                init_channel = 64
            else:
                init_channel = self.img_size[2]

            for stage_idx in range(len(res_block_config[str(self.network_type)])):
                self.model['stage'+str(stage_idx)] = ResNetStage(scope='stage'+str(stage_idx),
                                                                 train_flag=self.train_flag,
                                                                 stage_idx=stage_idx, init_channel=init_channel,
                                                                 num_blocks=self.num_blocks[stage_idx],
                                                                 network_type=self.network_type, reuse=self.reuse)
            if self.network_type >= 50:
                self.model['final_fc'] = FC(scope='final_FC', train_flag=self.train_flag, dim_input=2048,
                                            dim_output=512, reuse=self.reuse)

    def forward(self, inp, dropblock=(None, None), reuse=False):
        inp = tf.reshape(inp, [-1]+self.img_size)
        with tf.variable_scope(self.scope):
            if self.network_type != 12:
                outp = self.model['conv_init'].forward(inp, reuse=reuse)
                if self.init_conv_config[2]:
                    outp = tf.nn.max_pool(outp, ksize=3, strides=2, padding='SAME')
            else:
                outp = inp
            for stage_idx in range(2):
                outp = self.model['stage'+str(stage_idx)].forward(outp, reuse=reuse)
            outp = self.model['stage2'].forward(outp, reuse=reuse)
            outp = self.model['stage3'].forward(outp, reuse=reuse)
            outp = tf.reduce_mean(outp, [1, 2])
            if self.network_type >= 50:
                outp = self.model['final_fc'].forward(outp, reuse=reuse)
        return outp


class FCFeatureBackBone:

    def __init__(self, input_length, output_length=512, num_fc_layer=0, scope='feature_backbone', reuse=True):

        self.input_length = input_length
        self.output_length = output_length
        self.dim_fc_layer = 2048
        self.num_fc_layer = num_fc_layer
        self.scope = scope

        self.weights = {}
        with tf.variable_scope(self.scope, reuse):  #, custom_getter=self.nontrainable_getter):

            for itr1 in range(self.num_fc_layer):
                if itr1:
                    dim_input = self.dim_fc_layer
                else:
                    dim_input = np.prod(input_length)
                self.weights['fc_w'+str(itr1)] = tf.get_variable('fc_w'+str(itr1),
                                                                 [dim_input, self.dim_fc_layer],
                                                                 initializer=xavier_initializer, dtype=tf.float32)
                self.weights['fc_b'+str(itr1)] = tf.get_variable('fc_b'+str(itr1), [self.dim_fc_layer],
                                                                 initializer=xavier_initializer, dtype=tf.float32)
            if self.num_fc_layer > 0:
                dim_input = self.dim_fc_layer
            else:
                dim_input = np.prod(self.input_length)
            self.weights['fc_w_out'] = tf.get_variable('fc_w_out', [dim_input, self.output_length],
                                                       initializer=xavier_initializer, dtype=tf.float32)
                                                       # initializer = normal_initializer, dtype = tf.float32)
            self.weights['fc_b_out'] = tf.get_variable('fc_b_out', [self.output_length],
                                                       initializer=xavier_initializer, dtype=tf.float32)

    def forward(self, inp, train_flag, scope, norm_type='batch_norm', reuse=True, output_activate=True):

#        shape1, shape2 = tf.shape(inp)[0], tf.shape(inp)[1]
        inp = tf.reshape(inp, [-1, self.input_length])
        with tf.variable_scope(self.scope, reuse=reuse):  #, custom_getter=self.nontrainable_getter):
            hidden = inp
            for itr1 in range(self.num_fc_layer):
                hidden = fc_block(hidden, self.weights['fc_w'+str(itr1)],
                                  self.weights['fc_b'+str(itr1)], reuse, scope + str(itr1),
                                  train_flag=train_flag, norm_type=norm_type)
            if output_activate:
                activation = tf.nn.relu
                output_norm_type = norm_type
            else:
                output_norm_type = 'None'
                activation = None
            hidden = fc_block(hidden, self.weights['fc_w_out'], self.weights['fc_b_out'], reuse, scope + 'out',
                              train_flag=train_flag, norm_type=output_norm_type, activation=activation)
            feature_output = tf.contrib.layers.flatten(hidden)
        return feature_output

def softmax_with_mask(inp, task_mask, model_mask=None, axis=-1):
    if model_mask is None:
        outp = tf.exp(tf.maximum(tf.minimum(inp, 20.0), -20.0))
        mask1 = tf.repeat(tf.expand_dims(task_mask, axis=-1), tf.shape(inp)[-1], axis=-1)
        mask2 = tf.repeat(tf.expand_dims(task_mask, axis=-2), tf.shape(inp)[-2], axis=-2)
    else:
        outp = tf.exp(tf.maximum(tf.minimum(inp*1.0, 20.0), -20.0)) # 3.0  # debug!!!!!!!!!!!!
        mask1 = tf.tile(tf.reshape(model_mask, [1, 1, tf.shape(model_mask)[0], tf.shape(model_mask)[1]]),
                       [tf.shape(inp)[0], tf.shape(inp)[1], 1, 1])
        mask2 = tf.tile(tf.reshape(task_mask, [tf.shape(task_mask)[0], tf.shape(task_mask)[1], 1, 1]),
                       [1, 1, tf.shape(inp)[-2], tf.shape(inp)[-1]])
    outp = tf.where(tf.equal(mask1, tf.ones(tf.shape(mask1), dtype=tf.int32)), outp, tf.zeros(tf.shape(outp)))
    outp = tf.where(tf.equal(mask2, tf.ones(tf.shape(mask2), dtype=tf.int32)), outp, tf.zeros(tf.shape(outp)))
    outp /= tf.reduce_sum(outp, axis=axis, keepdims=True) + 1e-5
    return outp


def expand_and_repeat(inp, axis, n):
    return tf.repeat(tf.expand_dims(inp, axis), n, axis=axis)


def layer_norm(x):
    mean_x = tf.reduce_mean(x, axis=-1, keepdims=True)
    var_x = tf.math.reduce_variance(x, axis=-1, keepdims=True)
    x = (x-mean_x)/(var_x+1e-5)
    return x


class ModelTransformer:
    def __init__(self, num_ele_q, num_ele, input_dim, input_q_dim=None, output_dim=512,
                 scope='model_transformer', reuse=True):
        self.input_dim = input_dim
        if input_q_dim is None:
            self.input_q_dim = input_dim
        else:
            self.input_q_dim = input_q_dim
        self.num_ele_q, self.num_ele, self.output_dim, self.scope = num_ele_q, num_ele, output_dim, scope
        self.weights = {}
        with tf.variable_scope(self.scope, reuse):
            self.weights['w_q0'] = tf.get_variable('w_q0', [self.input_q_dim, self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_k0'] = tf.get_variable('w_k0', [self.input_dim, self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_v0'] = tf.get_variable('w_v0', [self.input_dim, self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_fc1'] = tf.get_variable('w_fc1', [self.output_dim, self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['b_fc1'] = tf.get_variable('b_fc1', [self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_fc2'] = tf.get_variable('w_fc2', [self.output_dim, self.output_dim],
                                                    initializer=xavier_initializer, dtype=tf.float32)
            self.weights['b_fc2'] = tf.get_variable('b_fc2', [self.output_dim],
                                                    initializer=xavier_initializer, dtype=tf.float32)

    def sampling_attention_layer(self, z_c, p, z_d=None, normalize=False):
        """
        :param z_d: [None, out_dim] default rep. Default zero matrix.
        :param z_c: [None, out_dim] content rep.
        :param p:   [None] Bernoulli probability.
        :return: [None, out_dim] sampled rep.
        """
        if z_d is None:
            z_d = tf.zeros(tf.shape(z_c))
        if normalize:
            gamma = eps = p / (
                tf.repeat(tf.maximum(tf.reduce_sum(p, axis=-1, keepdims=True), 1.0), tf.shape(p)[-1], axis=-1))
        else:
            gamma = eps = p
        gam = expand_and_repeat(gamma, -1, tf.shape(z_c)[-1])
        sampled_z = tf.multiply(z_c, gam) + tf.multiply(z_d, 1 - gam)
        return sampled_z, gamma, eps

    def model_transformer_block(self, inp, inp_q, reuse, scope, input_mask=None, input_mask_2=None,
                                a_ref=None, train_flag=True, activation=tf.nn.relu):
        """
        a block for simple transformer, currently does not support stacking.

        inp: [num_inp, num_ele, inp_dim] tensor
        inp_q: [num_inp_q, num_ele_q, inp_q_dim] tensor
        input_mask: [num_ele, inp_dim] tensor
        w_q: [inp_q_dim, out_dim] tensor
        w_k: [inp_dim, out_dim] tensor
        w_v: [inp_dim, out_dim] tensor

        if self-attention, assume num_ele_q == num_ele and num_inp == num_inp_q

        """
        w_k, w_v, w_q = self.weights['w_k0'], self.weights['w_k0'], self.weights['w_k0']
        #        w_fc0, b_fc0 = self.weights['w_fc0'], self.weights['b_fc0']
        w_fc1, b_fc1 = self.weights['w_fc1'], self.weights['b_fc1']
        w_fc2, b_fc2 = self.weights['w_fc2'], self.weights['b_fc2']
        v = inp @ w_v
        v_c = v[:, 1:]
        if a_ref is None:
            k, q = inp @ w_k, inp_q @ w_q
            norm_q = tf.sqrt(tf.reduce_sum(q ** 2, axis=-1, keepdims=True))
            norm_k = tf.sqrt(tf.reduce_sum(k ** 2, axis=-1, keepdims=True))
            norms = tf.tensordot(norm_q, norm_k, axes=[[-1], [-1]])
            norm_factor = (1e-5+tf.tensordot(norm_q,norm_k,axes=[[-1],[-1]])
                           / tf.sqrt(tf.cast(self.output_dim,tf.float32)))
            e = tf.tensordot(q, k, axes=[[-1], [-1]]) / norm_factor
            model_mask = tf.concat([tf.ones([tf.shape(input_mask_2)[0], 1], dtype=tf.int32), input_mask_2], axis=-1)
            a = softmax_with_mask(e, input_mask, model_mask=model_mask, axis=-1)
            e_dummy = tf.repeat(tf.expand_dims(tf.expand_dims(e[:,:,:,0],axis=-1),axis=-1), self.num_ele-1,axis=-2)
            e_model = tf.expand_dims(e[:, :, :, 1:], axis=-1)
            e = tf.concat([e_dummy, e_model], axis=-1)
            att_p = a[:, :, :, 1:]
            c_unave, att_w, eps = self.sampling_attention_layer(v_c, att_p, normalize=True)
            bn_reuse = reuse
        else:
            c_unave, att_w, eps = self.sampling_attention_layer(v_c, a_ref)
            bn_reuse = True
        outp2 = c_unave
        outp2 = tf.reduce_sum(outp2, axis=-2)
        outp2 = tf.matmul(outp2, w_fc1) + b_fc1
        outp2 = normalize(outp2, activation, scope=scope + 'bn0', norm_type='batch_norm',
                          train_flag=train_flag, reuse=bn_reuse)
        outp2 = tf.reduce_mean(outp2, axis=1)
        outp2 = tf.matmul(outp2, w_fc2) + b_fc2
        if a_ref is None:
            return outp2, c_unave, a, att_w, e, v, eps, norms
        else:
            return outp2

    def forward(self, inp, inp_q, a_ref=None, input_mask=None, input_mask_2=None, reuse=True, train_flag=True):
        with tf.variable_scope(self.scope, reuse=reuse):
            c, c_unave, a, att_w, e, v, eps, norms \
                = self.model_transformer_block(inp, inp_q, reuse, 'block0', input_mask=input_mask,
                                               input_mask_2=input_mask_2, train_flag=train_flag)
            c_ref = None
            if a_ref is not None:
                c_ref = self.model_transformer_block(inp, inp_q, reuse, 'block0', a_ref=a_ref, train_flag=train_flag)
            return c, c_ref, c_unave, att_w, a, e, v, eps, norms

class TaskTransformer:
    def __init__(self, num_ele, input_dim, output_dim=512, scope='simple_transformer', reuse=True):
        self.num_ele, self.input_dim, self.output_dim, self.scope = num_ele, input_dim, output_dim, scope
        self.weights = {}
        with tf.variable_scope(self.scope, reuse):
            self.weights['w_q0'] = tf.get_variable('w_q0', [self.input_dim, self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_k0'] = tf.get_variable('w_k0', [self.input_dim, self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_v0'] = tf.get_variable('w_v0', [self.input_dim, self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_fc1'] = tf.get_variable('w_fc1', [self.output_dim, self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['b_fc1'] = tf.get_variable('b_fc1', [self.output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)

    def task_transformer_block(self, inp, input_mask, train_flag=False):
        """
        a block for simple transformer, currently does not support stacking.

        inp: [num_inp, num_ele, inp_dim] tensor
        inp_q: [num_inp_q, num_ele_q, inp_q_dim] tensor
        input_mask: [num_ele, inp_dim] tensor
        w_q: [inp_q_dim, out_dim] tensor
        w_k: [inp_dim, out_dim] tensor
        w_v: [inp_dim, out_dim] tensor

        if self-attention, assume num_ele_q == num_ele and num_inp == num_inp_q

        """
        w_k, w_v, w_q = self.weights['w_k0'], self.weights['w_k0'], self.weights['w_k0']
        w_fc1, b_fc1 = self.weights['w_fc1'], self.weights['b_fc1']
        outp1 = inp
        k, v, q = inp @ w_k, inp @ w_v, inp @ w_q
        e = tf.reduce_sum(tf.multiply(tf.expand_dims(q, axis=2), tf.expand_dims(k, axis=1)), axis=-1) \
            / tf.sqrt(tf.cast(self.output_dim, tf.float32))
        a = softmax_with_mask(e, input_mask, axis=-1)
        c = a @ v
        outp1 = outp1 + c  # [num_inp, num_ele, inp_dim]
        outp1 = layer_norm(outp1)
        if FLAGS.source_dataset != 'dsprites':
            outp2 = outp1
            outp2 = tf.reduce_mean(outp2, axis=-2)
            outp2 = tf.matmul(outp2, w_fc1) + b_fc1
        else:
            outp2 = outp1
        return outp1, outp2, a

    def forward(self, inp, input_mask=None, reuse=True, train_flag=True):
        with tf.variable_scope(self.scope, reuse=reuse):
            content_x, x, a = self.task_transformer_block(inp, input_mask, train_flag=train_flag)
            return content_x, x, a

