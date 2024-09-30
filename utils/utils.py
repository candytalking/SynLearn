""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf
import scipy.stats
from tensorflow.python.platform import flags
from sklearn.metrics import roc_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import pickle
from pathlib import Path
from tensorflow.python.client import device_lib
from keras.datasets.cifar import load_batch
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100

FLAGS = flags.FLAGS

# ============================= useful calculation =================================
def multi_arange(num, stop, start=0, step=1, axis=0):
    return np.repeat(np.expand_dims(np.arange(start, stop, step), axis=axis), num, axis=axis)

def couple_numpy_idx(idx1, idx2, fix_idx2=False):
    """
    couple two idx sets.
    :param idx1: 1-D numpy index array of size m.
    :param idx2: 1-D numpy index array of size n.
    :return: a list of m*n coupled indexes.
    """
    m, n = idx1.size, idx2.size
    if fix_idx2:
        assert n % m == 0
        m_rep = int(n/m)
        coupled_idx = np.reshape(np.stack([np.repeat(np.expand_dims(idx1, axis=-1), m_rep, axis=-1), idx2],
                                          axis=-1), [-1, 2])
    else:
        coupled_idx = np.asarray([([(idx1[i], idx2[j]) for j in range(n)]) for i in range(m)])
    return coupled_idx


def expand_and_repeat(inp, axis, n):
    return np.repeat(np.expand_dims(inp, axis), n, axis=axis)

# =========================== block dropout ============================


class BernoulliSampler:
    def __init__(self, thresh):
        self.thresh = thresh

    def sample(self, shape):
        p = np.random.uniform(low=0, high=1, size=shape)
        s = np.where(p < self.thresh, 1, 0)
        return s


class DropBlock:
    def __init__(self, block_size, drop_rate, img_size, train_flag):
#        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.train_flag = train_flag
        self.drop_rate = drop_rate
        assert img_size[0] == img_size[1]
        self.dim_feature = img_size[0]
        self.num_channel = img_size[2]

    def forward(self, batch_size, itr):
        # shape: (batch_size, height, width, channels)
#        if self.train_flag:
        gamma = self._get_gamma(itr)
        rnd_sampler = BernoulliSampler(gamma)
        mask = rnd_sampler.sample((batch_size, self.dim_feature-(self.block_size-1),
                                   self.dim_feature-(self.block_size-1), self.num_channel))
        block_mask = self._compute_block_mask(mask)
        countM = np.prod(block_mask.shape)
        count_ones = np.sum(block_mask)
        return block_mask * countM/count_ones
#            return block_mask * x * (countM / count_ones)
#        else:
#            return x

    def _get_gamma(self, itr):
        keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (itr+1), 1.0 - self.drop_rate)
        gamma = (1 - keep_rate) / self.block_size ** 2 * self.dim_feature**2/(self.dim_feature-self.block_size+1)**2
        return gamma

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1)/2)
        right_padding = int(self.block_size/2)
        batch_size, height, width, channels = mask.shape
        padded_mask = np.pad(mask, [[0, 0], [left_padding, right_padding], [left_padding, right_padding], [0, 0]])
        non_zero_idxs = np.stack(np.where(mask > 0)).T
        num_non_zero_blocks = non_zero_idxs.shape[0]

        offsets = np.stack([multi_arange(self.block_size, self.block_size, axis=-1).reshape([-1]),
                            multi_arange(self.block_size, self.block_size, axis=0).reshape([-1])]).T
        offsets = np.concatenate([np.zeros([self.block_size**2, 1]), offsets,
                                  np.zeros([self.block_size**2, 1])], axis=-1)
        if num_non_zero_blocks > 0:
#            non_zero_idxs = np.tile(non_zero_idxs, [self.block_size ** 2, 1])
            non_zero_idxs = np.reshape(np.tile(np.expand_dims(non_zero_idxs, axis=1), [1, self.block_size ** 2, 1]),
                                       [-1, non_zero_idxs.shape[-1]])
            offsets = np.tile(offsets, [num_non_zero_blocks, 1])  # .view(-1, 4)
            block_idxs = non_zero_idxs + offsets
            block_idxs = block_idxs.astype(dtype=np.int)
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1
        block_mask = 1 - padded_mask
        return block_mask


# ==================================================================================

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def invert_permutation(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_images(paths, labels, nb_samples=None, shuffle=True, mode=0, disc_in_flag=True):

    if nb_samples is not None:
        if mode == 0: ############# must be random sample!
            sampler = lambda x: random.sample(x, nb_samples)
        else:
            sampler = lambda x: [random.choice(x) for _ in range(nb_samples)]
    else:
        sampler = lambda x: x

    if disc_in_flag:
        d = 0
    else:
        d = 1

    if mode == 0:
        images = [(i, os.path.join(path, image)) for i, path in zip(labels, paths) for image in sampler(os.listdir(path))]
    else:
        images = [(d, os.path.join(path, image)) for path in paths for image in os.listdir(path)]
        images = sampler(images)

    if shuffle:
        random.shuffle(images)
    return images


def class_score(pred, label):
    return -tf.tensordot(pred, label, 1)


def check_classification_error(true_y, pred_y):
    accuracy = true_y[pred_y == true_y].size/len(true_y)
#    accuracy = accuracy_score(true_y, pred_y) * 100
#    print("Classification Accuracy : ", accuracy)
    return accuracy


def est_error(prediction_prob, target_label):
    target_prediction = np.argmax(prediction_prob, axis=0)
    sum_prob = np.sum(prediction_prob, axis=0)
    num_uncertain = sum_prob[sum_prob == 0].size
    target_label = target_label[sum_prob != 0]
    target_prediction = target_prediction[sum_prob != 0]

    prediction_accuracy = target_prediction[target_prediction == target_label].size/len(target_label)
    return num_uncertain, prediction_accuracy


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
#    h = np.std(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h, m-h, m+h


def get_best_thresh(in_pred, out_pred, pred, y):
    num_data = in_pred.shape[0]
    anchor = int(0.05 * num_data)
    idx = np.argpartition(out_pred, -anchor)[-anchor]
    best_thresh = out_pred[idx]
    xixi_in_pred = in_pred[in_pred > best_thresh]
    acc_in_ratio = xixi_in_pred.shape[0]/num_data

    return best_thresh, acc_in_ratio


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def pairwise_distances(mu, distance_type='kl divergence'):
    """Compute the 2D matrix of distances between model & task embeddings.
    Args:
        mu: tensor of shape [meta_batch_size, embed_dim]
        # logvar: tensor of shape [meta_batch_size, embed_dim]

        distance_type: string. Type of distances.

    Returns:
        pairwise_distances: tensor of shape [batch_size, batch_size]
    """

    if distance_type == 'kl divergence':
        (batch_size, dim_z) = mu.shape
        dup_mu1 = np.reshape(np.tile(np.expand_dims(mu, axis=1), [1, batch_size, 1]), [-1, dim_z])
        dup_mu2 = np.reshape(np.tile(np.expand_dims(mu, axis=0), [batch_size, 1, 1]), [-1, dim_z])

        dup_kl_divergence1 = 0.5 * np.sum((dup_mu1-dup_mu2) ** 2, axis=1)
        dist = np.reshape(dup_kl_divergence1, [batch_size, batch_size])
        return dist


def bipartite_pairwise_distances(mu_t, mu_c, distance_type='kl divergence'):
    """Compute the 2D matrix of distances between model & task embeddings.
    Args:
        mu: tensor of shape [meta_batch_size, embed_dim]
        # logvar: tensor of shape [meta_batch_size, embed_dim]

        distance_type: string. Type of distances.

    Returns:
        pairwise_distances: tensor of shape [batch_size, batch_size]
    """

    if distance_type == 'kl divergence':
        (batch_size_task, dim_z) = mu_t.shape
        (batch_size_spec, dim_z) = mu_c.shape
        dup_mu_t = np.reshape(np.tile(np.expand_dims(mu_t, axis=1), [1, batch_size_spec, 1]), [-1, dim_z])
        dup_mu_c = np.reshape(np.tile(np.expand_dims(mu_c, axis=0), [batch_size_task, 1, 1]), [-1, dim_z])
        dup_mu_t1 = np.reshape(np.tile(np.expand_dims(mu_t, axis=1), [1, batch_size_task, 1]), [-1, dim_z])
        dup_mu_t2 = np.reshape(np.tile(np.expand_dims(mu_t, axis=0), [batch_size_task, 1, 1]), [-1, dim_z])
        dup_mu_c1 = np.reshape(np.tile(np.expand_dims(mu_c, axis=1), [1, batch_size_spec, 1]), [-1, dim_z])
        dup_mu_c2 = np.reshape(np.tile(np.expand_dims(mu_c, axis=0), [batch_size_spec, 1, 1]), [-1, dim_z])

        dup_kl_divergence1 = np.sum((dup_mu_t-dup_mu_c) ** 2, axis=1)
        pairwise_dist = np.reshape(dup_kl_divergence1, [batch_size_task, batch_size_spec])
        dup_kl_divergence2 = np.sum((dup_mu_t1-dup_mu_t2) ** 2, axis=1)
        pairwise_dist2 = np.reshape(dup_kl_divergence2, [batch_size_task, batch_size_task])
        dup_kl_divergence3 = np.sum((dup_mu_c1-dup_mu_c2) ** 2, axis=1)

        dist = np.ones([batch_size_spec+batch_size_task, batch_size_spec+batch_size_task]) * 50
        for i in range(batch_size_task):
            for j in range(batch_size_spec):
                dist[batch_size_spec+i, j] = pairwise_dist[i, j]
                dist[j, batch_size_spec+i] = pairwise_dist[i, j]
            for j in range(batch_size_task):
                dist[batch_size_spec+i, batch_size_spec+j] = pairwise_dist2[i, j]
                dist[batch_size_spec+j, batch_size_spec+i] = pairwise_dist2[j, i]

        return dist


def plot(samples, type, num, img_shape):

    fig = plt.figure(figsize=(type, num))  #16, 40
    gs = gridspec.GridSpec(nrows=type, ncols=num)
    # gs.update(wspace=0.02, hspace=0.02)

    for i in range(type):
        for j, sample in enumerate(samples[i]):
            # ax = plt.subplot(gs[i*type+j])
            ax = plt.subplot(gs[i, j])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(img_shape), cmap='Greys_r')

    return fig


def summerize_result(method_names, exp_result, exp_name='not entered', plot_result=False):
    num_methods = len(method_names)
    mean_acc, std_acc = {}, {}
    for itr1 in range(num_methods):
        mean_acc[method_names[itr1]] = np.mean(exp_result[:, itr1])
        std_acc[method_names[itr1]] = np.std(exp_result[:, itr1])

    if plot_result:
        print('result for: ' + exp_name)
        for itr1 in range(num_methods):
            print('mean acc %s:, %f, std: %f\n'%(method_names[itr1],
                                                 mean_acc[method_names[itr1]], std_acc[method_names[itr1]]))
    return mean_acc, std_acc


def save_file(file_path, file_name, file_content, config_spec=True):
    if config_spec:
        file_path = os.path.join(FLAGS.path_prefix, file_path, FLAGS.source_dataset, FLAGS.config_name)
    else:
        file_path = os.path.join(FLAGS.path_prefix, file_path)
    if not os.path.isdir(file_path):
        Path(file_path).mkdir(parents=True, exist_ok=True)
        # os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(file_content, f)


def load_file(file_path, file_name, config_spec=True):
    path_prefix = FLAGS.path_prefix
    source_datasource = FLAGS.source_dataset
    config_name = FLAGS.config_name
    if config_spec:
        file_path = os.path.join(path_prefix, file_path, source_datasource, config_name)
    else:
        file_path = os.path.join(path_prefix, file_path)

    if not os.path.isdir(file_path):
        raise NameError('no such file path!')
    file_path = os.path.join(file_path, file_name)
    with open(file_path, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


def save_session(sess, file_path, file_name, seed_name):
    file_path = os.path.join(FLAGS.path_prefix, file_path, FLAGS.source_dataset,
                             FLAGS.config_name, file_name, seed_name)
    if not os.path.isdir(file_path):
        # Path(file_path).mkdir(parents=True, exist_ok=True)
        os.makedirs(file_path)
    file_name = os.path.join(file_path, '../model')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    saver.save(sess, file_name)


def load_session(sess, file_path, file_name, seed_name, exclude_scope='no scope'):
    global_vars = [var for var in tf.global_variables() if exclude_scope not in var.name]
    saver = tf.train.Saver(global_vars, max_to_keep=10)
    file_path = os.path.join(FLAGS.path_prefix, file_path, FLAGS.source_dataset, FLAGS.config_name, file_name, seed_name)
    print("Restoring model weights from " + file_path)
    model_file = tf.train.latest_checkpoint(file_path)
    saver.restore(sess, model_file)


def show_progress(curr, total, num_part=5, show_string='working'):
    if curr % int(total / num_part) == 0 and curr > 0:
        prog = int(curr * 100 / total)
        print(show_string + ' progress: %d%%' % prog)


def load_data(dataset, path=None, label_mode='fine', first=True):
    """Loads CIFAR100 dataset.

    # Arguments
        label_mode: one of "fine", "coarse".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    if dataset == 'cifar100':
        if label_mode not in ['fine', 'coarse']:
            raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

        fpath = os.path.join(path, dataset, 'train')
        x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

        fpath = os.path.join(path, dataset, 'test')
        x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
        return (x_train, y_train), (x_test, y_test)
    elif dataset == 'mini_imagenet':
        source_data = load_file(FLAGS.dataset_dir, 'mini_imagenet84.pkl', config_spec=False)
        return source_data
    elif dataset == 'mnist':
        ((train_x, train_y), (test_x, test_y)) = mnist.load_data()
        train_x = np.repeat(np.expand_dims(train_x, axis=-1), 3, axis=-1) / 255.0
        test_x = np.repeat(np.expand_dims(test_x, axis=-1), 3, axis=-1) / 255.0
        mean_train_x = np.mean(train_x, axis=0)
        train_y, test_y = train_y.astype(np.float32), test_y.astype(np.float32)
        train_x = np.stack([train_x[train_y == i][:5000] for i in range(10)]) # - mean_train_x
        test_x = np.stack([test_x[test_y == i][:800] for i in range(10)]) # - mean_train_x
        return train_x, test_x
    elif dataset == 'fashion_mnist':
        ((train_x, train_y), (test_x, test_y)) = fashion_mnist.load_data()
        train_x = np.repeat(np.expand_dims(train_x, axis=-1), 3, axis=-1) / 255.0
        test_x = np.repeat(np.expand_dims(test_x, axis=-1), 3, axis=-1) / 255.0
        mean_train_x = np.mean(train_x, axis=0)
        train_y, test_y = train_y.astype(np.float32), test_y.astype(np.float32)
        train_x = np.stack([train_x[train_y == i][:6000] for i in range(10)]) - mean_train_x
        test_x = np.stack([test_x[test_y == i][:1000] for i in range(10)]) - mean_train_x
        return train_x, test_x
    elif dataset == 'cifar10':
        ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
        train_x = train_x.astype(np.float32) / 255.0
        test_x = test_x.astype(np.float32) / 255.0
        mean_train_x = np.mean(train_x, axis=0)
        train_y, test_y = train_y.astype(np.float32).reshape([-1]), test_y.astype(np.float32).reshape([-1])
        train_x = np.stack([train_x[train_y == i] for i in range(10)]) # - mean_train_x
        test_x = np.stack([test_x[test_y == i] for i in range(10)]) # - mean_train_x
        return train_x, test_x
    elif dataset == 'dsprites':
        if first:
            img_size = [64, 64, 1]
            domain_info = [3, 6, 40]
            loadpath = FLAGS.path_prefix + FLAGS.dataset_dir + 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
            dataset_zip = np.load(loadpath)
            x = dataset_zip['imgs']
            y = dataset_zip['latents_values']
            domain_label = dataset_zip['latents_classes'][:, 1:4]
            num_train, num_test = 800, 224
            x = x.reshape([x.shape[0]] + img_size)
            train_x, train_y, test_x, test_y = [], [], [], []
            for i in range(domain_info[0]):
                for j in range(domain_info[1]):
                    for k in range(domain_info[2]):
                        a1, a2, a3 = np.where(domain_label[:, 0] == i, 1, 0), \
                                     np.where(domain_label[:, 1] == j, 1, 0), \
                                     np.where(domain_label[:, 2] == k, 1, 0)
                        idx = np.arange(x.shape[0])[(a1 + a2 + a3) == 3]
                        rand_idx = np.random.permutation(idx)
                        train_idx, test_idx = rand_idx[:num_train], rand_idx[num_train:]
                        train_x.append(x[train_idx].astype(np.float32))
                        test_x.append(x[test_idx].astype(np.float32))
                        train_y.append(y[train_idx, -2:])
                        test_y.append(y[test_idx, -2:])
            y = np.reshape(y, [-1, 1024, 6])
            model_specs = np.concatenate([np.expand_dims(y[i, 0, 1:4], axis=0) for i in range(y.shape[0])])
            train_x, train_y = np.stack(train_x), np.stack(train_y)
            test_x, test_y = np.stack(test_x), np.stack(test_y)
        else:
            train_x, train_y, test_x, test_y, model_specs \
                = load_file(FLAGS.datasave_dir, 'curr_data.pkl')
            train_x = train_x.astype(np.float32)
            test_x = test_x.astype(np.float32)
        return train_x, train_y, test_x, test_y, model_specs
    else:
        raise NameError('Unrecogized dataset!')