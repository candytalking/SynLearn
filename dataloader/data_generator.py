""" Code for loading and processing data. """
import numpy as np
import os
import keras
from tensorflow.python.platform import flags
from utils.utils import couple_numpy_idx, expand_and_repeat
import pickle
from keras.datasets import mnist, fashion_mnist, cifar10
from PIL import Image
from utils.utils import save_file, load_file, load_data, multi_arange
from utils import global_objects
from data_loader.imagenet_data_loader import get_data_loader
from tqdm import tqdm


FLAGS = flags.FLAGS


class DataGenerator:
    """
    Data Generator capable of loading and processing data, as well as generating data batches.

    """

    def set_config(self):
        exp_config = global_objects.exp_config
        if exp_config.source_dataset != 'dsprites':
            file_name = exp_config.source_dataset + '.' + FLAGS.config_name + '.task_config.pkl'
            config_file = os.path.join(FLAGS.path_prefix, FLAGS.config_path, file_name)
            with open(config_file, 'rb') as f:
                (train_model_config, test_model_config) = pickle.load(f)
            self.train_model_config = train_model_config
            assert len(self.train_model_config) == exp_config.num_source_train_model
            self.test_model_config = test_model_config
            assert len(self.test_model_config) == exp_config.num_source_test_model
            haha = 1
        else:
            self.dim_output = self.dataset_info[self.dataset]['dim_output']
            file_path = FLAGS.config_path
            file_name = 'task_config.pkl'
            (self.train_domain_config, self.test_domain_config, self.train_domain_idx, self.test_domain_idx) \
                = load_file(file_path, file_name)
            self.num_train_model = len(self.train_domain_config)
            self.num_test_model = len(self.test_domain_config)
        haha = 1

    def load_data(self, mode='train'):
        exp_config = global_objects.exp_config
        if FLAGS.target_dataset in ('CUB', 'caltech'):
            if FLAGS.target_dataset == 'CUB':
                train_data_path = os.path.join(FLAGS.path_prefix, FLAGS.dataset_dir, 'CUB_200_2011/train')
                test_data_path = os.path.join(FLAGS.path_prefix, FLAGS.dataset_dir, 'CUB_200_2011/test')
                split_file = '/home/dingyx/exp/specification/save/dataset/CUB_200_2011/train_test_split.pkl'
                with open(split_file, 'rb') as f:
                    split = pickle.load(f)
                self.target_train_y = np.asarray([split[0][i][1] for i in range(len(split[0]))])
                self.num_target_train_y_per_class = np.asarray([sum(self.target_train_y == i) for i in range(200)])
            else:
                train_data_path = os.path.join(FLAGS.path_prefix, FLAGS.dataset_dir, 'caltech_256/train')
                test_data_path = os.path.join(FLAGS.path_prefix, FLAGS.dataset_dir, 'caltech_256/test')
                num_class_file = '/home/dingyx/exp/specification/save/dataset/caltech_256/num_classes.pkl'
                with open(num_class_file, 'rb') as f:
                    self.num_target_train_y_per_class = pickle.load(f)
            class_list = sorted(os.listdir(train_data_path))
            self.target_train_x = []
            print('loading training dataset %s' % FLAGS.target_dataset)
            for class_idx in tqdm(range(exp_config.num_target_dataset_class)):
                curr_class = class_list[class_idx]
                class_path = os.path.join(train_data_path, curr_class)
                imgs = np.stack([np.asarray(Image.open(os.path.join(class_path, img_path)).convert('RGB')
                                            .resize(
                    size=(exp_config.source_input_shape[0], exp_config.source_input_shape[1])))
                                .astype(dtype=np.float32) / 255.0
                                 for img_path in sorted(os.listdir(class_path))])
                self.target_train_x.append(imgs)
            class_list = sorted(os.listdir(test_data_path))
            self.target_test_x = []
            print('loading testing dataset %s' % FLAGS.target_dataset)
            for class_idx in tqdm(range(exp_config.num_target_dataset_class)):
                curr_class = class_list[class_idx]
                class_path = os.path.join(test_data_path, curr_class)
                imgs = np.stack([np.asarray(Image.open(os.path.join(class_path, img_path)).convert('RGB')
                                            .resize(
                    size=(exp_config.source_input_shape[0], exp_config.source_input_shape[1])))
                                .astype(dtype=np.float32) / 255.0
                                 for img_path in sorted(os.listdir(class_path))])
                self.target_test_x.append(imgs)
                haha = 1
            self.target_test_x = np.stack(self.target_test_x)

        elif FLAGS.target_dataset == 'cifar10':
            ((target_train_x, train_y), (target_test_x, test_y)) = cifar10.load_data()
            self.target_train_x, self.target_test_x = [], []
            print('loading testing dataset %s' % FLAGS.target_dataset)
            for i in tqdm(range(50000)):
                im = np.asarray(Image.fromarray(target_train_x[i])
                                .resize(size=(exp_config.source_input_shape[0],
                                              exp_config.source_input_shape[1]))).astype(dtype=np.float32)
                im /= 255.0
                self.target_train_x.append(im)
            if FLAGS.task != 'generate_feature':
                self.target_train_y = train_y.reshape([-1])
                self.target_train_x = np.stack(self.target_train_x, axis=0)
                self.target_train_x = np.concatenate([np.expand_dims(self.target_train_x[self.target_train_y==i],
                                                      axis=0) for i in range(10)], axis=0)
            for i in tqdm(range(10000)):
                im = np.asarray(Image.fromarray(target_test_x[i])
                                .resize(size=(exp_config.source_input_shape[0],
                                              exp_config.source_input_shape[1]))).astype(dtype=np.float32)
                im /= 255.0
                self.target_test_x.append(im)
            if FLAGS.task != 'generate_feature':
                self.target_test_y = test_y.reshape([-1])
                self.target_test_x = np.stack(self.target_test_x, axis=0)
                self.target_test_x = np.concatenate([np.expand_dims(self.target_test_x[self.target_test_y==i],
                                                      axis=0) for i in range(10)], axis=0)
            self.num_target_train_y_per_class = 500
            haha = 1
        elif FLAGS.target_dataset == 'fashion_mnist':
            ((target_train_x, train_y), (target_test_x, test_y)) = fashion_mnist.load_data()
            #            target_train_x = target_train_x.astype(np.float32) / 255.0
            self.target_train_x, self.target_test_x = [], []
            print('loading testing dataset %s' % FLAGS.target_dataset)
            for i in tqdm(range(60000)):
                im = np.asarray(Image.fromarray(target_train_x[i]).convert('RGB')
                                .resize(size=(exp_config.source_input_shape[0],
                                              exp_config.source_input_shape[1]))).astype(dtype=np.float32)
                im /= 255.0
                self.target_train_x.append(im)
            if FLAGS.task != 'generate_feature':
                self.target_train_y = train_y.reshape([-1])
                self.target_train_x = np.stack(self.target_train_x, axis=0)
                self.target_train_x = np.concatenate([np.expand_dims(self.target_train_x[self.target_train_y==0],
                                                      axis=0) for i in range(10)], axis=0)
            for i in tqdm(range(10000)):
                im = np.asarray(Image.fromarray(target_test_x[i]).convert('RGB')
                                .resize(size=(exp_config.source_input_shape[0],
                                              exp_config.source_input_shape[1]))).astype(dtype=np.float32)
                im /= 255.0
                self.target_test_x.append(im)
            if FLAGS.task != 'generate_feature':
                self.target_test_y = test_y.reshape([-1])
                self.target_test_x = np.stack(self.target_test_x, axis=0)
                self.target_test_x = np.concatenate([np.expand_dims(self.target_test_x[self.target_test_y==i],
                                                      axis=0) for i in range(10)], axis=0)
            self.num_target_train_y_per_class = 600
            haha = 1
        else:
            raise NameError('unrecognized dataset')


    def generate_source_dataloader(self, task_idx):
        exp_config = global_objects.exp_config
        if exp_config.source_dataset == 'imagenet':
            model_config = (self.train_model_config+self.test_model_config)
            train_iterator, test_iterator, num_train_data, num_test_data = get_data_loader(model_config[task_idx])
            return train_iterator, test_iterator, num_train_data, num_test_data


    def load_feature(self, mode='test'):
        gap = 25
        if mode == 'train':
            start, end = 0, 100
        elif mode == 'test':
            start, end = 100, 200
        else:
            raise NameError('unrecognized mode')
        self.target_test_pred, self.target_test_feature = [], []
        for i in range(start, end, gap):
            target_test_feature = load_file(FLAGS.datasave_dir, 'target_test_feature' + str(i)
                                             + '.' + FLAGS.target_dataset + '.pkl')
            self.target_test_feature.extend(target_test_feature)
            target_test_pred = load_file(FLAGS.datasave_dir, 'target_test_pred' + str(i)
                                             + '.' + FLAGS.target_dataset + '.pkl')
            self.target_test_pred.extend(target_test_pred)
        self.target_test_feature = np.stack(self.target_test_feature)
        self.target_test_pred = np.stack(self.target_test_pred)
        if FLAGS.target_dataset == 'CUB':
            self.target_test_feature = self.target_test_feature.reshape([100, 200, 30, -1])
            self.target_test_pred = self.target_test_pred.reshape([100, 200, 30, -1])
            split_file = '/home/dingyx/exp/specification/save/dataset/CUB_200_2011/train_test_split.pkl'
            with open(split_file, 'rb') as f:
                split = pickle.load(f)
            self.target_test_y = np.concatenate([np.ones(30)*i for i in range(200)])
        elif FLAGS.target_dataset == 'caltech':
            self.target_test_feature = self.target_test_feature.reshape([100, 256, 30, -1])
            self.target_test_pred = self.target_test_pred.reshape([100, 256, 30, -1])
            self.target_test_y = np.concatenate([np.ones(30)*i for i in range(256)])
            haha = 1
        elif FLAGS.target_dataset == 'cifar10':
            ((_, _), (_, target_test_y)) = cifar10.load_data()
            self.target_test_y = target_test_y.reshape([-1])
            self.target_test_feature = np.concatenate([np.expand_dims(self.target_test_feature[:, self.target_test_y == i],
                                                                      axis=1) for i in range(10)], axis=1)
            self.target_test_pred = np.concatenate([np.expand_dims(self.target_test_pred[:, self.target_test_y == i],
                                                                      axis=1) for i in range(10)], axis=1)
        elif FLAGS.target_dataset == 'fashion_mnist':
            ((_, _), (_, target_test_y)) = fashion_mnist.load_data()
            self.target_test_y = target_test_y.reshape([-1])
            self.target_test_feature = np.concatenate([np.expand_dims(self.target_test_feature[:, self.target_test_y == i],
                                                                      axis=1) for i in range(10)], axis=1)
            self.target_test_pred = np.concatenate([np.expand_dims(self.target_test_pred[:, self.target_test_y == i],
                                                                   axis=1) for i in range(10)], axis=1)
        else:
            raise NameError('unrecognized target dataset')

    def load_pred(self, mode='train'):
        gap = 25
        if mode == 'train':
            start, end = 0, 100
        elif mode == 'test':
            start, end = 100, 200
        else:
            raise NameError('unrecognized mode')
        self.target_train_pred = []
        for i in range(start, end, gap):
            target_train_pred = load_file(FLAGS.datasave_dir, 'target_train_pred' + str(i)
                                          + '.' + FLAGS.target_dataset + '.pkl')
            self.target_train_pred.extend(target_train_pred)
        self.target_train_pred = np.stack(self.target_train_pred)
        if FLAGS.target_dataset == 'CUB':
            split_file = '/home/dingyx/exp/specification/save/dataset/CUB_200_2011/train_test_split.pkl'
            with open(split_file, 'rb') as f:
                split = pickle.load(f)
            self.target_train_y = np.asarray([split[0][i][1] for i in range(len(split[0]))])
            num_train_y = [np.sum(self.target_train_y == i) for i in range(200)]
        elif FLAGS.target_dataset == 'caltech':
            num_class_file = '/home/dingyx/exp/specification/save/dataset/caltech_256/num_classes.pkl'
            with open(num_class_file, 'rb') as f:
                self.num_class = pickle.load(f)
            self.target_train_y = np.concatenate([np.ones(self.num_class[i]) * i for i in range(256)])
            haha = 1
        elif FLAGS.target_dataset == 'cifar10':
            ((_, target_train_y), (_, _)) = cifar10.load_data()
            self.target_train_y = target_train_y.reshape([-1])
            self.target_train_pred = np.concatenate([np.expand_dims(self.target_train_pred[:, self.target_train_y==i],
                                                                    axis=1) for i in range(10)], axis=1)
            haha = 1
        elif FLAGS.target_dataset == 'fashion_mnist':
            ((_, target_train_y), (_, _)) = fashion_mnist.load_data()
            self.target_train_y = target_train_y.reshape([-1])
            self.target_train_pred = np.concatenate([np.expand_dims(self.target_train_pred[:, self.target_train_y==i],
                                                                    axis=1) for i in range(10)], axis=1)
        else:
            raise NameError('unrecognized target dataset')


    def load_source_model_specification(self, mode):
        model_specs = []
        if mode == 'train':
            load_range = range(0, 100, 50)
        else:
            load_range = range(100, 200, 50)
        for idx in load_range:
            curr_model_specs = load_file(FLAGS.datasave_dir, 'train_pred' + str(idx) + '.pkl')
            model_specs += curr_model_specs
        model_specs = np.stack(model_specs)
        self.model_specs = np.swapaxes(np.mean(model_specs, axis=-2), -1, -2)
        self.model_specs[self.model_specs < 0.2] = 0
        self.model_specs[self.model_specs > 0.8] = 1
        mask = np.ones(self.model_specs.shape)
        mask[self.model_specs == 0] = 0
        mask[self.model_specs == 1] = 0
        self.model_specs[mask == 1] = 0.5
        self.model_specs = 2 * self.model_specs - 1
        print('model specification has been loaded')


    def recover_env(self):
        self.dataset = FLAGS.datasource
        if FLAGS.visualize or FLAGS.datasource == 'dsprites':
            self.target_dataset = self.dataset
        else:
            self.target_dataset = FLAGS.target_datasource
        self.config_name = FLAGS.config_name
        self.meta_batch_size_task = FLAGS.meta_batch_size_task
        self.path_prefix = FLAGS.path_prefix
        self.config_path = FLAGS.config_path
        self.num_min_task_classes = FLAGS.num_min_task_classes
        self.num_max_task_classes = FLAGS.num_max_task_classes
        self.model_pool_train_batch_size = FLAGS.model_pool_train_batch_size
        self.train_n_shot = FLAGS.train_n_shot
        self.test_n_shot = FLAGS.test_n_shot
        self.dataset_dir = FLAGS.dataset_dir
        self.datasave_dir = FLAGS.datasave_dir
        self.num_pairwise_batch = FLAGS.num_pairwise_batch
#        self.use_part_train_data = FLAGS.use_part_train_data
        self.set_config()
        if self.dataset != 'dsprites':
            if self.dataset == 'mnist':
                self.train_x, self.test_x = load_data('mnist')
            elif self.dataset == 'fashion_mnist':
                source_data = fashion_mnist.load_data()
            elif self.dataset == 'cifar10':
                self.train_x, self.test_x = load_data('cifar10')
            if not FLAGS.visualize:
                if FLAGS.target_datasource == 'CUB':
                    self.target_train_x, self.target_train_y, self.target_test_x, self.target_test_y \
                        = load_file(FLAGS.dataset_dir, 'CUB_84.pkl', config_spec=False)
                    mean_target_train_x = np.mean(self.target_train_x, axis=0)
                    self.target_train_x -= mean_target_train_x
                    self.target_test_x -= mean_target_train_x
                elif FLAGS.target_datasource == 'caltech':
                    self.target_train_x, self.target_train_y, self.target_test_x, self.target_test_y \
                        = load_file(FLAGS.dataset_dir, 'caltech_84_parted.pkl', config_spec=False)
                    mean_target_train_x = np.mean(np.concatenate(self.target_train_x), axis=0)
                    self.target_train_x = [self.target_train_x[i] - mean_target_train_x
                                           for i in range(self.num_target_dataset_classes)]
                    self.target_test_x = [self.target_test_x[i] - mean_target_train_x
                                          for i in range(self.num_target_dataset_classes)]
                else:
                    self.train_x, self.test_x, self.target_train_x, self.target_test_x, \
                    _, _, _, _ = load_file(self.datasave_dir, 'curr_data.pkl')
        else:
            self.train_x, self.train_y, self.test_x, self.test_y, self.model_specs = load_data('dsprites', first=False)

        if FLAGS.datasource == 'dsprites' or FLAGS.visualize:
            self.num_target_train_data_per_class = self.num_train_data_per_class = self.train_x.shape[1]
            self.num_target_test_data_per_class = self.num_test_data_per_class = self.test_x.shape[1]
        else:
            self.num_source_train_data_per_class = self.train_x.shape[1]
            self.num_source_test_data_per_class = self.test_x.shape[1]
            if self.target_dataset in self.available_target_datasets:
                self.num_target_train_data_per_class \
                    = np.asarray([self.target_train_y[self.target_train_y==i].shape[0]
                                  for i in range(self.num_target_dataset_classes)])
                self.num_target_test_data_per_class \
                    = np.asarray([self.target_test_y[self.target_test_y==i].shape[0]
                                  for i in range(self.num_target_dataset_classes)])
                haha = 1
            else:
                self.num_target_train_data_per_class = self.target_train_x.shape[1]
                self.num_target_test_data_per_class = self.target_test_x.shape[1]

        if FLAGS.task == 'generate_feature':
            if not FLAGS.visualize:
                if self.target_dataset == 'caltech':
                    self.target_train_x = np.concatenate(self.target_train_x)
                    self.target_test_x = np.concatenate(self.target_test_x)
            print('data are successfully loaded.')
        elif FLAGS.task == 'syntrain_prepare':
            if self.dataset == 'dsprites' or FLAGS.visualize:
                self.train_pred = []
                if self.dataset == 'dsprites':
                    train_num = 800
                else:
                    train_num = 20
                for i in range(0, train_num, 200):
                    train_pred = load_file(self.datasave_dir, 'train_pred' + str(i) + '.pkl')
                    self.train_pred += train_pred
                self.target_train_pred = self.train_pred = np.stack(self.train_pred)[:train_num]
                haha = 1

        elif FLAGS.task == 'syntrain':
            if self.dataset != 'dsprites':
                model_specs = []
                if self.dataset == 'cifar100':
                    load_range = range(0, 100, 100)
                elif self.dataset == 'mini_imagenet':
                    load_range = range(0, 100, 50)
                elif self.dataset == 'mnist' or self.dataset == 'cifar10':
                    load_range = range(0, 20, 20)
                else:
                    raise NameError('unknown dataset!')
                for idx in load_range:
                    curr_model_specs = load_file(self.datasave_dir, 'train_pred' + str(idx) + '.pkl')
                    model_specs += curr_model_specs
                model_specs = np.stack(model_specs)
                self.model_specs = np.swapaxes(np.mean(model_specs, axis=-2), -1, -2)
                self.model_specs[self.model_specs < 0.2] = 0
                self.model_specs[self.model_specs > 0.8] = 1
                mask = np.ones(self.model_specs.shape)
                mask[self.model_specs == 0] = 0
                mask[self.model_specs == 1] = 0
                self.model_specs[mask == 1] = 0.5
                self.model_specs = 2 * self.model_specs - 1
                if not FLAGS.visualize:
                    if FLAGS.target_datasource == 'CUB':
                        self.target_train_x \
                            = np.stack(
                            [self.target_train_x[np.arange(self.target_train_x.shape[0])[self.target_train_y == i]][:29]
                             for i in range(self.num_target_dataset_classes)])
                else:
                    self.model_specs = self.model_specs[:20].reshape([-1, 10])
            else:
                z1 = np.zeros([720, 3])
                z1[np.arange(720),np.reshape(expand_and_repeat(np.arange(3),1,240),[-1])]=1
                z2 = np.zeros([720, 6])
                z2[np.arange(720),np.reshape(expand_and_repeat(expand_and_repeat(np.arange(6),1,40),0,3),[-1])]=1
                z3 = np.zeros([720, 40])
                z3[np.arange(720),np.reshape(expand_and_repeat(np.arange(40), 0, 18),[-1])]=1
                model_specs = np.concatenate([z1, z2, z3], axis=-1)
                self.model_specs = model_specs[self.train_domain_idx]
        elif FLAGS.task == 'test': # or FLAGS.task == 'simple_test':
            if self.dataset != 'dsprites':
                if self.target_dataset == 'CUB': # in self.available_target_datasets:
                    self.target_test_x = \
                        [self.target_test_x[np.arange(self.target_test_x.shape[0])[self.target_test_y == i]]
                         for i in range(self.num_target_dataset_classes)]
                self.train_pred, self.test_pred, self.test_feature, model_specs = [], [], [], []
                if not FLAGS.visualize:
                    self.target_train_pred, self.target_test_pred, self.target_test_feature = [], [], []
                if FLAGS.visualize:
                    load_range = range(0, 40, 40)
                else:
                    load_range = range(100, 200, 50)
                for idx in load_range:
                    curr_model_specs = load_file(self.datasave_dir, 'train_pred' + str(idx) + '.pkl')
                    model_specs += curr_model_specs
                    curr_test_pred = load_file(self.datasave_dir, 'test_pred' + str(idx) + '.pkl')
                    self.test_pred += curr_test_pred
                    curr_test_feature = load_file(self.datasave_dir, 'test_feature' + str(idx) + '.pkl')
                    self.test_feature += curr_test_feature
                    if not FLAGS.visualize:
                        curr_target_test_pred = load_file(self.datasave_dir, 'target_test_pred' + str(idx)
                                                          + '.pkl')
                        self.target_test_pred += curr_target_test_pred
                        curr_target_test_feature = load_file(self.datasave_dir, 'target_test_feature' + str(idx)
                                                             + '.pkl')
                        self.target_test_feature += curr_target_test_feature
                        curr_target_train_pred = load_file(self.datasave_dir, 'target_train_pred' + str(idx)
                                                           + '.pkl')
                        self.target_train_pred += curr_target_train_pred
                if not FLAGS.visualize:
                    self.test_pred = np.stack(self.test_pred)
                    self.test_feature = np.stack(self.test_feature)
                    self.target_test_pred = np.stack(self.target_test_pred)
                    self.target_test_feature = np.stack(self.target_test_feature)
                    model_specs = np.stack(model_specs)
                    if self.target_dataset in self.available_target_datasets:
                        self.target_test_pred = [self.target_test_pred[:, self.target_test_y == i]
                                                 for i in range(self.num_target_dataset_classes)]
                        self.target_test_feature = [self.target_test_feature[:, self.target_test_y == i]
                                                    for i in range(self.num_target_dataset_classes)]
                else:
                    self.test_pred = np.stack(self.test_pred)[20:]
                    self.test_feature = np.stack(self.test_feature)[20:]
                    model_specs = np.stack(model_specs)[20:]
                self.model_specs = np.swapaxes(np.mean(model_specs, axis=-2), -1, -2)
                self.model_specs[self.model_specs < 0.2] = 0
                self.model_specs[self.model_specs > 0.8] = 1
                mask = np.ones(self.model_specs.shape)
                mask[self.model_specs == 0] = 0
                mask[self.model_specs == 1] = 0
                self.model_specs[mask == 1] = 0.5
                self.model_specs = 2 * self.model_specs - 1
                if FLAGS.visualize:
                    self.model_specs = self.model_specs[:20].reshape([-1, 10])
            else:
                self.test_pred = []
                for i in range(0, 800, 200):
                    test_pred = load_file(self.datasave_dir, 'test_pred' + str(i) + '.pkl')
                    self.test_pred += test_pred
                self.test_pred = np.stack(self.test_pred)
                haha = 1
                z1 = np.zeros([720, 3])
                z1[np.arange(720),np.reshape(expand_and_repeat(np.arange(3),1,240),[-1])]=1
                z2 = np.zeros([720, 6])
                z2[np.arange(720),np.reshape(expand_and_repeat(expand_and_repeat(np.arange(6),1,40),0,3),[-1])]=1
                z3 = np.zeros([720, 40])
                z3[np.arange(720),np.reshape(expand_and_repeat(np.arange(40), 0, 18),[-1])]=1
                model_specs = np.concatenate([z1, z2, z3], axis=-1)
                if FLAGS.test_mode == 'in_train':
                    self.model_specs = model_specs[self.train_domain_idx]
                else:
                    self.model_specs = model_specs[self.test_domain_idx]
            haha = 1

    def load_valid_results(self):
        if FLAGS.source_dataset != 'dsprites':
            all_seed = range(2, 3)
        else:
            all_seed = range(1)  # 1
        path_name = FLAGS.recorder_savedir
        valid_results = {'task': [], 'model': [], 'score': [], 'valid_matrix': []}
        for seed in all_seed:
            print('load valid result seed: %d' % seed)
            if FLAGS.source_dataset != 'dsprites':
                file_name = 'LEEP_recorder' + str(seed) + '.' + FLAGS.target_dataset + '.pkl'
            else:
                file_name = 'pred_recorder' + str(seed) + '.pkl'
            results = load_file(path_name, file_name)
            valid_results['task'].extend(results['task'])
            valid_results['valid_matrix'].extend(results['valid_matrix'])
        self.valid_results = valid_results


    def get_model_pool_train_data(self, task, target=False, global_model=False):
        if global_model:
            task = np.arange(self.num_source_dataset_classes)
        if self.dataset != 'dsprites':
            if not target or FLAGS.transfer_method == 'direct_predict':
                train_x, test_x = self.train_x, self.test_x
            else:
                train_x, test_x = self.target_train_x, self.target_test_x
            x_train, x_test = train_x[task], test_x[task]
            y_train = np.reshape(multi_arange(x_train.shape[1], task.size, axis=-1), [-1])
            y_test = np.reshape(multi_arange(x_test.shape[1], task.size, axis=-1), [-1])
            y_train = keras.utils.to_categorical(y_train, task.size)
            y_test = keras.utils.to_categorical(y_test, task.size)
            x_train = np.reshape(x_train, [-1, self.img_size[0], self.img_size[1], self.img_size[2]])
            x_test = np.reshape(x_test, [-1, self.img_size[0], self.img_size[1], self.img_size[2]])
            num_classes = task.size
        else:
            x_train, x_test = self.train_x[task], self.test_x[task]
            y_train, y_test = self.train_y[task], self.test_y[task]
            num_classes = 1
        # Input image dimensions.
        input_shape = x_train.shape[1:]
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)
        return input_shape, num_classes, x_train, y_train, x_test, y_test

    def generate_syntrain_prepare_batch(self, domain_idx):

        if FLAGS.source_dataset != 'dsprites':
            valid_pred, valid_y, p, p1, train_idx, classes, domains = [], [], [], [], [], [], []
            for itr1 in range(domain_idx.shape[0]):
                task_classes = domain_idx[itr1]
                if FLAGS.target_dataset in ('CUB', 'caltech'):
                    task_valid_pred = [self.target_train_pred[:, self.target_train_y==i] for i in task_classes]
                    task_valid_y = np.concatenate([np.ones(np.sum(self.target_train_y==task_classes[i])) * i
                                                   for i in range(task_classes.shape[0])])
                    task_pc = np.concatenate([np.sum(task_valid_pred[i], axis=1, keepdims=True)
                                              for i in range(len(task_valid_pred))], axis=-2)
                    task_p = task_pc/np.sum(task_pc, axis=-2, keepdims=True)
                    task_p1 = task_pc/np.sum(task_pc, axis=-1, keepdims=True)
                    task_valid_pred = np.concatenate(task_valid_pred, axis=-2)
                    p.append(task_p)
                    p1.append(task_p1)
                    haha = 1
                else:
                    task_valid_pred = self.target_train_pred[:, task_classes, :100]
                    task_valid_y = np.concatenate([np.ones(100) * i for i in range(5)])
                    p, p1 = None, None
                valid_pred.append(task_valid_pred)
                valid_y.append(task_valid_y)
            if FLAGS.target_dataset not in ('CUB', 'caltech'):
                valid_pred = np.stack(valid_pred)
                valid_y = np.stack(valid_y)
        else:
            valid_pred = np.swapaxes(self.train_pred[:, domain_idx], 0, 1)
            valid_y = self.train_y[domain_idx]
            domains = domain_idx
        return valid_pred, valid_y, p, p1

    def generate_syntrain_batch(self, feature_itr, metric_itr):
        exp_config = global_objects.exp_config
        all_train_x = self.target_train_x
        meta_batch_size_task = exp_config.meta_batch_size_task
        num_pairwise_batch = exp_config.num_pairwise_batch
        n_shot = FLAGS.train_n_shot
        input_shape = exp_config.source_input_shape
        num_task_class = exp_config.num_target_task_class
        pairwise_batch_size_per_class = int(num_pairwise_batch*n_shot)
        task_domains = self.valid_results['task'][metric_itr][:meta_batch_size_task]
        train_y = np.stack(task_domains)
        data_idx = self.valid_results['data_idx'][metric_itr]
        if FLAGS.source_dataset != 'dsprites':
            data_idx_pre = self.valid_results['data_idx_pre'][feature_itr]
            train_y_pre = self.valid_results['train_y_pre'][feature_itr]
            coupled_idx_pre = couple_numpy_idx(train_y_pre.reshape([-1]), data_idx_pre, fix_idx2=True)
            if FLAGS.target_dataset in ('CUB', 'caltech'):
                train_x_pre = np.stack([all_train_x[coupled_idx_pre[i, 0]][coupled_idx_pre[i, 1]]
                                        for i in range(coupled_idx_pre.shape[0])]).reshape([meta_batch_size_task, 2,
                                        pairwise_batch_size_per_class]+input_shape)
            else:
                train_x_pre \
                    = all_train_x[coupled_idx_pre[:, 0], coupled_idx_pre[:, 1]].reshape([meta_batch_size_task, 2,
                                                        pairwise_batch_size_per_class]+input_shape)

        else:
            train_x_pre, train_y_pre = None, None
        coupled_idx = couple_numpy_idx(train_y.reshape([-1]), data_idx, fix_idx2=True)
        if FLAGS.target_dataset in ('CUB', 'caltech'):
            train_x = np.stack([all_train_x[coupled_idx[i, 0]][coupled_idx[i, 1]]
                for i in range(coupled_idx.shape[0])]).reshape([meta_batch_size_task, num_task_class,
                                                                n_shot] +input_shape)
        else:
            train_x = all_train_x[coupled_idx[:, 0], coupled_idx[:, 1]]
        if FLAGS.source_dataset != 'dsprites':
            train_x = train_x.reshape([meta_batch_size_task, num_task_class, n_shot]+input_shape)
        else:
            train_x = train_x.reshape([meta_batch_size_task, n_shot]+input_shape)
        return train_x_pre, train_y_pre, train_x, train_y

    def generate_test_batch(self, n_task, test_batch_size):
        """
        :param n_task: number of tasks to generate.
        :return:
        task_classes: a list of 1-d array of task class index.
        query_x, query_pred, query_feature: lists of ori_data, model predictions and model features
        for the target training set.
        test_x, test_pred, test_feature: similar to the query ones.

        """
        exp_config = global_objects.exp_config
        query_batch_size = FLAGS.test_n_shot
        if FLAGS.source_dataset != 'dsprites':
            n_classes_per_task = 5
            self.num_model_classes = 20
            self.test_classes = exp_config.num_target_dataset_class

            task_classes=np.concatenate([np.random.choice(self.test_classes, n_classes_per_task, replace=False)
                                         for i in range(n_task)])
            query_data_idx=np.asarray([np.random.choice(exp_config.num_test_data_per_class,
                                                        query_batch_size+test_batch_size,
                                       replace=False) for i in range(n_task * n_classes_per_task)])
        else:
            haha = 1
            if FLAGS.test_mode == 'in_train':
                test_domain_idx = self.train_domain_idx
                test_domain_config = self.train_domain_config
            else:
                test_domain_idx = self.test_domain_idx
                test_domain_config = self.test_domain_config
            domain_idx = np.random.choice(test_domain_idx.shape[0], n_task, replace=False)
            task_classes = test_domain_idx[domain_idx]
            domain_factor = test_domain_config[domain_idx]
            query_data_idx=np.asarray([np.random.choice(exp_config.num_test_data_per_class,
                                                        query_batch_size+test_batch_size,
                                       replace=False) for i in range(n_task)])
        coupled_idx = couple_numpy_idx(task_classes, query_data_idx, fix_idx2=True)
        if FLAGS.source_dataset != 'dsprites':
            x, pred, feature = self.target_test_x[coupled_idx[:, 0], coupled_idx[:, 1]], \
                               self.target_test_pred[:, coupled_idx[:, 0], coupled_idx[:, 1]], \
                               self.target_test_feature[:, coupled_idx[:, 0], coupled_idx[:, 1]]
            x = x.reshape([n_task, n_classes_per_task, query_batch_size+test_batch_size]+exp_config.source_input_shape)
            pred = pred.reshape([exp_config.num_source_test_model, n_task, n_classes_per_task,
                                 query_batch_size+test_batch_size, self.num_model_classes])
            feature = feature.reshape([exp_config.num_source_test_model, n_task, n_classes_per_task,
                                       query_batch_size+test_batch_size, -1])
            query_x, query_pred, query_feature \
                = x[:, :, :query_batch_size], pred[:, :, :, :query_batch_size], feature[:, :, :, :query_batch_size]
            test_pred, test_feature \
                = pred[:, :, :, query_batch_size:], feature[:, :, :, query_batch_size:]
            return query_x, query_pred, query_feature, test_pred, test_feature
        else:
            x = self.test_x[coupled_idx[:, 0], coupled_idx[:, 1]]
            pred = self.test_pred[:, coupled_idx[:, 0], coupled_idx[:, 1]]
            y = self.test_y[coupled_idx[:, 0], coupled_idx[:, 1]]
            if FLAGS.test_mode != 'in_train':
                pred = pred[self.test_domain_idx]
                num_model = self.num_test_model
            else:
                pred = pred[self.train_domain_idx]
                num_model = self.num_train_model
            x = x.reshape([n_task, query_batch_size+test_batch_size]+self.img_size)
            pred = pred.reshape([num_model, n_task, query_batch_size+test_batch_size, 2])
            y = y.reshape([n_task, query_batch_size+test_batch_size, 2])
            query_x, query_pred, query_y  \
                = x[:, :query_batch_size], pred[:, :, :query_batch_size], y[:, :query_batch_size]
            test_pred, test_y = pred[:, :, query_batch_size:], y[:, query_batch_size:]
            return query_x, query_pred, query_y, test_pred, test_y, task_classes, domain_factor
