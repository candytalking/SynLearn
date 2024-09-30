from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

dataset_config = {
    'imagenet': {
        'input_shape': [224, 224, 3],  # [224, 224, 3],
        'num_class': 1000,
        'num_model_class': 100,
        'num_task_class': None,
        'num_train_model': 100,
        'test_data_per_class': 10,
        'num_test_model': 100,
        'model_depth':18,
        'model_type': 'normal'
    },
    'mini_imagenet': {
        'input_shape': [84, 84, 3],
        'num_class': 100,
        'num_model_class': 20,
        'num_task_class': 5,
        'num_train_model': 100,
        'test_data_per_class': 100,
        'num_test_model': 100,
        'model_depth': 18,
        'model_type': 'normal'
    },
    'cifar100': {
        'input_shape': [32, 32, 3],
        'num_class': 100,
        'num_model_class': 20,
        'num_task_class': 5,
        'test_data_per_class': 100,
        'num_train_model': 100,
        'num_test_model': 100,
        'model_depth': 20,
        'model_type': 'simple'
    },
    'CUB': {
        'num_class': 200,
        'num_model_class': None,
        'num_task_class': 5,
        'test_data_per_class': 30
    },
    'caltech': {
        'num_class': 256,
        'num_model_class': None,
        'test_data_per_class': 30,
        'num_task_class': 5
    },
    'cifar10': {
        'num_class': 10,
        'num_model_class': None,
        'test_data_per_class': 1000,
        'num_task_class': 5
    },
    'fashion_mnist': {
        'num_class': 10,
        'num_model_class': None,
        'test_data_per_class': 1000,
        'num_task_class': 5
    },
    'dsprites': {
        'input_shape': [64, 64, 1],
        'dim_output': 2,
        'domain_info': [3, 6, 40],
        'num_domain': 720,
        'num_train_domain': 503,
        'num_test_domain': 217,
        'model_depth': 18,
        'model_type': 'normal'
    }
}


class ExpConfig:

    def init_exp_config(self):
        # ----------------------- data settings -----------------------------
        self.source_dataset = FLAGS.source_dataset
        self.target_dataset = FLAGS.target_dataset
        self.source_input_shape = dataset_config[self.source_dataset]['input_shape']
        self.model_type = dataset_config[self.source_dataset]['model_type']
        self.model_depth = dataset_config[self.source_dataset]['model_depth']
        if self.source_dataset != 'dsprites':
            self.num_source_dataset_class = dataset_config[self.source_dataset]['num_class']
            self.num_source_model_class = dataset_config[self.source_dataset]['num_model_class']
            self.num_source_train_model = dataset_config[self.source_dataset]['num_train_model']
            self.num_source_test_model = dataset_config[self.source_dataset]['num_test_model']
        else:
            self.num_source_domain = dataset_config[self.source_dataset]['num_domain']
            self.source_domain_info = dataset_config[self.source_dataset]['domain_info']
        if self.target_dataset != 'dsprites':
            self.num_target_dataset_class = dataset_config[self.target_dataset]['num_class']
            self.num_target_task_class = dataset_config[self.target_dataset]['num_task_class']
            self.num_test_data_per_class = dataset_config[self.target_dataset]['test_data_per_class']
        else:
            self.num_target_domain = dataset_config[self.target_dataset]['num_domain']
            self.target_domain_info = dataset_config[self.target_dataset]['domain_info']

        # --------------------- source model settings ---------------------------
        self.model_pool_train_epochs = 120
        self.model_pool_train_batch_size = 128
        self.model_pool_data_augmentation = True
        if self.source_dataset != 'dsprites':
            self.dim_spec_input = 100
        else:
            self.dim_spec_input = 3

        # ----------------- MRE model training settings -------------------------
        self.w_reg = 1e-5
        self.c_gamma = 5.0
        self.gamma = 0.5
        self.meta_batch_size_spec = 100
        self.meta_batch_size_task = 5
        self.num_pairwise_batch = 5
        self.init_model_update_lr = 1e-3
        self.spec_iterations = 40000
        self.feature_pretrain_iterations = 20000

        # ----------------------- testing settings ------------------------------
        self.num_test_task= 500
        if self.target_dataset in ('CUB', 'caltech'):
            self.test_batch_size = 10
        else:
            self.test_batch_size = 50  # n data per class for testing