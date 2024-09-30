""" Code for loading and processing data. """
import numpy as np
from utils import global_objects

from tensorflow.python.platform import flags
from utils.utils import save_file

FLAGS = flags.FLAGS
exp_config = global_objects.exp_config


def generate_source_config():
    dataset = exp_config.source_dataset
    num_dataset_class = exp_config.num_source_dataset_class
    num_model_class = exp_config.num_source_model_class
    num_train_model = exp_config.num_source_train_model
    num_test_model = exp_config.num_source_test_model

    def get_config(num_model, thresh=10):
        config = []
        class_count = np.zeros(num_dataset_class)
        for itr1 in range(num_model):
            p = thresh - class_count
            p[p <= 0] = 1e-5
            p /= np.sum(p)
            class_idx = np.random.choice(num_dataset_class, num_model_class, replace=False, p=p)
            class_count[class_idx] += 1
            config.append(class_idx)
        return config, class_count
    train_model_config, train_model_count \
        = get_config(num_train_model, thresh=int(num_train_model * num_model_class / num_dataset_class))
    test_model_config, test_model_count \
        = get_config(num_test_model, thresh=int(num_test_model * num_model_class / num_dataset_class))

    # ====================================== save config ========================================
    path_name = FLAGS.config_path
    file_name = 'task_config.pkl'
    file_content = (train_model_config, test_model_config)
    save_file(path_name, file_name, file_content)







