from tensorflow.python.platform import flags
import numpy as np
from tqdm import tqdm
from utils.utils import save_file, set_seed
from utils import global_objects
import time
from transfer_methods.transfer_methods import evidence

FLAGS = flags.FLAGS

SHOW_TIME = True


def prepare(seed, exp_config, data_generator):
    num_train_itr = exp_config.spec_iterations
    recorder = {'task': [], 'input': [], 'model': [], 'idx': [], 'valid_matrix': []}
    print('start doing preparation training')
    start = time.time()
    for train_itr in tqdm(range(num_train_itr)):
        if exp_config.source_dataset != 'dsprites':
            domain_idx = np.stack([np.random.choice(exp_config.num_target_dataset_class,
                                           exp_config.num_target_task_class, replace=False)
                                   for i in range(exp_config.meta_batch_size_task)])
        else:
            domain_idx = np.random.choice(data_generator.train_domain_idx,
                                          exp_config.meta_batch_size_task, replace=False)
        valid_pred, valid_y, p, p1 = data_generator.generate_syntrain_prepare_batch(domain_idx)
        if exp_config.source_dataset != 'dsprites':
            mode = 'classification'
        else:
            mode = 'regression'
        if mode == 'regression':
            mse = np.mean(np.sum((valid_pred-np.expand_dims(valid_y, axis=1))**2, axis=-1), axis=-1)
            valid_recorder = mse
        else:
            p, valid_matrix, valid_matrix1 = evidence(valid_pred, valid_y, p, p1)
            valid_recorder = (p.astype(dtype=np.float16), valid_matrix.astype(dtype=np.float16),
                              valid_matrix1.astype(dtype=np.float16))
        if SHOW_TIME and (train_itr % 100 == 0):
            end = time.time()
            duration = end-start
            print('seed: %d, valid iter: %d, duration: %f' % (seed, train_itr, duration))
            start = time.time()
        recorder['task'].append(domain_idx)
        recorder['valid_matrix'].append(valid_recorder)
    return recorder


def syntrain_prepare():
    seed = FLAGS.exp_seed + FLAGS.prepare_seed_increment
    set_seed(seed)
    data_generator = global_objects.data_generator
    exp_config = global_objects.exp_config
    data_generator.set_config()
    data_generator.load_pred(mode='train')
    recorder = prepare(seed, exp_config, data_generator)
    path_name = FLAGS.recorder_savedir

    if exp_config.source_dataset == 'dsprites':
        file_name = 'pred_recorder'+str(seed)+'.pkl'
    else:
        file_name = 'LEEP_recorder'+str(seed)+ '.' + exp_config.target_dataset + '.pkl'
    file_content = recorder
    save_file(path_name, file_name, file_content)
    print('Spec training accomplished.')
    return recorder
