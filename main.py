import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # show less details of tensorflow training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # str(np.argmax(memory_gpu))
import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
## config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

# ====================== basic setups ==========================
flags.DEFINE_string('gpu', '0', 'GPU to use. Must be string type.')
flags.DEFINE_integer('start', 4, 'seed to use during experiments. should be prime numbers.')
flags.DEFINE_integer('end', 1000, 'seed to use during experiments. should be prime numbers.')
flags.DEFINE_integer('exp_seed', 2, 'seed to use during experiments. should be prime numbers.')
flags.DEFINE_integer('seed_increment', 0, 'seed to use during exp.')
flags.DEFINE_integer('prepare_seed_increment', 0, 'seed to use during exp.')
flags.DEFINE_string('source_dataset', 'cifar100', 'datasets to generate source models and learn MRE model')
flags.DEFINE_string('target_dataset', 'imagenet', 'datasets for new target tasks')
flags.DEFINE_string('config_name', 'exp_0', 'exp configs')
flags.DEFINE_string('task', 'generate_source_model',
                    'generate_source_config or generate_source_model or generate_feature or syntrain_prepare '
                    'or syntrain or test or check_result')
flags.DEFINE_bool('reuse_test_result', False, 'in_train or in_test or out_test')
flags.DEFINE_bool('generate_train_sampler', True, 'whether or not to use batch normalization')
flags.DEFINE_integer('start_train_iteration', 0, 'number of iteration to start training')
flags.DEFINE_integer('start_upload_idx', 0, 'number of model to start upload')
flags.DEFINE_integer('num_model_upload', 25, 'number of model to start upload')
flags.DEFINE_integer('source_model_idx', 0, 'number of model to start upload')
flags.DEFINE_bool('use_metric_pretrain', False, 'whether or not to use metric pretrain stage')
flags.DEFINE_float('part_proportion', 1, 'whether or not to use full data for metric training')
flags.DEFINE_float('att_w', 1, 'whether or not to use full data for metric training')
# Training options
flags.DEFINE_integer('train_n_shot', 10, 'training K')
flags.DEFINE_integer('test_n_shot', 20, 'testing K')
# ====================== logging, saving, and testing options ==========================
flags.DEFINE_string('path_prefix', '/home/worker/exp/smr/', 'prefix of saved paths.')
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('config_path', 'config/', 'directory for exp configs.')
flags.DEFINE_string('logdir', 'log/', 'directory for summaries and checkpoints.')
flags.DEFINE_string('spec_savedir', 'specification/', 'directory for saving specs.')
flags.DEFINE_string('model_savedir', 'model/', 'directory for saving models.')
flags.DEFINE_string('recorder_savedir', 'recorder/', 'directory for saving recorders.')
flags.DEFINE_string('model_pool_dir', 'model_pool/', 'directory for saving model pool.')
flags.DEFINE_string('pred_savedir', 'pred/', 'directory for saving predictions.')
flags.DEFINE_string('dataset_dir', 'dataset/', 'directory for datasets.')
flags.DEFINE_string('datasave_dir', 'datasave/', 'directory for saving temp data.')
flags.DEFINE_string('result_dir', 'result/', 'directory for saving exp_results.')
flags.DEFINE_string('summary_dir', 'summary/', 'directory for saving tensorboad summaries.')

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # show less details of tensorflow training
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu  # str(np.argmax(memory_gpu))
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from functionality.generate_source_config import generate_source_config
from functionality.generate_source_model import generate_models
from functionality.syntrain_prepare import syntrain_prepare
from functionality.synmodel_train import syn_train

from utils import global_objects

function_mapper = {
    'generate_soruce_config': generate_source_config,
    'generate_source_model': generate_models,
    'syntrain_prepare': syntrain_prepare,
    'syntrain': syn_train
}

def main():
    task = FLAGS.task
    seed = FLAGS.exp_seed  # TO DO: new version: set seed in task functions
    global_objects.exp_config.init_exp_config()
    function_mapper[task]()
    print('task successfully completed.')


if __name__ == "__main__":
    main()
