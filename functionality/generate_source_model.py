from __future__ import print_function
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import global_objects
from tensorflow.python.platform import flags
from model.build_graph import SourceModel
from utils.utils import save_session

FLAGS = flags.FLAGS

def lr_schedule(epoch):
    steps = (60, 90, 200)
    if epoch < steps[0]:
        return 0.1
    elif epoch < steps[1]:
        return 0.01
    else:
        return 0.001

def generate_models():
    exp_config = global_objects.exp_config
    data_generator = global_objects.data_generator
    data_generator.set_config()

    # Training parameters
    epochs = exp_config.model_pool_train_epochs
    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.InteractiveSession()
# ------------------------------ start training -----------------------------------
        task_idx = FLAGS.source_model_idx
        print('curr task: '+str(task_idx))
        train_iterator, test_iterator, num_train_data, num_test_data\
            = data_generator.generate_source_dataloader(task_idx)
        train_model = SourceModel(img_size=exp_config.source_input_shape, num_class=exp_config.num_source_model_class,
                                  train_data_iterator=train_iterator, test_data_iterator=None,
                                  train_flag=True, network_type=exp_config.model_type, depth=exp_config.model_depth)
        test_model = SourceModel(img_size=exp_config.source_input_shape, num_class=exp_config.num_source_model_class,
                                 train_data_iterator=None, test_data_iterator=test_iterator,
                                 train_flag=False, network_type=exp_config.model_type, depth=exp_config.model_depth,
                                 reuse=True)
        model_var = tf.global_variables()
        sess.run(tf.variables_initializer(model_var))
        sess.run([train_iterator.initializer, test_iterator.initializer])
        train_steps = np.ceil(num_train_data/exp_config.model_pool_train_batch_size).astype(np.int)
        test_steps = np.ceil(num_test_data/exp_config.model_pool_train_batch_size).astype(np.int)
        for epoch in range(epochs):
            lr = lr_schedule(epoch)
            train_loss, reg_loss = np.zeros(train_steps), np.zeros(train_steps)
            for step in tqdm(range(train_steps)):
                feed_dict = {train_model.lr: lr}
                outputs = [train_model.train_op, train_model.xent_loss, train_model.reg_loss,
                           train_model.input_x, train_model.true_y, train_model.pred_y]
                result = sess.run(outputs, feed_dict=feed_dict)
                train_loss[step], reg_loss[step] = result[1], result[2]
            mean_train_loss, mean_reg_loss = np.mean(train_loss), np.mean(reg_loss)
            true_y, pred_y, test_i = [], [], []
            for step in range(test_steps):
                result = sess.run([test_model.pred_y, test_model.true_y, test_model.test_i, test_model.xent_loss]) #, feed_dict=feed_dict)
                pred_y.append(result[0])
                true_y.append(result[1])
                test_i.append(result[2])
            if test_steps>1:
                pred_y = np.concatenate(pred_y, axis=0)
                true_y = np.concatenate(true_y, axis=0)
            else:
                pred_y = pred_y[0]
            pred_y = np.argmax(pred_y, axis=-1)
            num_correct = np.sum(pred_y==true_y)
            test_acc = num_correct/num_test_data
            print('epoch: %d, lr: %f, train_loss: %f, reg_loss: %f, '
                  'test_acc: %f'%(epoch, lr, mean_train_loss, mean_reg_loss, test_acc))
            haha = 1
        sess_name = 'source_model%d'%task_idx
        save_session(sess, FLAGS.model_pool_dir, sess_name, 'seed' + str(FLAGS.exp_seed))
    return 0
