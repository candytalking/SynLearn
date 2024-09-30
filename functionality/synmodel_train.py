import tensorflow as tf
from model.build_graph import SpecModel
from tensorflow.python.platform import flags
import numpy as np
from utils.utils import save_session, load_session, save_file, load_file
import os
from tqdm import tqdm
from utils import global_objects

FLAGS = flags.FLAGS
SPEC_AUC_PRINT_INTERVAL = 500
SPEC_PRINT_INTERVAL = 100
DATA_PRINT_INTERVAL = 100
SUMMARY_INTERVAL = 100
SAVE_INTERVAL = 1000
SPEC_DRAW_INTERVAL = 1000
PRE_TRAIN_ROUND = 20000


def training_schedule(itr, training_parameters, use_reg=True, in_pretrain_flag=False):
    # loss weight settings
    if FLAGS.use_metric_pretrain:
        if itr < PRE_TRAIN_ROUND:
            wc, w_metric = training_parameters['wc'][0], training_parameters['w_metric'][0]
            train_op = training_parameters['train_op'][0]
        else:
            wc, w_metric = training_parameters['wc'][1], training_parameters['w_metric'][1]
            train_op = training_parameters['train_op'][1]
    else:
        wc, w_metric = training_parameters['wc'][2], training_parameters['w_metric'][2]
        train_op = training_parameters['train_op'][2]
    # regularization weight settings
    if use_reg:
        w_reg = FLAGS.w_reg
    else:
        w_reg = 0.0
    if FLAGS.use_metric_pretrain:
        if itr < int(PRE_TRAIN_ROUND):
            train_feature_lr = FLAGS.init_model_update_lr
            train_metric_lr = 0.0
        elif itr < int(FLAGS.spec_iterations/2) + PRE_TRAIN_ROUND:
            train_feature_lr = FLAGS.init_model_update_lr/10
            train_metric_lr = FLAGS.init_model_update_lr
        else:
            train_feature_lr = train_metric_lr = FLAGS.init_model_update_lr/10
    else:
        if itr < int(FLAGS.spec_iterations/2):
            train_metric_lr = train_feature_lr = FLAGS.init_model_update_lr
        else:
            train_feature_lr = train_metric_lr = FLAGS.init_model_update_lr/10
    return wc, w_metric, w_reg, train_feature_lr, train_metric_lr, train_op


def generate_train_sampler(exp_config, data_generator, num_spec_iter):
    meta_batch_size_task = exp_config.meta_batch_size_task
    n_shot = FLAGS.train_n_shot
    if FLAGS.source_dataset != 'dsprites':
        num_task_classes = exp_config.num_target_task_class
        num_total_classes = exp_config.num_target_dataset_class
        n_data_provider = 100
        data_providers = np.stack([np.random.choice(num_total_classes, 5, replace=False)
                                   for i in range(n_data_provider)])
        num_total_iterations = exp_config.feature_pretrain_iterations + exp_config.spec_iterations
        num_total_dp = num_total_iterations * meta_batch_size_task
        sampled_data_providers = np.random.choice(n_data_provider, num_total_dp, replace=True)
        train_y_pre = np.stack([np.random.choice(data_providers[sampled_data_providers[i]], 2, replace=False)
                                for i in range(num_total_dp)])
        train_y_pre = train_y_pre.reshape([num_total_iterations, meta_batch_size_task, 2])
        n_shot_pre = exp_config.num_pairwise_batch * n_shot
        num_total_sample_pre = num_total_iterations * meta_batch_size_task * 2
        if FLAGS.target_dataset in ('caltech', 'CUB'):
            data_idx_pre \
                = np.asarray([np.random.choice(data_generator.num_target_train_y_per_class[train_y_pre.reshape([-1])[i]],
                              n_shot_pre, replace=True)
                              for i in range(num_total_sample_pre)]).reshape([num_total_iterations, -1, n_shot_pre])
        else:
            if data_generator.num_target_train_y_per_class > n_shot_pre:
                data_replace = False
            else:
                data_replace = True
            data_idx_pre \
                = np.asarray([np.random.choice(data_generator.num_target_train_y_per_class,
                                               n_shot_pre, replace=data_replace)
                              for i in range(num_total_sample_pre)]).reshape([num_total_iterations, -1, n_shot_pre])
        useful_data_per_class = int(np.ceil(FLAGS.part_proportion * 500))
        haha = 1
    else:
        train_y_pre, data_idx_pre = None, None
        num_total_iterations = num_spec_iter
        useful_data_per_class = data_generator.num_train_data_per_class
    # ===================================== spec train =====================================
    num_total_sample = exp_config.spec_iterations * meta_batch_size_task * num_task_classes
    if FLAGS.target_dataset in ('CUB', 'caltech'):
        train_y = np.stack([data_generator.valid_results['task'][i][:meta_batch_size_task]
                            for i in range(exp_config.spec_iterations)])
        data_idx = np.asarray([np.random.choice(data_generator.num_target_train_y_per_class[train_y.reshape([-1])[i]],
        n_shot, replace=True) for i in range(num_total_sample)]).reshape([exp_config.spec_iterations, -1, n_shot])
    else:
        num_total_sample = int(num_total_iterations * meta_batch_size_task * num_task_classes)
        if useful_data_per_class >= n_shot:
            replace = False
        else:
            replace = True
        data_idx = np.asarray([np.random.choice(useful_data_per_class, n_shot, replace=replace)
                               for i in range(num_total_sample)]).reshape([num_total_iterations, -1, n_shot])
    file_path = FLAGS.recorder_savedir
    file_name = 'nshot.' + str(n_shot) + '.pp.' + str(FLAGS.part_proportion) \
                + '.' + FLAGS.target_dataset + '.train_sampler.pkl'
    file_content = (train_y_pre, data_idx_pre, data_idx)
    save_file(file_path, file_name, file_content)


def feature_learn(exp_config, sess, data_generator, spec_model, feature_sess_name, seed):
    NUM_RESULT_TO_SHOW = 3
    spec_sum_result = np.zeros([NUM_RESULT_TO_SHOW, SPEC_PRINT_INTERVAL])
    meta_batch_size_task, n_shot, num_pairwise_batch \
        = exp_config.meta_batch_size_task, FLAGS.train_n_shot, exp_config.num_pairwise_batch
    recorder = []
    save_flag = 0
    # ===================================== the main training loop =======================================
    # ====================================================================================================
    ######################## generate train settings ###############################
    num_total_iterations = exp_config.feature_pretrain_iterations
    target_dataset = FLAGS.target_dataset
    file_path = FLAGS.recorder_savedir
    file_name = 'nshot.' + str(n_shot) + '.pp.' + str(FLAGS.part_proportion) \
                + '.' + target_dataset + '.train_sampler.pkl'
    (train_y_pre, data_idx_pre, data_idx) = load_file(file_path, file_name)
    data_generator.valid_results['train_y_pre'] = train_y_pre
    data_generator.valid_results['data_idx_pre'] = data_idx_pre
    data_generator.valid_results['data_idx'] = data_idx
    print('Done initializing, starting training.')
    for train_itr in tqdm(range(num_total_iterations)):
        sample_iter = train_itr
        num_class_per_domain = 2
        input_task_x_pre, input_task_y_pre, input_task_x, input_task_y \
            = data_generator.generate_syntrain_batch(feature_itr=train_itr, metric_itr=sample_iter)
        reshaped_input_task_x_pre \
            = np.reshape(input_task_x_pre, [meta_batch_size_task, num_class_per_domain,
                                            num_pairwise_batch, n_shot] + exp_config.source_input_shape)
        # ================================== generate pairwise masks =====================================
        pairwise_num_task_classes = meta_batch_size_task*num_class_per_domain
        task_mask_pre = np.eye(pairwise_num_task_classes)
        if num_class_per_domain > 1:
            for itr2 in range(pairwise_num_task_classes):
                task_mask_pre[itr2, itr2 + 1-2*(itr2 % 2)] = 1
        input_mask_pre = np.eye(pairwise_num_task_classes)
        input_mask_pre = np.reshape(np.tile(np.expand_dims(np.expand_dims(input_mask_pre, axis=-1), axis=2),
                                            [1, num_pairwise_batch, 1, num_pairwise_batch]),
                                    [pairwise_num_task_classes*num_pairwise_batch, -1])
        input_mask_pre = input_mask_pre - 2 * np.eye(pairwise_num_task_classes*num_pairwise_batch)
        task_mask_pre = np.reshape(np.tile(np.expand_dims(np.expand_dims(task_mask_pre, axis=-1), axis=2),
                                           [1, num_pairwise_batch, 1, num_pairwise_batch]),
                                   [pairwise_num_task_classes*num_pairwise_batch, -1])

        # ============================ obtain training parameters from schedule ===========================
        if train_itr < int(num_total_iterations/2):
            lr = exp_config.init_model_update_lr
        else:
            lr = exp_config.init_model_update_lr/10
        # ======================================= single-iter training =============================================
        feed_dict = {spec_model.input_task_pre: reshaped_input_task_x_pre,
                     spec_model.input_mask_pre: input_mask_pre, spec_model.task_mask_pre: task_mask_pre,
                     spec_model.w_reg: exp_config.w_reg, spec_model.train_feature_lr: lr}
        output_tensors = [spec_model.pretrain_op, spec_model.classwise_distance]
        len_nonprint_output = len(output_tensors)
        output_tensors.extend([spec_model.reg_loss, spec_model.classwise_triplet_loss,
                               spec_model.classwise_fraction_valid_triplets])
        len_print = len(output_tensors) - len_nonprint_output
        assert len_print == spec_sum_result.shape[0]
        result = sess.run(output_tensors, feed_dict=feed_dict)
        time_idx = int(train_itr % SPEC_PRINT_INTERVAL)
        for itr2 in range(len_print):
            spec_sum_result[itr2, time_idx] = result[itr2+len_nonprint_output]
        if train_itr % SUMMARY_INTERVAL == 0:
            print('\n')
            print('max in-class dist: %f, min out-class dist: %f\n'
                  %(np.max(result[1][0, :5]), np.min(result[1][0, 5:])))
            mean_result = np.mean(spec_sum_result, axis=1)
            str1 = '\n source: %s, target: %s'%(FLAGS.source_dataset, FLAGS.target_dataset)\
                   + ', config name: ' + str(FLAGS.config_name) \
                   + ', total iteration: ' + str(num_total_iterations) + ', curr iter: ' + str(train_itr) + '\n'
            str2 = 'IN FEATURE PRETRAIN paras: train lr: ' + str(lr) + ', nk: ' + str(FLAGS.train_n_shot) + '\n'
            str3 = ' reg loss: '+str(mean_result[0])+'\n' \
                   +' c-loss: '+str(mean_result[1])+' frac useful c-loss: '+str(mean_result[2])+'\n'
            print_str = str1+str2+str3
            print(print_str)
    if save_flag == 0:
        save_session(sess, FLAGS.logdir, feature_sess_name, 'seed'+str(seed))
    return recorder


def spec_learn(exp_config, sess, data_generator, spec_model, sess_name, seed, show_pretrain=False, show_ref=False):
    NUM_RESULT_TO_SHOW = 5
    if FLAGS.source_dataset != 'dsprites':
        NUM_RESULT_TO_SHOW += 2
        if show_ref:
            NUM_RESULT_TO_SHOW += 4
        if show_pretrain:
            NUM_RESULT_TO_SHOW += 2
    spec_sum_result = np.zeros([NUM_RESULT_TO_SHOW, SPEC_PRINT_INTERVAL])
    part_proportion = FLAGS.part_proportion
    input_size = exp_config.source_input_shape
    num_spec_iter = exp_config.spec_iterations
    num_train_model = exp_config.num_source_train_model
    meta_batch_size_task = exp_config.meta_batch_size_task
    meta_batch_size_spec = num_train_model
    n_shot, num_pairwise_batch = FLAGS.train_n_shot, exp_config.num_pairwise_batch
    if FLAGS.source_dataset != 'dsprites':
        model_config = data_generator.train_model_config
        model_specs = data_generator.model_specs
        num_task_classes = exp_config.num_target_task_class
        num_source_model_classes = 20
    else:
        model_specs = data_generator.model_specs
        num_task_classes, num_source_model_classes = 1, 1
    recorder = []
    save_flag = 0
    # ===================================== the main training loop =======================================
    # ====================================================================================================
    start_train_iteration = FLAGS.start_train_iteration
    ######################## generate train settings ###############################
    if FLAGS.source_dataset != 'dsprites' and FLAGS.use_metric_pretrain:
        pre_train_round = exp_config.feature_pretrain_iterations
    else:
        pre_train_round = 0
    target_dataset = FLAGS.target_dataset
    num_total_iterations = pre_train_round + num_spec_iter
    file_path = FLAGS.recorder_savedir
    file_name = 'nshot.' + str(n_shot) + '.pp.' + str(FLAGS.part_proportion) \
                + '.' + target_dataset + '.train_sampler.pkl'
    (train_y_pre, data_idx_pre, data_idx) = load_file(file_path, file_name)
    data_generator.valid_results['train_y_pre'] = train_y_pre
    data_generator.valid_results['data_idx_pre'] = data_idx_pre
    (train_y_pre, data_idx_pre, data_idx) = load_file(file_path, file_name)
    data_generator.valid_results['data_idx'] = data_idx
    print('Done initializing, starting training.')
    for train_itr in tqdm(range(start_train_iteration, num_total_iterations)):
        if FLAGS.use_metric_pretrain:
            sample_itr = np.maximum(train_itr - PRE_TRAIN_ROUND, 0)
        else:
            sample_itr = train_itr
        input_task_x_pre, input_task_y_pre, input_task_x, input_task_y \
            = data_generator.generate_syntrain_batch(feature_itr=sample_itr, metric_itr=sample_itr)
        # ================================== generate pairwise masks =====================================
        if FLAGS.source_dataset != 'dsprites':
            num_class_per_domain = 2
            pairwise_num_task_classes = input_task_x_pre.shape[0] * num_class_per_domain
            task_mask_pre = np.eye(pairwise_num_task_classes)
            for itr2 in range(pairwise_num_task_classes):
                task_mask_pre[itr2, itr2 + 1-2*(itr2 % 2)] = 1
            input_mask_pre = np.eye(pairwise_num_task_classes)
            input_mask_pre = np.reshape(np.tile(np.expand_dims(np.expand_dims(input_mask_pre, axis=-1), axis=2),
                                                [1, num_pairwise_batch, 1, num_pairwise_batch]),
                                        [pairwise_num_task_classes * num_pairwise_batch, -1])
            input_mask_pre = input_mask_pre - 2 * np.eye(pairwise_num_task_classes * num_pairwise_batch)
            task_mask_pre = np.reshape(np.tile(np.expand_dims(np.expand_dims(task_mask_pre, axis=-1), axis=2),
                                               [1, num_pairwise_batch, 1, num_pairwise_batch]),
                                       [pairwise_num_task_classes * num_pairwise_batch, -1])
            reshaped_input_task_x_pre = np.reshape(input_task_x_pre, [meta_batch_size_task, num_class_per_domain,
                                                                      num_pairwise_batch, n_shot,
                                                                      input_size[0], input_size[1], input_size[2]])
        # ============================= generate task and model class masks ==============================
        # ================================================================================================
        if FLAGS.source_dataset != 'dsprites':
            reshaped_input_task_x = np.zeros([meta_batch_size_task, num_task_classes, n_shot,
                                              input_size[0], input_size[1], input_size[2]])
            input_class_mask = np.zeros([meta_batch_size_task, num_task_classes])
            for task_idx in range(meta_batch_size_task):
                k = num_task_classes # [task_idx]
                reshaped_input_task_x[task_idx, :k] = input_task_x[task_idx]
                input_class_mask[task_idx, :k] = 1
            input_model_mask = np.zeros([meta_batch_size_spec, num_source_model_classes])
            for model_idx in range(meta_batch_size_spec):
                input_model_mask[model_idx, :num_source_model_classes] = 1
        else:
            reshaped_input_task_x = input_task_x.reshape([meta_batch_size_task, num_task_classes, n_shot,
                                                          input_size[0], input_size[1], input_size[2]])
        # ============================ test if spec models fit the task data =============================
        # ================================================================================================
        valid_keyword = 'valid_matrix'
        if FLAGS.source_dataset != 'dsprites':
            input_mask_per_model_class \
                = np.zeros([meta_batch_size_task, num_task_classes, 100, num_source_model_classes])
            a = data_generator.valid_results[valid_keyword][sample_itr]
            if FLAGS.target_dataset in ('CUB', 'caltech', 'cifar10', 'fashion_mnist'): # or FLAGS.visualize:
                (score, p, p1) = data_generator.valid_results[valid_keyword][sample_itr]
            else:
                (score, p, p1) = data_generator.valid_results[valid_keyword][sample_itr][0]
            score = score[:meta_batch_size_task]
            p = p[:meta_batch_size_task]
            p1 = p1[:meta_batch_size_task]
            input_mask_per_model_class[p > 0.7] = 1   # 0.7
            input_mask_per_model_class[p1 < 0.2] = 0
            # score is the larger the better!
            rank = np.argsort(np.argsort(score.reshape([-1])))
            input_mask = rank.reshape(score.shape)
            haha = 1
        else:
            score = np.stack([data_generator.valid_results[valid_keyword][sample_itr][i, data_generator.train_domain_idx]
                              for i in range(meta_batch_size_task)])
            rank = np.argsort(np.argsort(-score.reshape([-1])))
            input_mask = rank.reshape(score.shape)
        # ======================================== order permutation ==============================================
        sampled_model_idx_no_perm = np.arange(meta_batch_size_spec)
        perm_model_idx = np.random.permutation(meta_batch_size_spec)
        sampled_model_idx = sampled_model_idx_no_perm[perm_model_idx]
        input_spec_x, input_mask \
            = model_specs[sampled_model_idx], input_mask[:, sampled_model_idx]
        if FLAGS.source_dataset != 'dsprites':
            input_mask_per_model_class = input_mask_per_model_class[:, :, sampled_model_idx]
        if np.sum(np.abs(input_mask)) > 0:
            w_reg = exp_config.w_reg
        else:
            w_reg = 0.0
        if train_itr < pre_train_round + (num_total_iterations - pre_train_round)/2:
            train_feature_lr = train_metric_lr = exp_config.init_model_update_lr
        else:
            train_feature_lr = train_metric_lr = exp_config.init_model_update_lr/10
        if FLAGS.use_metric_pretrain:
            train_feature_lr = exp_config.init_model_update_lr/10
        # ======================================= single-iter training =============================================
        feed_dict = {spec_model.input_task: reshaped_input_task_x, spec_model.input_spec: input_spec_x,
                     spec_model.input_mask: input_mask, spec_model.w_reg: w_reg,
                     spec_model.train_feature_lr: train_feature_lr, spec_model.train_metric_lr: train_metric_lr}
        if FLAGS.source_dataset != 'dsprites': # and not FLAGS.visualize:
            feed_dict[spec_model.input_mask_per_model_class] = input_mask_per_model_class
            feed_dict[spec_model.input_class_mask] = input_class_mask
            feed_dict[spec_model.input_model_mask] = input_model_mask
            if show_pretrain:
                feed_dict[spec_model.input_task_pre] = reshaped_input_task_x_pre
                feed_dict[spec_model.input_mask_pre] = input_mask_pre
                feed_dict[spec_model.task_mask_pre] = task_mask_pre

        output_tensors = [spec_model.total_train_op, spec_model.show_pairwise_distance]
        if show_pretrain:
            output_tensors.extend([spec_model.classwise_distance])
        if FLAGS.source_dataset != 'dsprites':
            output_tensors.extend([spec_model.task_attention_weight, spec_model.model_attention_weight])
        len_nonprint_output = len(output_tensors)
        output_tensors.extend([spec_model.reg_loss,
                               spec_model.triplet_loss_eff, spec_model.frac_valid_triplets_eff,
                               spec_model.triplet_loss_pla, spec_model.frac_valid_triplets_pla])
        if FLAGS.source_dataset != 'dsprites':
            output_tensors.extend([spec_model.attention_loss, spec_model.bin_loss])
            if show_ref:
                output_tensors.extend([spec_model.triplet_loss_eff_ref, spec_model.frac_valid_triplets_eff_ref,
                                       spec_model.triplet_loss_pla_ref, spec_model.frac_valid_triplets_pla_ref])
        if show_pretrain:
            output_tensors.extend([spec_model.classwise_triplet_loss, spec_model.classwise_fraction_valid_triplets])
        len_print = len(output_tensors) - len_nonprint_output
        assert len_print == spec_sum_result.shape[0]
        result = sess.run(output_tensors, feed_dict=feed_dict)
        dist = result[1]
        if FLAGS.source_dataset != 'dsprites':
            if show_pretrain:
                c_dist, m_att = result[2], result[4]
            else:
                m_att = result[3]
        time_idx = int(train_itr % SPEC_PRINT_INTERVAL)
        for itr2 in range(len_print):
            spec_sum_result[itr2, time_idx] = result[itr2+len_nonprint_output]
        if train_itr % SUMMARY_INTERVAL == 0:
            print('\n')
            if show_pretrain:
                print('max in-class dist: %f, min out-class dist: %f\n'
                      %(np.max(c_dist[0, :5]), np.min(c_dist[0, 5:])))
            print('min metric dist: %f, max dist: %f\n'%(np.min(dist), np.max(dist)))
            if FLAGS.source_dataset != 'dsprites':
                corr = np.stack([input_mask[i, np.argsort(dist, axis=-1)[i]][:30]
                                 for i in range(meta_batch_size_task)])
                all_error = np.abs(m_att - input_mask_per_model_class)
                num_positive = np.sum(input_mask_per_model_class)
                num_negative = np.prod(input_mask_per_model_class.shape) - num_positive
                false_negative = np.sum(np.where(input_mask_per_model_class == 1, all_error, 0)) / num_positive
                false_positive = np.sum(np.where(input_mask_per_model_class == 0, all_error, 0)) / num_negative
                print(corr)
                print('prop false neg %f, prop false pos %f'%(false_negative, false_positive))
            mean_result = np.mean(spec_sum_result, axis=1)
            str1 = '\nSource: %s, Target: %s'%(FLAGS.source_dataset, FLAGS.target_dataset) \
                   + ', config name: ' + str(FLAGS.config_name) \
                   + ', Total Iteration: ' + str(num_total_iterations) \
                   + ', curr iter: ' + str(train_itr) + '\n' # + ', t-method: ' + str(FLAGS.transfer_method) + '\n'
            str2 = 'paras: train feature lr: ' + str(train_feature_lr) \
                   + ', train metric lr: ' + str(train_metric_lr) + ', w_reg: ' + str(w_reg) + '\n' \
                   + ', mp: ' + str(FLAGS.use_metric_pretrain) + ', pp: ' + str(part_proportion) \
                   + ', att_w: ' + str(FLAGS.att_w) + ', nk: ' + str(FLAGS.train_n_shot) + '\n'
            str3 = ' reg loss: '+str(mean_result[0])+'\n' \
                   +' m-loss-eff: '+str(mean_result[1])+' frac useful m-loss-eff: '+str(mean_result[2])+'\n' \
                   +' m-loss-pla: '+str(mean_result[3])+' frac useful m-loss-pla: '+str(mean_result[4])+'\n'
            if FLAGS.source_dataset != 'dsprites':
                str3 += ' a-loss: '+str(mean_result[5])+' b-loss: '+str(mean_result[6])+'\n'
                if show_ref:
                    str3 += ' ref m-loss-eff: '+str(mean_result[7]) \
                            + ' frac useful ref m-loss-eff: '+str(mean_result[8])+'\n'\
                            + ' ref m-loss-pla: '+str(mean_result[9])\
                            + ' frac useful ref m-loss-pla: '+str(mean_result[10]) \

            print_str = str1+str2+str3
            if show_pretrain:
                str4 = ' c-loss: ' + str(mean_result[-2]) + ' frac useful c-loss: ' + str(mean_result[-1]) + '\n'
                print_str += str4
            print(print_str)

    if save_flag == 0:
        save_session(sess, FLAGS.logdir, sess_name, 'seed'+str(seed))
    return recorder


def syn_train():
    seed = FLAGS.exp_seed
    exp_config = global_objects.exp_config
    data_generator = global_objects.data_generator
    data_generator.set_config()
    data_generator.load_source_model_specification(mode='train')
    data_generator.load_valid_results()
    data_generator.load_data()
    pretrain_batch = train_batch = None
    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.InteractiveSession()
        feature_sess_name='nk_'+str(FLAGS.train_n_shot)+'.pwb_'+str(exp_config.num_pairwise_batch)+\
                           '.seed_'+str(seed)+'.name_'+str('feature')
        spec_model = SpecModel(train_flag=True, exp_config=exp_config, reuse=False)
        if FLAGS.source_dataset != 'dsprites' and FLAGS.use_metric_pretrain:
            if FLAGS.start_train_iteration == 0:
                spec_model.construct_pretrain_model(reuse=False, pretrain_batch=pretrain_batch)
                exclude_var = []
            else:
                show_pretrain=False
                spec_model.construct_pretrain_model(reuse=False, pretrain_flag=False,
                                                    pretrain_batch=pretrain_batch, show_result=show_pretrain)
                load_session(sess, FLAGS.logdir, feature_sess_name, 'seed' + str(seed))
                exclude_var = tf.global_variables()
                spec_model.construct_model(reuse=False, reuse_feature_backbone=True, train_batch=train_batch,
                                           show_pretrain=show_pretrain)
        else:
            spec_model.construct_model(reuse=False, show_pretrain=False,
                                       train_batch=train_batch, reuse_feature_backbone=False)
            exclude_var = []
        sess.run(tf.variables_initializer([var for var in tf.global_variables() if var not in exclude_var]))
        if FLAGS.generate_train_sampler:
            generate_train_sampler(exp_config, data_generator, exp_config.spec_iterations)
        sess.close()
    print('sync model training has accomplished.')
    return 0
