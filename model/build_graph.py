""" building tf graphs. """

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.platform import flags
from model.NN_building_block import batch_triplet_loss, batch_classwise_triplet_loss, pairwise_distances, \
     ResNetFeatureBackBone, FCFeatureBackBone, TaskTransformer, ModelTransformer, xent_without_softmax_binary


FLAGS = flags.FLAGS


XentLoss = tf.nn.sparse_softmax_cross_entropy_with_logits
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class SingleSimplePredictor:

    def __init__(self, feature_length):
        self.scope = 'single_simple_predictor'
        with tf.variable_scope(self.scope, reuse=False):
            w_initializer = tf.initializers.zeros(dtype=tf.float32)
            b_initializer = tf.initializers.zeros(dtype=tf.float32)
            self.feature_length = feature_length
            self.num_predictor_class = 5
            self.weight_w = tf.get_variable('weight_w', [feature_length, self.num_predictor_class],
                                            initializer=w_initializer, dtype=tf.float32)
            self.weight_b = tf.get_variable('weight_b', [self.num_predictor_class],
                                            initializer=b_initializer, dtype=tf.float32)
            self.input_x = tf.placeholder(tf.float32, shape=[None, feature_length])
            self.input_y = tf.placeholder(tf.int32, shape=[None])
            self.input_w = tf.placeholder(tf.float32, shape=[feature_length,
                                                             self.num_predictor_class])
            self.input_b = tf.placeholder(tf.float32, shape=[self.num_predictor_class])
            self.assgin_weight_w = self.weight_w.assign(self.input_w)
            self.assgin_weight_b = self.weight_b.assign(self.input_b)
            reshaped_input_x = tf.reshape(self.input_x, [-1, feature_length])
            self.pred_y = tf.matmul(reshaped_input_x, self.weight_w) + self.weight_b
            loss = tf.reduce_mean(XentLoss(labels=self.input_y, logits=self.pred_y))
            self.test_output_y = tf.nn.softmax(self.pred_y, axis=-1)
            self.train_loss = tf.reduce_sum(loss)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            opt_vars = tf.trainable_variables()
            gvs = optimizer.compute_gradients(self.train_loss, var_list=opt_vars)
            self.train_op = optimizer.apply_gradients(gvs)


class ParallelSimplePredictor:

    def __init__(self, num_predictor, feature_length, num_predictor_class):
        self.scope = 'parallel_simple_predictor'
        with tf.variable_scope(self.scope, reuse=False):
            w_initializer = tf.initializers.zeros(dtype=tf.float32)
            b_initializer = tf.initializers.zeros(dtype=tf.float32)
            self.num_predictor = num_predictor
            self.feature_length = feature_length
            self.num_predictor_class = num_predictor_class
            curr_num = num_predictor # * self.num_predictor_class
            self.weight_w = tf.get_variable('weight_w', [feature_length, curr_num*self.num_predictor_class],
                                            initializer=w_initializer, dtype=tf.float32)
            self.weight_b = tf.get_variable('weight_b', [curr_num, self.num_predictor_class],
                                            initializer=b_initializer, dtype=tf.float32)
            self.input_x = tf.placeholder(tf.float32, shape=[curr_num, None, feature_length])
            self.input_y = tf.placeholder(tf.int32, shape=[curr_num, None])
            self.input_w = tf.placeholder(tf.float32, shape=[feature_length,
                                                             self.num_predictor_class*curr_num])
            self.input_b = tf.placeholder(tf.float32, shape=[curr_num, self.num_predictor_class])
            self.assgin_weight_w = self.weight_w.assign(self.input_w)
            self.assgin_weight_b = self.weight_b.assign(self.input_b)
            reshaped_input_x = tf.reshape(self.input_x, [-1, feature_length])
            self.pred_y = tf.reshape(tf.matmul(reshaped_input_x, self.weight_w),
                                     [curr_num, -1, curr_num, self.num_predictor_class])
            sliced_pred_y = tf.concat([tf.expand_dims(self.pred_y[i, :, i]+self.weight_b[i], axis=0)
                                       for i in range(curr_num)], axis=0)
            train_output_y = sliced_pred_y
            loss = tf.reduce_mean(XentLoss(labels=self.input_y, logits=train_output_y), axis=1)
            self.test_output_y = tf.nn.softmax(train_output_y, axis=-1)
            self.train_loss = tf.reduce_sum(loss)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            opt_vars = [var for var in tf.trainable_variables() if self.scope in var.name]
            gvs = optimizer.compute_gradients(self.train_loss, var_list=opt_vars)
            self.train_op = optimizer.apply_gradients(gvs)
            self.initializer = tf.variables_initializer(opt_vars)


class SourceModel:
    def __init__(self, img_size, num_class, train_flag, train_data_iterator=None,
                 test_data_iterator=None, network_type='normal', depth=18, reuse=False):
        self.scope = 'source_model'
        with tf.variable_scope(self.scope, reuse=reuse):
            if network_type == 'normal':
                output_length = 512
            else:
                output_length = 64
            self.feature_backbone_type = depth
            self.w_initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            self.b_initializer = tf.keras.initializers.Zeros(dtype=tf.float32)
            self.train_flag = train_flag
            self.img_size = img_size
            self.num_class = num_class

            if train_data_iterator is not None:
                self.train_data_iterator = train_data_iterator
                self.input_x, self.true_y, self.train_i = self.train_data_iterator.get_next()

            if test_data_iterator is not None:
                self.test_data_iterator = test_data_iterator
                self.input_x, self.true_y, self.test_i = self.test_data_iterator.get_next()

            self.lr = tf.placeholder(tf.float32)
            # =================== build model =======================
            self.feature_backbone = ResNetFeatureBackBone(self.img_size, reuse=reuse, train_flag=self.train_flag,
                                                          network_type=self.feature_backbone_type,
                                                          scope='feature_backbone')
            self.weight_w = tf.get_variable('final_w', [output_length, self.num_class],
                                            initializer=self.w_initializer, dtype=tf.float32)
            self.weight_b = tf.get_variable('final_b', [self.num_class],
                                            initializer=self.b_initializer, dtype=tf.float32)
            # ================= generate output =====================
            self.pred_feature = self.feature_backbone.forward(self.input_x, reuse=reuse)
            self.pred_y = tf.matmul(self.pred_feature, self.weight_w) + self.weight_b
            self.output_y = tf.nn.softmax(self.pred_y, axis=-1)
            # ================= compute losses ======================
            self.xent_loss = tf.reduce_mean(XentLoss(labels=self.true_y, logits=self.pred_y))
            # ================ set training env =====================
            if self.train_flag:
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
                feature_opt_vars = [var for var in tf.trainable_variables() if (self.scope in var.name)
                                    and ('feature_backbone' in var.name)]
                head_opt_vars = [var for var in tf.trainable_variables() if (self.scope in var.name)
                                 and ('feature_backbone' not in var.name)]
                opt_vars = feature_opt_vars + head_opt_vars
                reg_vars = [var for var in feature_opt_vars if ('bn' not in var.name) and ('bias' not in var.name)]
                self.reg_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(var) for var in reg_vars])
                self.train_loss = self.xent_loss + self.reg_loss
                gvs = optimizer.compute_gradients(self.train_loss, var_list=opt_vars)
                gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = optimizer.apply_gradients(gvs)


class SpecModel:

    def __init__(self, train_flag, exp_config, scope='spec_model', reuse=False):
        """ must call construct_model() after initialization """

        self.scope, self.reuse, self.train_flag = scope, reuse, train_flag
        self.exp_config = exp_config
        self.img_size = exp_config.source_input_shape
        self.dim_spec_input = exp_config.dim_spec_input
        if FLAGS.source_dataset == 'dsprites':
            self.num_class_per_domain_pre = 1
            self.num_max_task_classes = 1
        else:
            self.num_class_per_domain_pre = 2
            self.num_max_task_classes = 5
            self.num_max_model_classes = 20
        self.norm_type = 'batch_norm'
        self.meta_batch_size_task = exp_config.meta_batch_size_task
        self.meta_batch_size_spec = exp_config.meta_batch_size_spec
        if self.train_flag:
            self.n_shot = FLAGS.train_n_shot
        else:
            self.n_shot = FLAGS.test_n_shot
        self.num_pairwise_batch = exp_config.num_pairwise_batch
        self.test_batch_size_task = 1 # FLAGS.test_batch_size_task
        self.feature_backbone_type = 12
        if self.feature_backbone_type != 12:
            self.dim_metric_space = 512 # 512
        else:
            self.dim_metric_space = 640
        self.dim_task_feature = self.dim_spec_feature = self.dim_metric_space
        self.test_batch_size_spec = 100
        if self.train_flag:
            self.num_task = self.meta_batch_size_task
            self.num_model = self.meta_batch_size_spec
        else:
            self.num_task = self.test_batch_size_task
            self.num_model = self.test_batch_size_spec
        self.train_metric_lr = tf.placeholder(tf.float32)
        self.train_feature_lr = tf.placeholder(tf.float32)
        self.wc,self.w_metric,self.w_reg\
            =tf.placeholder(tf.float32),tf.placeholder(tf.float32),tf.placeholder(tf.float32)

        self.input_task = tf.placeholder(tf.float32, shape=[None, self.num_max_task_classes, self.n_shot,
                                                            self.img_size[0], self.img_size[1], self.img_size[2]])
        self.input_mask = tf.placeholder(tf.float32, shape=[None, None])
        if FLAGS.source_dataset != 'dsprites':

            self.input_mask_pre = tf.placeholder(tf.float32, shape=[None, None])
            self.task_mask_pre = tf.placeholder(tf.float32, shape=[None, None])
            self.input_spec = tf.placeholder(tf.float32,
                                             shape=[None, self.num_max_model_classes, self.dim_spec_input])
            self.input_class_mask = tf.placeholder(tf.int32, shape=[None, self.num_max_task_classes])
            self.input_mask_per_model_class \
                = tf.placeholder(tf.int32, shape=[None, self.num_max_task_classes, None, self.num_max_model_classes])
            self.input_model_mask = tf.placeholder(tf.int32, shape=[None, self.num_max_model_classes])
        else:
            self.input_spec = tf.placeholder(tf.float32, shape=[None, self.dim_spec_input])


    def construct_embedding_bags(self, inp):
        """
        :input:
        :return:
        [num_classes, num_pairwise_batch, dim_feature_input] class bag tensor
        """
        reshape_inp = tf.reshape(inp, [-1, self.num_pairwise_batch, self.n_shot, tf.shape(inp)[-1]])
        task_bags = tf.reduce_mean(reshape_inp, axis=-2)
        return [task_bags]

    def cal_dist(self, inp):
        x, c = inp
        x = tf.reshape(x, [-1, self.dim_metric_space]) # for debug!!!!!
        distance = pairwise_distances(x, c)
        return distance

    def cal_triplet_loss(self, inp):
        input_mask, input_distance = inp
        ori_loss, masks, triplet_loss, num_triplets, \
        num_valid_triplets, _ = batch_triplet_loss(input_mask, input_distance, gamma=self.exp_config.gamma)
        (triplet_loss_eff, triplet_loss_pla) \
            = [triplet_loss[i] / (num_valid_triplets[i] + 1e-5) for i in range(len(triplet_loss))]
        (frac_valid_triplets_eff, frac_valid_triplets_pla) \
            = [num_valid_triplets[i] / (num_triplets[i] + 1e-5) for i in range(len(num_valid_triplets))]
        return triplet_loss_eff, triplet_loss_pla, frac_valid_triplets_eff, frac_valid_triplets_pla

    def construct_pretrain_model(self, pretrain_batch=None, reuse=True, pretrain_flag=None, show_result=False):
        if pretrain_flag is None:
            pretrain_flag = self.train_flag
#        self.reuse = reuse
#        if pretrain_batch is None:
        self.input_task_pre \
            = tf.placeholder(tf.float32, shape=[None, self.num_class_per_domain_pre, self.num_pairwise_batch,
                                                    self.n_shot] + self.img_size)
#        else:
#            self.input_task_pre = pretrain_batch

        with tf.variable_scope(self.scope):
            with tf.variable_scope('model', reuse=reuse) as training_scope:
#               ====================================== models to use =====================================
                self.task_feature_backbone = ResNetFeatureBackBone(self.img_size, reuse=reuse, train_flag=self.train_flag,
                                                                   network_type=self.feature_backbone_type,
                                                                   scope='task_feature_backbone')
                if pretrain_flag or show_result:
                    feature_output = self.task_feature_backbone.forward(self.input_task_pre, reuse=reuse)
                else:
                    feature_output = self.task_feature_backbone.forward(self.input_task, reuse=reuse)
                if pretrain_flag or show_result:
#               ==========================================================================================
#               ===================================== feature pretrain ===================================
#               ==========================================================================================
                    self.input_feature_x_pre = tf.reshape(feature_output, [-1, self.num_class_per_domain_pre,
                                                                           self.num_pairwise_batch, self.n_shot,
                                                                           self.dim_task_feature])
#               ==================================== construct task bags =====================================
                    out_dtype = ([tf.float32])
                    result = tf.map_fn(self.construct_embedding_bags, elems=self.input_feature_x_pre,
                                       dtype=out_dtype, parallel_iterations=self.meta_batch_size_task)
                    self.mean_feature_x_pre = result
#               =================================== calibrate input size =====================================
                    self.task_z_mu_x_pre = tf.reshape(self.mean_feature_x_pre, [-1, self.dim_task_feature])
#               ==================================== class metric loss =======================================
                    self.show_classwise_triplet_loss, self.classwise_mask, self.classwise_triplet_loss, \
                    self.classwise_fraction_valid_triplets, self.classwise_distance \
                        = batch_classwise_triplet_loss(self.input_mask_pre, self.task_mask_pre,
                                                       self.task_z_mu_x_pre, c_gamma=self.exp_config.c_gamma,
                                                       distance_type='kl divergence')
                    self.embedding_loss = self.classwise_triplet_loss
#           =============================================================================================
#           ======================================= optimization ========================================
#           =============================================================================================
            if pretrain_flag:
                with tf.variable_scope('opt'):
                    self.spec_total_loss_unreg = self.embedding_loss
                    self.train_feature_optimizer, self.train_feature_optimizer_reset_op, self.train_feature_loss, \
                    self.train_feature_op = self.build_optimizer(self.train_feature_lr, scope='train',
                                                                 exclude_scope='metric')
                    self.pretrain_op = self.train_feature_op
                    self.total_train_op = self.train_op = self.train_feature_op
                    self.reg_loss = self.w_reg * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                    self.total_loss = self.spec_total_loss_unreg + self.reg_loss
                # ========================== for performance measure ============================
                with tf.name_scope('performance'):
                    # Summaries need to be displayed
                    # Whenever you need to record the loss, feed the mean loss to this placeholder
                    self.pretrain_loss_ph = tf.placeholder(tf.float32, shape=None, name='pretrain_loss_summary')
                    # Create a scalar summary object for the loss so it can be displayed
                    self.pretrain_loss_summary = tf.summary.scalar('pretrain loss', self.pretrain_loss_ph)
                    self.performance_summaries = tf.summary.merge([self.pretrain_loss_summary])

    def construct_model(self, train_batch=None, reuse=True, show_pretrain=False,
                        reuse_feature_backbone=False, show_ref=False):
        self.reuse = reuse

#            self.input_task = train_batch
        if FLAGS.source_dataset == 'dsprites':
            assert FLAGS.transfer_method == 'direct_predict'
        with tf.variable_scope(self.scope):
            with tf.variable_scope('model', reuse=reuse) as training_scope:
#               ====================================== models to use ==========================================
                if not reuse_feature_backbone:
                    if show_pretrain and self.train_flag:
                        self.construct_pretrain_model(reuse=False, pretrain_flag=True, show_result=True)
                    else:
                        self.task_feature_backbone = ResNetFeatureBackBone(self.img_size, reuse=False,
                                                                           train_flag=self.train_flag,
                                                                           network_type=self.feature_backbone_type,
                                                                           scope='task_feature_backbone')
                    task_feature = self.task_feature_backbone.forward(self.input_task, reuse=False) # build BN parameters
                self.spec_feature_extractor = FCFeatureBackBone(input_length=self.dim_spec_input,
                                                                output_length=self.dim_spec_feature,
                                                                num_fc_layer=2, reuse=reuse,
                                                                scope='metric/spec_feature_extractor')
                if FLAGS.source_dataset != 'dsprites':
                    self.task_transformer = TaskTransformer(self.num_max_task_classes, self.dim_task_feature,
                                                            output_dim=self.dim_metric_space, reuse=reuse,
                                                            scope='metric/task_transformer')
#                if FLAGS.transfer_method == 'on_top_feature':
                self.spec_transformer = ModelTransformer(self.num_max_task_classes, self.num_max_model_classes+1,
                                                         self.dim_spec_feature, input_q_dim=self.dim_metric_space,
                                                         output_dim=self.dim_metric_space, reuse=reuse,
                                                         scope='metric/spec_transformer')
                with tf.variable_scope('dummy'):
                    self.dummy_spec = tf.zeros([self.num_model, 1, self.dim_spec_feature])
#               ==========================================================================================
#               ===================================== triplet loss =======================================
#               ==========================================================================================
#               ============================== generate task and model rep ==============================
                task_feature = self.task_feature_backbone.forward(self.input_task, reuse=True)
#                                                                  dropblock=(self.dropblock_mask0,
#                                                                             self.dropblock_mask1))
                if self.num_class_per_domain_pre > 1:
                    task_feature = tf.reshape(task_feature, [-1, self.num_max_task_classes, self.n_shot,
                                                             self.dim_task_feature])
                    # average over intra-class instances
                    self.task_feature = tf.reduce_mean(task_feature, axis=2)
                    # unmasked_task_bag_size: [num_task, num_max_task_class, dim_metric_space]
                    self.unmasked_task_bags, self.z_mu_x, self.task_attention_weight \
                        = self.task_transformer.forward(self.task_feature, input_mask=self.input_class_mask,
                                                        reuse=reuse, train_flag=self.train_flag)
                    if FLAGS.source_dataset != 'dsprites':
                        spec_feature = self.spec_feature_extractor.forward(self.input_spec, train_flag=self.train_flag,
                                                                           output_activate=True, scope='', reuse=reuse)
                        # reshape to [num_model, num_max_model_classes, dim_spec_feature]
                        self.spec_feature = tf.reshape(spec_feature, [self.num_model, self.num_max_model_classes,
                                                                      self.dim_spec_feature])
                        self.spec_feature = tf.concat([self.dummy_spec, self.spec_feature], axis=1)

                        if show_ref:
                            a_ref = tf.cast(self.input_mask_per_model_class,tf.float32)
                        else:
                            a_ref = None
                        self.z_mu_c, self.z_mu_c_ref, self.z_mu_c_unave, self.model_attention_weight, \
                        self.a_w, self.e_w, self.z_mu_c_v, self.eps, self.norms \
                            = self.spec_transformer.forward(inp=self.spec_feature, inp_q=self.unmasked_task_bags,
                                                            a_ref=a_ref, input_mask=self.input_class_mask,
                                                            input_mask_2=self.input_model_mask,
                                                            reuse=self.reuse, train_flag=self.train_flag)
                        num_t = self.num_task
                        self.z_mu_x = tf.reshape(self.z_mu_x, [num_t, self.dim_metric_space])
                        self.z_mu_c = tf.reshape(self.z_mu_c, [num_t, -1, self.dim_metric_space])
                        result = tf.map_fn(self.cal_dist, elems=(self.z_mu_x, self.z_mu_c),
                                           dtype=tf.float32, parallel_iterations=self.num_task)
                        pairwise_distance = tf.reshape(result, [num_t, -1])
                        self.show_pairwise_distance = pairwise_distance
                        if show_ref:
                            self.z_mu_c_ref = tf.reshape(self.z_mu_c_ref, [num_t, -1, self.dim_metric_space])
                            result = tf.map_fn(self.cal_dist, elems=(self.z_mu_x, self.z_mu_c_ref),
                                               dtype=tf.float32, parallel_iterations=self.num_task)
                            pairwise_distance_ref = tf.reshape(result, [num_t, -1])
                            self.show_pairwise_distance_ref = pairwise_distance_ref
                    else:
                        spec_feature = self.spec_feature_extractor.forward(self.input_spec, train_flag=self.train_flag,
                                                                           output_activate=True, scope='', reuse=reuse)
                        self.z_mu_x = tf.reshape(self.z_mu_x, [-1, self.dim_metric_space])
                        self.z_mu_c = self.z_mu_c_ref = self.spec_feature = spec_feature
                        self.task_attention_weight = self.z_mu_x
                        self.model_attention_weight = self.z_mu_c
                        num_t = self.num_task
                        self.z_mu_x = tf.reshape(self.z_mu_x, [num_t, -1, self.dim_metric_space])
                        self.z_mu_c = tf.repeat(tf.expand_dims(self.z_mu_c, axis=0), num_t, axis=0)
                        result = tf.map_fn(self.cal_dist, elems=(self.z_mu_x, self.z_mu_c),
                                           dtype=tf.float32, parallel_iterations=self.num_task)
                        pairwise_distance = result
#                            = tf.reshape(result, [self.num_task*self.num_max_task_classes, -1])
#                        pairwise_distance_ref = pairwise_distance = pairwise_distances(self.z_mu_x, self.z_mu_c)
                        self.pairwise_distance = self.show_pairwise_distance = pairwise_distance
                else:
                    task_feature = tf.reshape(task_feature, [-1, self.n_shot, self.dim_task_feature])
                    self.z_mu_x = tf.reduce_mean(task_feature, axis=1)
                    self.z_mu_c = self.spec_feature_extractor.forward(self.input_spec, train_flag=self.train_flag,
                                                                      output_activate=False, scope='', reuse=reuse) #!!!!!output_activate=True!!!!
                    self.show_pairwise_distance = pairwise_distances(self.z_mu_x, self.z_mu_c,
                                                                     distance_type='kl divergence')
#               ==========================================================================================
#               ===================================== compute loss =======================================
#               ==========================================================================================
#               ===================================== metric loss =======================================
                if FLAGS.source_dataset != 'dsprites':
                    self.triplet_loss_eff, self.triplet_loss_pla, self.frac_valid_triplets_eff, \
                self.frac_valid_triplets_pla = self.cal_triplet_loss(inp=(self.input_mask, self.show_pairwise_distance))
                    if show_ref:
                        self.triplet_loss_eff_ref, self.triplet_loss_pla_ref, self.frac_valid_triplets_eff_ref, \
        self.frac_valid_triplets_pla_ref = self.cal_triplet_loss(inp=(self.input_mask, self.show_pairwise_distance_ref))
                    self.learn_triplet_loss = self.triplet_loss_eff + self.triplet_loss_pla
                    # =============================== attention loss ===============================
                    self.show_attention_loss = xent_without_softmax_binary(self.e_w[:, :, :, :, 1],
                                                                           self.input_mask_per_model_class)
                    show_attention_loss = tf.reshape(self.show_attention_loss, [-1])
                    self.dummy_attention_loss = tf.reduce_mean(tf.boolean_mask(show_attention_loss,
                                                       tf.equal(tf.reshape(self.input_mask_per_model_class, [-1]), 0)))
                    self.useful_attention_loss = tf.reduce_mean(tf.boolean_mask(show_attention_loss,
                                                       tf.equal(tf.reshape(self.input_mask_per_model_class, [-1]), 1)))
                    self.attention_loss = 1.0 * self.dummy_attention_loss + 1.0 * self.useful_attention_loss
                    self.total_metric_loss = self.learn_triplet_loss + FLAGS.att_w * self.attention_loss
#               ===================================== central loss =======================================
                    bin_loss_1 = self.a_w ** 2
                    bin_loss_2 = (1 - self.a_w) ** 2
                    self.bin_loss = tf.reduce_mean(tf.where(tf.greater(tf.stop_gradient(self.a_w), 0.5),
                                                            bin_loss_2, bin_loss_1))
                else:
                    self.triplet_loss_eff, self.triplet_loss_pla, self.frac_valid_triplets_eff, \
                self.frac_valid_triplets_pla = self.cal_triplet_loss(inp=(self.input_mask, self.show_pairwise_distance))
                    self.total_metric_loss = self.learn_triplet_loss = self.triplet_loss_eff + self.triplet_loss_pla
#           =============================================================================================
#           ======================================= optimization ========================================
#           =============================================================================================
            with tf.variable_scope('opt'):
                self.spec_total_loss_unreg = self.total_metric_loss # + self.wc * self.embedding_loss
                self.train_feature_optimizer, self.train_feature_optimizer_reset_op, self.train_feature_loss, \
                self.train_feature_op = self.build_optimizer(self.train_feature_lr, scope='train',
                                                             exclude_scope='metric')
                self.pretrain_op = self.train_feature_op
                # if FLAGS.transfer_method == 'direct_predict':
                self.train_metric_optimizer, self.train_metric_optimizer_reset_op, self.train_metric_loss, \
                self.train_metric_op = self.build_optimizer(self.train_metric_lr, scope='train',
                                                            exclude_scope='backbone')
                self.total_train_op = self.train_op = tf.group([self.train_metric_op, self.train_feature_op])
                self.reg_loss = self.w_reg * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                self.total_loss = self.spec_total_loss_unreg + self.reg_loss
            # ========================== for performance measure ============================
            with tf.name_scope('performance'):
                # Summaries need to be displayed
                # Whenever you need to record the loss, feed the mean loss to this placeholder
                self.total_loss_ph = tf.placeholder(tf.float32, shape=None, name='total_loss_summary')
                self.triplet_loss_ph = tf.placeholder(tf.float32, shape=None, name='triplet_loss_summary')
                self.kl_c_loss_ph = tf.placeholder(tf.float32, shape=None, name='kl_c_loss_summary')
                self.kl_x_loss_ph = tf.placeholder(tf.float32, shape=None, name='kl_x_loss_summary')
                self.kl_d_loss_ph = tf.placeholder(tf.float32, shape=None, name='kl_d_loss_summary')
                self.metric_loss_ph = tf.placeholder(tf.float32, shape=None, name='metric_loss_summary')

                # Create a scalar summary object for the loss so it can be displayed
                self.total_loss_summary = tf.summary.scalar('total loss', self.total_loss_ph)
                self.triplet_loss_summary = tf.summary.scalar('recon loss', self.triplet_loss_ph)
                self.kl_c_loss_summary = tf.summary.scalar('kl_c loss', self.kl_c_loss_ph)
                self.kl_x_loss_summary = tf.summary.scalar('kl_x loss', self.kl_x_loss_ph)
                self.kl_d_loss_summary = tf.summary.scalar('kl_d loss', self.kl_d_loss_ph)
                self.metric_loss_summary = tf.summary.scalar('metric loss', self.metric_loss_ph)

                self.performance_summaries = tf.summary.merge([self.total_loss_summary, self.triplet_loss_summary,
                                                               self.kl_c_loss_summary, self.kl_x_loss_summary,
                                                               self.kl_d_loss_summary, self.metric_loss_summary])

    def build_optimizer(self, lr, scope='', exclude_scope='empty scope', fc=False):
        with tf.variable_scope(scope):
            update_ops_all = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS)]
            update_ops = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                          if ('spec_model' in var.name and exclude_scope not in var.name)]
            if self.train_flag:
                ops = update_ops
            else:
                ops = []
            if not fc:
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)  # rr
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)  # rr
            optimizer_reset_op = tf.variables_initializer(optimizer.variables()) # rr
            opt_vars = [var for var in tf.trainable_variables() if 'spec_model' in var.name
                        and exclude_scope not in var.name]
            loss = self.spec_total_loss_unreg + self.w_reg * tf.add_n([tf.nn.l2_loss(var) for var in opt_vars]) # rr
            gvs = optimizer.compute_gradients(loss, var_list=opt_vars) # rr
            clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None] # rr
            with tf.control_dependencies(ops):
                train_op = optimizer.apply_gradients(clipped_gvs) # rr
            return optimizer, optimizer_reset_op, loss, train_op



