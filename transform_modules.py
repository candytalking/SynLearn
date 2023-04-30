"""
Simplified demo code implemented based on Tensorflow 1 for NeurIPS'22 submission:
"Pre-trained model reusability evaluation for small-data transfer learning".
The code is NOT runnable: it is only used for illustrating the task and model transform modules.
The full code will be released when the paper gets published.

"""


import tensorflow as tf


xavier_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)


class TaskTransform:

    def __init__(self, input_dim=640, output_dim=640, scope='task_transform', reuse=True):
        self.input_dim, self.output_dim, self.scope = input_dim, output_dim, scope
        self.weights = {}
        with tf.variable_scope(self.scope, reuse):
            self.weights['w_q'] = tf.get_variable('w_q', [input_dim, output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_k'] = tf.get_variable('w_k', [input_dim, output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_v'] = tf.get_variable('w_v', [input_dim, output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_fc'] = tf.get_variable('w_fc', [output_dim, output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)
            self.weights['b_fc'] = tf.get_variable('b_fc', [output_dim],
                                                   initializer=xavier_initializer, dtype=tf.float32)

    def task_transform_block(self, inp):
        """
        INPUT:
            inp: input feature for each task class
        OUTPUT:
            each_class_x: output feature for each task class
            x: output feature for task
        """
        w_k, w_v, w_q = self.weights['w_k'], self.weights['w_k'], self.weights['w_k']
        w_fc, b_fc = self.weights['w_fc'], self.weights['b_fc']
        outp1 = inp
        k, v, q = inp @ w_k, inp @ w_v, inp @ w_q
        a = tf.reduce_sum(tf.multiply(tf.expand_dims(q, axis=2), tf.expand_dims(k, axis=1)), axis=-1) \
            / tf.sqrt(tf.cast(self.output_dim, tf.float32))
        c = a @ v
        outp1 = outp1 + c
        outp2 = outp1
        outp2 = tf.reduce_mean(outp2, axis=-2)
        outp2 = tf.matmul(outp2, w_fc) + b_fc
        return outp1, outp2

    def forward(self, inp, reuse=True):
        with tf.variable_scope(self.scope, reuse=reuse):
            each_class_x, x = self.task_transform_block(inp)
            return each_class_x, x


class ModelTransform:

    def __init__(self, input_dim=640, output_dim=640, scope='model_transform', reuse=True):
        self.weights = {}
        self.input_dim, self.output_dim, self.scope = input_dim, output_dim, scope
        with tf.variable_scope(scope, reuse):
            self.weights['w_q'] = tf.get_variable('w_q', [input_dim, output_dim],
                                                  initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_k'] = tf.get_variable('w_k', [input_dim, output_dim],
                                                  initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_v'] = tf.get_variable('w_v', [input_dim, output_dim],
                                                  initializer=xavier_initializer, dtype=tf.float32)
            self.weights['w_fc'] = tf.get_variable('w_fc', [output_dim, output_dim],
                                                    initializer=xavier_initializer, dtype=tf.float32)
            self.weights['b_fc'] = tf.get_variable('b_fc', [output_dim],
                                                    initializer=xavier_initializer, dtype=tf.float32)
    def weighted_ensemble_layer(self, z_c, p):
        """
        INPUT:
            z_c: feature to combine.
            p: weight.
        OUTPUT:
            z: weighted combined feature.
        """
        gamma = p/(tf.repeat(tf.maximum(tf.reduce_sum(p, axis=-1, keepdims=True), 1.0), tf.shape(p)[-1], axis=-1))
        gam = tf.repeat(tf.expand_dims(gamma, axis=-1), tf.shape(z_c)[-1], axis=-1)
        z = tf.multiply(z_c, gam)
        return z

    def model_transform_block(self, inp, each_class_x):
        """
        INPUT:
            inp: input feature for each model class
            each_class_x: input feature for each task class
        OUTPUT:
            c: output feature for model
            a: task-model attention weight, used for attention supervision.
        """
        w_k, w_v, w_q = self.weights['w_k'], self.weights['w_v'], self.weights['w_q']
        w_fc, b_fc = self.weights['w_fc'], self.weights['b_fc']
        v, k, q = inp @ w_v, inp @ w_k, each_class_x @ w_q
        norm_q = tf.sqrt(tf.reduce_sum(q ** 2, axis=-1, keepdims=True))
        norm_k = tf.sqrt(tf.reduce_sum(k ** 2, axis=-1, keepdims=True))
        norm_factor = (1e-5+tf.tensordot(norm_q,norm_k,axes=[[-1],[-1]])
                       /tf.sqrt(tf.cast(self.output_dim,tf.float32)))
        a = tf.tensordot(q, k, axes=[[-1], [-1]]) / norm_factor
        outp = self.weighted_ensemble_layer(v, a)
        outp = tf.reduce_sum(outp, axis=-2)
        outp = tf.matmul(outp, w_fc) + b_fc
        return outp, a

    def forward(self, inp, inp_q, reuse=True):
        with tf.variable_scope(self.scope, reuse=reuse):
            c, a = self.model_transform_block(inp, inp_q)
            return c, a



