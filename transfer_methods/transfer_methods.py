import keras
from tensorflow.python.platform import flags
import numpy as np
from utils.utils import margin_calculator, multi_arange
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import Callback
from baselines.evidence_methods import LEEP_score

FLAGS = flags.FLAGS


class FeatureBasedModel:
    def __init__(self, model, weight):
        self.model = model
        self.weight = weight

    def recover_weight(self):
        self.model.set_weights(self.weight)


class TransferConfig:
    def __init__(self):
        self.transfer_method_names = ("direct_predict", "on_top_feature", "fine_tune",)
        self.direct_pred = direct_pred
        self.on_top_feature = on_top_feature
        self.fine_tune = fine_tune
        self.transfer_method = {"direct_predict": direct_pred, "on_top_feature": on_top_feature, "fine_tune": fine_tune}
        self.num_transfer_method = 2


def calculate_accuracy(pred_y, test_y, num_task_classes, num_test_data_per_class):
    test_y = np.reshape(test_y, [-1])
    num_data = test_y.shape[0]
    acc_vec = np.zeros(num_task_classes+1)
    acc_vec[0] = np.sum(pred_y == test_y)/num_data
    for itr1 in range(num_task_classes):
        curr_pred_y = pred_y[itr1*num_test_data_per_class:(itr1+1)*num_test_data_per_class]
        curr_test_y = test_y[itr1*num_test_data_per_class:(itr1+1)*num_test_data_per_class]
        acc_vec[itr1+1] = np.sum(curr_pred_y == curr_test_y)/num_test_data_per_class
    return acc_vec


def get_predictor_data(ori_x, num_class, num_data_per_class, num_max_task_classes, class_idx=None):
    if class_idx is None:
        y = np.repeat(np.expand_dims(np.arange(num_class), axis=-1), num_data_per_class, axis=-1)
    else:
        y = np.ones([ori_x[0].shape[0], ori_x[0].shape[1]])
        y[class_idx] = 0
    x = np.concatenate([np.expand_dims(np.reshape(ori_x[i], [-1, ori_x[i].shape[-1]]), axis=0)
                        for i in range(len(ori_x))], axis=0)
    y = np.reshape(y, [-1])
    one_hot_y = keras.utils.to_categorical(y, num_max_task_classes)
    return x, y, one_hot_y


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


class NBatchLogger(Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        num_per_batch = logs['size']
        self.seen += num_per_batch
        curr_epoch = int(self.seen/num_per_batch)
        if curr_epoch % self.display == 0:
            print('\n{0}/{1} - Batch Loss: {2}'.format(curr_epoch, self.params['epochs'], self.params['metrics'][0]))


def build_predictor(input_dim, num_classes):
    inputs = Input(shape=[input_dim])
    x = inputs
    #x = Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
    class_outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=class_outputs)
    optimizer = Adam(lr=FLAGS.model_pool_finetune_lr)
    lr_metric = get_lr_metric(optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', lr_metric])
    return model


def direct_pred(train_feature, pred_output, pred_y, pred_model, sess, mode='classification'):
    if mode == 'classification':
        pred_ave_output = np.mean(pred_output, axis=1)
        label_wise_margin = margin_calculator(pred_ave_output)
        insta_wise_margin = np.transpose(margin_calculator(np.transpose(pred_ave_output)))
        total_margin = np.concatenate([np.expand_dims(label_wise_margin, axis=-1),
                                      np.expand_dims(insta_wise_margin, axis=-1)], axis=-1)
        return total_margin, None
    else:
        mean_squared_loss = np.mean(np.sum((pred_output-pred_y)**2, axis=-1), axis=0)
        return mean_squared_loss, None


def fine_tune(model, x, output=False):
    num_task_classes = np.shape(x)[0]
    num_valid_data_per_class = FLAGS.spec_valid_batch_size_per_class
    num_train_per_class = int(0.1 * num_valid_data_per_class)
    num_test_per_class = num_valid_data_per_class-num_train_per_class
    train_x, train_y, train_one_hot_y = get_predictor_data(x, num_task_classes, num_train_per_class)
    test_x, test_y, test_one_hot_y = get_predictor_data(x, num_task_classes, num_test_per_class, start_idx=num_train_per_class)
    pred_model_input, pred_model_feature = model[1][0], model[1][1]
    pred_model = build_predictor(pred_model_input, pred_model_feature, num_task_classes)
    show_progress = NBatchLogger(display=20)
    pred_model.fit(train_x, train_one_hot_y, batch_size=32, epochs=100, shuffle=True, verbose=0)
                   # validation_data=(test_x, test_one_hot_y), shuffle=True, callbacks=[show_progress])
    pred_output = pred_model.predict(test_x)
    pred_y = np.argmax(pred_output, axis=-1)
    valid_acc = calculate_accuracy(pred_y, test_y, num_task_classes, num_test_per_class)
    if output:
        return valid_acc, pred_output
    else:
        return valid_acc


def calculate_valid_model(pred_ave_output, num_model, num_task_classes, train_model_config):
    summary = []
    thresh = (0.1, 0.2, 0.3, 0.4, 0.5)
    for curr_thresh in thresh:
        class_summary = []
        right_idx = np.tile(np.reshape(np.arange(num_task_classes), [1, -1]),
                            [num_model, 1])
        positive_indicator = np.arange(num_model)[np.sum(np.argmax(pred_ave_output, axis=-1) == right_idx, axis=-1)
                                                  == num_task_classes]
        useful_indicator = np.zeros([num_task_classes, num_model])
        for class_idx in range(num_task_classes):
            useful_indicator[class_idx] = np.where(pred_ave_output[:, class_idx, class_idx] > curr_thresh, 1, 0)
        useful_idx = list(np.arange(num_model)[np.sum(useful_indicator, axis=0)==num_task_classes])
        useful_idx = [i for i in useful_idx if i in positive_indicator]
        useful_model = [train_model_config[i] for i in useful_idx]
        summary.append((useful_idx, useful_model))
    return summary


def evidence(pred, y, p, p1):
    if FLAGS.target_dataset in ('CUB', 'caltech'): #and not FLAGS.visualize:
        score, p, p1 = LEEP_score(pred, y, p=p, p1=p1)
        p = np.swapaxes(p, 1, 2)
        p1 = np.swapaxes(p1, 1, 2)
    else:
        (n_task, n_model, n_task_class, n_data_per_class, n_model_class) = pred.shape
        y = np.reshape(multi_arange(n_data_per_class, stop=n_task_class, axis=1), [-1])
        pred = pred.reshape([n_task, n_model, -1, n_model_class])
        score, p, p1 = LEEP_score(pred, y)
        score = score.reshape([n_task, n_model])
        #    y = np.swapaxes(y, -1, -2)
        p = np.swapaxes(p.reshape([n_task, n_model, n_task_class, n_model_class]), 1, 2)
        p1 = np.swapaxes(p1.reshape([n_task, n_model, n_task_class, n_model_class]), 1, 2)
    return score, p, p1


def on_top_feature(train_feature, valid_feature, pred_model, sess, num_iter=2000, mode='validation'):
    num_model, num_max_task_classes, num_train_data_per_class \
        = train_feature.shape[0], train_feature.shape[1], train_feature.shape[2]
    num_task_classes = np.shape(train_feature[0])[0]
    input_dim = train_feature.shape[-1]
    train_x = np.reshape(train_feature, [num_model, -1, input_dim])

    train_y = np.reshape(np.tile(np.reshape(np.arange(num_task_classes), [1, num_task_classes, 1]),
                                 [num_model, 1, num_train_data_per_class]), [num_model, -1])
    num_train_data = num_train_data_per_class*num_task_classes
    batch_size = int(num_train_data_per_class*num_task_classes)
    num_batch_per_epoch = int(num_train_data/batch_size)

    # ========================== start fine tune ===========================
    if mode == 'prediction':
        print('fine-tuning')
        num_fine_tune_round = num_iter
    else:
        num_fine_tune_round = num_iter
    sess.run(pred_model.initializer)
    for itr1 in range(num_fine_tune_round):
        # print('itr: ', itr1)
        data_idx = np.random.permutation(num_train_data)
        for itr2 in range(num_batch_per_epoch):
            curr_data_idx = data_idx[itr2*batch_size:(itr2+1)*batch_size]
            curr_train_x, curr_train_y = train_x[:, curr_data_idx, :], train_y[:, curr_data_idx]
            feed_dict = {pred_model.input_x: curr_train_x,
                         pred_model.input_y: curr_train_y
                         }
            result = sess.run([pred_model.train_op, # single_model.train_op,
                               pred_model.train_loss], feed_dict=feed_dict)
        if itr1 % 50 == 0:
            print('iter: %d, train loss: %f'%(itr1, result[1]))
    # ========================== do testing ===========================
    num_valid_data_per_class = valid_feature.shape[2]
    valid_x = np.reshape(valid_feature, [num_model, -1, input_dim])
    feed_dict = {pred_model.input_x: valid_x}
    pred_output = sess.run(pred_model.test_output_y, feed_dict=feed_dict)
    pred_output = np.reshape(pred_output, [num_model, num_task_classes, num_valid_data_per_class, num_task_classes])
    pred_ave_output = np.mean(pred_output, axis=-2)
    return pred_ave_output, pred_output
