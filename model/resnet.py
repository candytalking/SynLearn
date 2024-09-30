from __future__ import print_function
import keras
from keras.layers import Convolution2D, Dense, BatchNormalization, Activation, \
     GlobalAveragePooling2D, Input
# from custom_layers.scale_layer import Scale
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def lr_classification_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    if FLAGS.datasource == 'mnist' or FLAGS.datasource == 'cifar10':
        epoch_seg = (20, 25, 30)
    else:
        epoch_seg = (60, 90, 120)
    lr = 1e-1
    if epoch > epoch_seg[2]:
        lr *= 1/1000
    elif epoch > epoch_seg[1]:  # 160
        lr *= 1/100
    elif epoch > epoch_seg[0]:  # 120
        lr *= 1/10
    print('Learning rate: ', lr)
    return lr


def lr_regression_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-2
    if epoch > 120:  # 160
        lr *= 1e-3
    elif epoch > 90:  # 120
        lr *= 1e-2
    elif epoch > 60:  # 80
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def conv_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu',
                 batch_normalization=True, conv_first=True, is_trainable=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Convolution2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                         kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), trainable=is_trainable)
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(trainable=is_trainable)(x)
        if activation is not None:
            x = Activation(activation, trainable=is_trainable)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(trainable=is_trainable)(x)
        if activation is not None:
            x = Activation(activation, trainable=is_trainable)(x)
        x = conv(x)
    return x


# def resnet_v1(input_shape, num_res_blocks, num_classes=10, regression=False, feature_generator=False):
def build_resnet(input_shape, depth=20, mode='simple', dim_output=10, regression=False, trainable=True):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if not trainable:
        is_trainable = False
    else:
        is_trainable = True

    # Start model definition.
    if mode == 'simple':
#        num_filters = [16, 32, 64, 128]
        num_filters = [64, 128, 256, 512]
    else:
        num_filters = [64, 128, 256, 512]

    if depth == 20:
        n_stage = 3
    else:
        n_stage = 4
    num_res_blocks = 2

    inputs = Input(shape=input_shape)
    x = conv_layer(inputs=inputs, num_filters=num_filters[0], is_trainable=is_trainable)
    # Instantiate the stage of residual units
    for stage in range(n_stage):
        for res_block in range(num_res_blocks):
            strides = 1
            if stage > 0 and res_block == 0:  # first layer but not first stage
                strides = 2  # downsample
            y = conv_layer(inputs=x, num_filters=num_filters[stage], strides=strides, is_trainable=is_trainable)
            if depth == 20:
                y = conv_layer(inputs=y, num_filters=num_filters[stage], strides=1, is_trainable=is_trainable)
            y = conv_layer(inputs=y, num_filters=num_filters[stage], strides=1,
                           activation=None, is_trainable=is_trainable)
            if stage > 0 and res_block == 0:  # first layer but not first stage
                # linear projection residual shortcut connection to match
                # changed dims
                x = conv_layer(inputs=x, num_filters=num_filters[stage], kernel_size=1, strides=strides,
                                 activation=None, is_trainable=is_trainable)
            x = keras.layers.add([x, y])
            x = Activation('relu', trainable=is_trainable)(x)
    # Add classifier on top.
    x = GlobalAveragePooling2D()(x)
    feature_outputs = x
    if not regression:
        class_outputs = Dense(dim_output, activation='softmax', kernel_initializer='he_normal')(feature_outputs)
    else:
        class_outputs = Dense(dim_output, activation=None, kernel_initializer='he_normal')(feature_outputs)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=class_outputs)
    feature_model_base = (inputs, feature_outputs)
    return model, feature_model_base


def feature_based_learner(feature_model_base, num_classes):
    inputs = feature_model_base[0]
    feature_outputs = feature_model_base[1]
    class_outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(feature_outputs)
    model = Model(inputs=inputs, outputs=class_outputs)
    return model


def data_augmentation_generator():

    datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                 samplewise_center=False,  # set each sample mean to 0
                                 featurewise_std_normalization=False,  # divide inputs by std of dataset
                                 samplewise_std_normalization=False,  # divide each input by its std
                                 zca_whitening=False,  # apply ZCA whitening
                                 zca_epsilon=1e-06,  # epsilon for ZCA whitening
                                 rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
                                 width_shift_range=0.1,  # randomly shift images horizontally
                                 height_shift_range=0.1,  # randomly shift images vertically
                                 shear_range=0.,  # set range for random shear
                                 zoom_range=0.,  # set range for random zoom
                                 channel_shift_range=0.,  # set range for random channel shifts
                                 fill_mode='nearest',  # set mode for filling points outside the input boundaries
                                 cval=0.,   # value used for fill_mode = "constant"
                                 horizontal_flip=True,  # randomly flip images
                                 vertical_flip=False,  # randomly flip images
                                 rescale=None,  # set rescaling factor (applied before any other transformation)
                                 preprocessing_function=None,  # set function that will be applied on each input
                                 data_format=None,  # image data format, either "channels_first" or "channels_last"
                                 validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)
    return datagen
