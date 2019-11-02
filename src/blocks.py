import tensorflow as tf


def downscale_conv2D(tensor,
                     n_filters,
                     kernel_size=4,
                     strides=2,
                     name=None,
                     use_bn=True):
    _x = tf.keras.layers.Conv2D(filters=n_filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                use_bias=False,
                                name='downscale_block_' + name + '_conv2d',
                                activation=None)(tensor)
    if use_bn:
        _x = tf.keras.layers.BatchNormalization(
            name='downscale_block_' + name + '_bn')(_x)
    _x = tf.keras.layers.LeakyReLU(
        alpha=0.2, name='downscale_block_' + name + '_lrelu')(_x)
    return _x


def upscale_deconv2d(tensor,
                     n_filters,
                     kernel_size=4,
                     strides=2,
                     name=None):
    _x = tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding='same',
                                         use_bias=False,
                                         name='upscale_block_' +
                                         name + '_conv2d',
                                         activation=None)(tensor)
    _x = tf.keras.layers.BatchNormalization(
        name='upscale_block_' + name + '_bn')(_x)
    _x = tf.keras.layers.ReLU(name='upscale_block_' + name + '_relu')(_x)
    return _x
