from blocks import downscale_conv2D, upscale_deconv2d
import tensorflow as tf


def build_generator():
    _input = tf.keras.Input(shape=[256, 256, 1], name='image_input')
    x = downscale_conv2D(_input, 64, strides=1, name='0')
    features = [x]
    for i, n_filters in enumerate([64, 128, 256, 512, 512, 512, 512]):
        x = downscale_conv2D(x, n_filters, name=str(i + 1))
        features.append(x)

    for i, n_filters in enumerate([512, 512, 512, 256, 128, 64, 64]):
        x = upscale_deconv2d(x, n_filters, name=str(i + 1))
        x = tf.keras.layers.Concatenate()([features[-(i + 2)], x])
    _output = tf.keras.layers.Conv2D(filters=3,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='output_conv2d',
                                     activation='tanh')(x)
    return tf.keras.Model(inputs=[_input],
                          outputs=[_output],
                          name='Generator')


def build_discriminator():
    _input = tf.keras.Input(shape=[256, 256, 4])
    x = downscale_conv2D(_input, 64, strides=2, name='0', use_bn=False)
    x = downscale_conv2D(x, 128, strides=2, name='1')
    x = downscale_conv2D(x, 256, strides=2, name='2')
    x = downscale_conv2D(x, 512, strides=1, name='3')
    _output = tf.keras.layers.Conv2D(filters=1,
                                     kernel_size=4,
                                     strides=1,
                                     padding='same',
                                     name='output_conv2d',
                                     activation=None)(x)
    return tf.keras.Model(inputs=[_input],
                          outputs=[_output],
                          name='Discriminator')
