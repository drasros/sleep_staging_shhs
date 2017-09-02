import tensorflow as tf

import tensorflow.contrib.layers as lyr
from tensorflow.python.ops import init_ops

# TODO: add batch normalisation, layer normalisation and COMPARE
# REM: batch normalisation not suitable with WGAN-GP

# leaky relu activations
def lrelu(x, leak=0.1, name='lrelu'):
    return tf.maximum(x, leak*x)

def dense(x, num_units, reuse=False, nonlinearity=None,
          normalizer="", databased_init=True, init_g=1., bn_training=True, scope=""):

    # data-dependent initialization with weightnorm
    # in original OpenAI code the authors have an init=True option which they run ONLY ONCE (with bigger batch size)
    # this implementation is slightly different, no need to initialize separately if batch size is sufficient,
    # we just use the same batch size and the first batch
    # REM: a feed_dict with data will need to be fed to the init op.
    # if not using databased init, g is initialized to init_g and b to zeros.
    # REM 2: with data dependent init, a 'smart initializer' is required to make sure variables are initialized
    # in the right order

    # normalizer: None, or "weightnorm", or "layernorm"
    # databased init: True or False
    # init_g: if databased_unit is False, scale of g initializer. If databased_init is True, scale g by this after initialized.

    # note: normalization before or after nonlinearity? what is commonly done is:
    # BATCH NORM AFTER NONLINEARITY
    # WEIGHTNORM BEFORE NONLINEARITY
    # LAYERNORM BEFORE NL

    assert normalizer in ["", "weightnorm", "layernorm", "batchnorm"]

    with tf.variable_scope(scope, reuse=reuse):
        V = tf.get_variable('V', [int(x.get_shape()[1]), num_units], tf.float32,
                            tf.random_normal_initializer(0, 0.05))

        if normalizer in ["", "layernorm", "batchnorm"]:#is "" or normalizer is "layernorm":
            b = tf.get_variable('b', [1, num_units],
                                initializer=tf.constant_initializer(0.), dtype=tf.float32)
            x = tf.matmul(x, V) + b
            if normalizer == "layernorm":
                mean, variance = tf.nn.moments(x, [-1], keep_dims=True)
                x = tf.nn.batch_normalization(x, mean, variance, None, None, 1e-10)
                if nonlinearity is not None:
                    x = nonlinearity(x)
            if normalizer == "batchnorm":
                # nonlinearity before BN
                if nonlinearity is not None:
                    x = nonlinearity(x)
                x = tf.layers.batch_normalization(x, training=bn_training)

        if normalizer == "weightnorm":
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x = tf.matmul(x, V_norm)
            if databased_init is True:
                m_init, v_init = tf.nn.moments(x, [0], keep_dims=True)
                scale_init = init_g / tf.sqrt(1e-10 + v_init)
                g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init)
                b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init)
            else:
                g = tf.get_variable('g', [1, num_units],
                                    initializer=tf.constant_initializer(init_g), dtype=tf.float32)
                b = tf.get_variable('b', [1, num_units],
                                    initializer=tf.constant_initializer(0.), dtype=tf.float32)
            x = tf.multiply(x, g) + b
            if nonlinearity is not None:
                x = nonlinearity(x)

        return x


def conv2d(x, num_filters, reuse, filter_size=[3, 3],
           stride=[1, 1], pad='SAME', nonlinearity=None,
           normalizer="", databased_init=True, init_g=1., bn_training=True, scope=""):

    # data-dependent initialization with weightnorm
    # in original OpenAI code the authors have an init=True option which they run ONLY ONCE (with bigger batch size)
    # this implementation is slightly different, no need to initialize separately if batch size is sufficient,
    # we just use the same batch size and the first batch
    # REM: a feed_dict with data will need to be fed to the init op.
    # if not using databased init, g is initialized to init_g and b to zeros.
    # REM 2: with data dependent init, a 'smart initializer' is required to make sure variables are initialized
    # in the right order
    # REM 3: Weightnorm is not as useful as BN for CNNs. But let's try it anyways (with WGANGP cannot use BN)

    # Layernorm:
    # NOTE: WHICH NORMALISATION AXIS ??? HERE [1, 2, 3] = all except batch (same effect as in weight norm)

    # normalizer: None, or "weightnorm", or "layernorm"
    # databased init: True or False
    # init_g: if databased_unit is False, scale of g initializer. If databased_init is True, scale g by this after initialized.

    assert normalizer in ["", "weightnorm", "layernorm", "batchnorm"]

    with tf.variable_scope(scope, reuse=reuse):
        V = tf.get_variable('V', filter_size+[int(x.get_shape()[-1]),num_filters],
                            tf.float32, tf.random_normal_initializer(0, 0.05))

        if normalizer in ["", "layernorm", "batchnorm"]:
            b = tf.get_variable('b', [1, 1, 1, num_filters],
                                initializer=tf.constant_initializer(0.), dtype=tf.float32)
            x = tf.nn.conv2d(x, V, [1]+stride+[1], pad) + b
            if normalizer == "layernorm":
                mean, variance = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
                x = tf.nn.batch_normalization(x, mean, variance, None, None, 1e-10)
                if nonlinearity is not None:
                    x = nonlinearity(x)
            if normalizer == "batchnorm":
                if nonlinearity is not None:
                    x = nonlinearity(x)
                x = tf.layers.batch_normalization(x, axis=-1, training=bn_training)

        if normalizer == "weightnorm":
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
            if databased_init is True:
                m_init, v_init = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
                scale_init = init_g / tf.sqrt(1e-10 + v_init)
                g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init)
                b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init)
            else:
                g = tf.get_variable('g', [1, 1, 1, num_filters],
                                    initializer=tf.constant_initializer(init_g), dtype=tf.float32)
                b = tf.get_variable('b', [1, 1, 1, num_filters],
                                    initializer=tf.constant_initializer(0.), dtype=tf.float32)
            x = tf.multiply(x, g) + b
            if nonlinearity is not None:
                x = nonlinearity(x)
        return x

