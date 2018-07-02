import tensorflow as tf
import numpy as np

def residual_block(incoming, num_outputs, kernel_size, scope, data_format = 'NHWC'):
    conv1 = tf.contrib.layers.conv2d(
        incoming, num_outputs, kernel_size, scope=scope+'/conv1',
        data_format=data_format, activation_fn=None, biases_initializer=None)
    conv1_bn = tf.contrib.layers.batch_norm(
            conv1, decay=0.9, center=True, activation_fn=lrelu,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm1',
            data_format=data_format)
    conv2 = tf.contrib.layers.conv2d(
        conv1, num_outputs, kernel_size, scope=scope+'/conv2',
        data_format=data_format, activation_fn=None, biases_initializer=None)
    conv2_bn = tf.contrib.layers.batch_norm(
            conv2, decay=0.9, center=True, activation_fn=None,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm2',
            data_format=data_format)
    incoming += conv2_bn
    return tf.nn.relu(incoming)

def kl_loss(mean, stddev, epsilon=1e-8):
    loss = tf.reduce_sum(0.5*(
        tf.square(mean) + tf.square(stddev) -
        tf.scalar_mul(2.0, tf.log(stddev+epsilon))-1.0))
    return loss/mean.shape.num_elements()

def l2_loss(x, y):
    return tf.losses.mean_squared_error(x, y, scope='l2_loss')

def l1_loss(x, y):
    return tf.losses.absolute_difference(x, y, scope='l1_loss')


def msssim_loss(x, y):
    return l2_loss(x, y)

def lrelu(inputs, name='lrelu'):
    return tf.maximum(inputs, 0.3*inputs, name=name)

def fully_conc(z, dim, scope):
    return tf.contrib.layers.fully_connected(
        z, dim, activation_fn=lrelu, biases_initializer=None,
        scope=scope+'/fully')

def conv2d(inputs, num_outputs, kernel_size, stride, scope, norm=True, ac_fn = lrelu, 
           d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope, stride=stride,
        data_format=d_format, activation_fn=None, biases_initializer=None)
    if norm:
        outputs = tf.contrib.layers.batch_norm(
            outputs, decay=0.9, center=True, activation_fn=ac_fn,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
            data_format=d_format)
    else:
        outputs = lrelu(outputs, name=scope+'/relu')
    return outputs

def deconv(inputs, out_num, kernel_size, scope, activation_fn = lrelu, d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d_transpose(
        inputs, out_num, kernel_size, scope=scope, stride=[2, 2],
        data_format=d_format, activation_fn=None, biases_initializer=None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=activation_fn, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)

def dilated_conv(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    axis = (d_format.index('H'), d_format.index('W'))
    conv0 = conv2d(inputs, out_num, kernel_size, 1, scope+'/conv0', False)
    conv1 = conv2d(conv0, out_num, kernel_size, 1, scope+'/conv1', False)
    dilated_conv0 = dilate_tensor(conv0, axis, 0, 0, scope+'/dialte_conv0')
    dilated_conv1 = dilate_tensor(conv1, axis, 1, 1, scope+'/dialte_conv1')
    conv1 = tf.add(dilated_conv0, dilated_conv1, scope+'/add1')
    with tf.variable_scope(scope+'/conv2'):
        shape = list(kernel_size) + [out_num, out_num]
        weights = tf.get_variable(
            'weights', shape, initializer=tf.truncated_normal_initializer())
        weights = tf.multiply(weights, get_mask(shape, scope))
        strides = [1, 1, 1, 1]
        conv2 = tf.nn.conv2d(conv1, weights, strides, padding='SAME',
                             data_format=d_format)
    outputs = tf.add(conv1, conv2, name=scope+'/add2')
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.elu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)

def get_mask(shape, scope):
    new_shape = (shape[0]*shape[1], shape[2], shape[3])
    mask = np.ones(new_shape, dtype=np.float32)
    for i in range(0, new_shape[0], 2):
        mask[i, :, :] = 0
    mask = np.reshape(mask, shape, 'F')
    return tf.constant(mask, dtype=tf.float32, name=scope+'/mask')

def dilate_tensor(inputs, axis, row_shift, column_shift, scope):
    rows = tf.unstack(inputs, axis=axis[0], name=scope+'/rowsunstack')
    row_zeros = tf.zeros(
        rows[0].shape, dtype=tf.float32, name=scope+'/rowzeros')
    for index in range(len(rows), 0, -1):
        rows.insert(index-row_shift, row_zeros)
    row_outputs = tf.stack(rows, axis=axis[0], name=scope+'/rowsstack')
    columns = tf.unstack(
        row_outputs, axis=axis[1], name=scope+'/columnsunstack')
    columns_zeros = tf.zeros(
        columns[0].shape, dtype=tf.float32, name=scope+'/columnzeros')
    for index in range(len(columns), 0, -1):
        columns.insert(index-column_shift, columns_zeros)
    column_outputs = tf.stack(
        columns, axis=axis[1], name=scope+'/columnsstack')
    return column_outputs

def pool2d(inputs, kernel_size, scope, data_format='NHWC'):
    return tf.contrib.layers.max_pool2d(
        inputs, kernel_size, scope=scope, padding='SAME',
        data_format=data_format)