import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from scipy.misc import imsave
import ops
from ops import *

class UVAE(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        self.data_format = 'NHWC'
        self.axis, self.channel_axis = (1, 2), 3
        self.input_shape = [
            conf.batch, conf.height, conf.width, conf.channel]
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sample_dir):
            os.makedirs(conf.sample_dir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def configure_networks(self):
        self.build_network()
        variables = tf.trainable_variables()
        self.vae_vars = [var for var in variables if var.name.startswith('VAE')]
        self.super_vars = [var for var in variables if var.name.startswith('SUPER')]
        optimizer_g = tf.train.AdamOptimizer(self.conf.learning_rate, beta1 = 0.5)
        self.train_opg = optimizer_g.minimize(self.loss_opg, name='VAE/train_opg')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.valid = tf.placeholder(tf.bool)
        self.inputs = tf.placeholder(tf.float32, self.input_shape)
        with tf.variable_scope('VAE') as scope:
            self.mean, self.stddev = self.encoder(self.inputs, self.valid)
            self.predictions = self.decoder(self.mean, self.stddev)
        with tf.variable_scope('SUPER') as scope:
            self.super_predictions = self.super_resolution(self.predictions, 'super')
        self.kl_loss = self.get_kl_loss()
        self.rec_loss_l = self.get_rec_loss(self.predictions, self.inputs,'low')
        self.rec_loss_h = l1_loss(self.super_predictions, self.inputs)
        self.loss_opg = self.kl_loss + self.rec_loss_l + self.rec_loss_h

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/kl_loss', self.kl_loss))
        summarys.append(tf.summary.scalar(name+'/rec_loss_l', self.rec_loss_l))
        summarys.append(tf.summary.scalar(name+'/rec_loss_h', self.rec_loss_h))
        if name == 'valid':
            summarys.append(tf.summary.image(
                name+'/lprediction',self.predictions, 20))
            summarys.append(tf.summary.image(
                name+'/hprediction', self.super_predictions, 20))
        summary = tf.summary.merge(summarys)
        return summary

    def super_resolution(self, inputs, scope):
        conv1 = conv2d(inputs, 64, 5, 1, scope+'/conv1', False)
        residual1 = residual_block(conv1, 64, 3, scope+'/residual1')
        residual2 = residual_block(residual1, 64, 3, scope+'/residual2')
        residual3 = residual_block(residual2, 64, 3, scope+'/residual3')
        residual4 = residual_block(residual3, 64, 3, scope+'/residual4')
        residual5 = residual_block(residual4, 64, 3, scope+'/residual5')
        conv2 = conv2d(residual5, 64, 3, 1, scope+'/conv2', True)
        conv2 = conv2 + conv1
        conv3 = conv2d(conv2, 256, 3, 1, scope+'/conv3', True)
        conv5 = conv2d(conv3, 64, 3, 1, scope+'/conv5', False)
        conv6 = conv2d(conv5, 3, 3, 1, scope+'/conv6', False)
        result = tf.nn.tanh(conv6)
        return result
        
    def discriminator(self, inputs):
        conv1_1 = conv2d(inputs, 64, 3, 1, '/conv1_1')
        conv1_2 = conv2d(conv1_1, 64, 3, 2, '/conv1_2')
        conv2_1 = conv2d(conv1_2, 128, 3, 1, '/conv2_1')
        conv2_2 = conv2d(conv2_1, 128, 3, 2, '/conv2_2')
        conv3_1 = conv2d(conv2_2, 256, 3, 1, '/conv3_1')
        conv3_2 = conv2d(conv2_1, 256, 3, 2, '/conv3_2')
        conv4_1 = conv2d(conv3_2, 512, 3, 1, '/conv4_1')
        conv4_2 = conv2d(conv4_1, 512, 3, 2, '/conv4_2')
        linear1 = fully_conc(conv4_2, 1024, '/fully_1')
        linear2 = fully_conc(linear1, 1, '/fully_2')
        return linear2

    def decoder(self, mean, stddev):
        z = tf.random_normal(
            [self.conf.batch, self.conf.latent_len], name='z/z')
        z = tf.multiply(z, stddev, name='z/mul')
        z = tf.add(z, mean, name='z/add')
        shape = self.up_outputs[0].shape
        dim = int(shape.num_elements()/shape[0].value)
        outputs = ops.fully_conc(z, dim, 'dec_fully')
        outputs = tf.reshape(outputs, shape, 'dec_fully/dec_reshape')
        self.up_outputs = [outputs]
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index == 0 else False
            name = 'dec_up%s' % layer_index
            outputs = self.construct_up_block(outputs, name, is_final)
            self.up_outputs.append(outputs)
        return outputs

    def encoder(self, inputs, valid):
        outputs = self.inputs
        self.up_outputs = [self.inputs]
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'enc_down%s' % layer_index
            outputs = self.construct_down_block(outputs, name, is_first)
            self.up_outputs.insert(0, outputs)
        outputs = tf.contrib.layers.flatten(outputs, scope='enc_fully/flat')
        outputs = ops.fully_conc(outputs, self.conf.latent_len*2, 'enc_fully')
        with tf.variable_scope('mean_std'):
            mean, stddev = tf.split(outputs, 2, axis=1, name='enc_split')
            stddev = tf.sqrt(
                tf.scalar_mul(4, tf.sigmoid(stddev, name='sigmoid')),
                name='mean_std/sqrt')
            mean = tf.where(
                self.valid, tf.zeros_like(mean, name='mean_zeros'),
                mean, name='mean_std/mean_where')
            stddev = tf.where(
                self.valid, tf.ones_like(stddev, name='stddev_ones'),
                stddev, name='mean_std/stddev_where')
        return mean, stddev

    def construct_down_block(self, inputs, name, first=False):
        num_outputs = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        inputs = ops.conv2d(
            inputs, num_outputs, self.conv_size, 1, name+'/conv1')
        inputs = ops.conv2d(
            inputs, num_outputs, self.conv_size, 2, name+'/conv2')
        return inputs

    def construct_up_block(self, inputs, name, final=False):
        num_outputs = inputs.shape[self.channel_axis].value
        num_outputs = self.conf.channel if final else int(num_outputs/2)
        activation_fn = tf.nn.tanh if final else lrelu
        inputs = self.deconv_func()(
            inputs, num_outputs, self.conv_size, name+'/conv1', activation_fn)
        return inputs

    def get_kl_loss(self):
        with tf.variable_scope('loss/kl_loss'):
            loss = ops.kl_loss(self.mean, self.stddev)
        return loss

    def get_rec_loss(self, predictions, targets, scope):
        with tf.variable_scope('loss/'+scope):
            loss = getattr(ops, self.conf.loss_name)(
                targets, predictions)
            loss = tf.scalar_mul(self.conf.sigma, loss)
        return loss

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_epoch > 0:
            self.reload(self.conf.reload_epoch)
        data_reader = H5DataLoader('../../Data/data/celeba_train_test.h5')
        epoch_num = 1
        epoch = 0
        while epoch < self.conf.max_epoch:
            if epoch_num % self.conf.test_step == 1:
                inputs = np.zeros(self.input_shape)
                feed_dict = {self.inputs: inputs, self.valid: True}
                summary = self.sess.run(
                    self.valid_summary, feed_dict=feed_dict)
                self.save_summary(summary, epoch_num)
            elif epoch_num % self.conf.summary_step == 1:
                targets = data_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: targets, self.valid: False}
                _, summary = self.sess.run(
                    [self.train_opg, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num)
            else:
                targets = data_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: targets, self.valid: False}
                rec_loss_l, kl_loss, _ = self.sess.run(
                    [self.rec_loss_l,self.kl_loss, self.train_opg],
                    feed_dict=feed_dict)
                print('epoch = ', epoch ,'----training loss l=', rec_loss_l, ', kl= ', kl_loss)
            if epoch_num % self.conf.save_step == 0:
                self.save(epoch_num)
            epoch = data_reader.epoch
            epoch_num += 1

    def predict(self):
        print('---->predicting', self.conf.test_epoch)
        if self.conf.test_epoch > 0:
            self.reload(self.conf.test_epoch)
        else:
            print("please set a reasonable test_epoch")
            return
        inputs = np.zeros(self.input_shape)
        feed_dict = {self.inputs: inputs, self.valid: True}
        img_l, img_h = self.sess.run([self.predictions, self.super_predictions], feed_dict=feed_dict)
        for i in range(img_h.shape[0]):
                imsave(self.conf.sample_dir +
                       str(i)+'h.png',img_h[i])
        
    def output_train(self):
        print('---->output', self.conf.test_epoch)
        if self.conf.test_epoch > 0:
            self.reload(self.conf.test_epoch)
        else:
            print("please set a reasonable test_epoch")
            return
        data_reader = H5DataLoader('../../Data/data/celeba_train_test.h5')
        targets = data_reader.next_batch(self.conf.batch)
        feed_dict = {self.inputs: targets, self.valid: False}
        predictions,super_predictions = self.sess.run(
                [self.predictions,self.super_predictions],
                feed_dict=feed_dict)
        super_predictions = np.array(super_predictions)
        predictions = np.array(predictions)
        for i in range(predictions.shape[0]):
            imsave(self.conf.sample_dir + str(i) + '_o.png', np.reshape(targets[i], (128,128,3)))
            #imsave(self.conf.sample_dir + str(i) + 'l.png', np.reshape(predictions[i], (128,128,3)))
            imsave(self.conf.sample_dir + str(i) + '_h.png', np.reshape(super_predictions[i], (128,128,3)))


    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)