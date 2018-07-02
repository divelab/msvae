import os
import time
import argparse
import tensorflow as tf
from model import UVAE

def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_epoch', 20, '# of step in an epoch')
    flags.DEFINE_integer('test_step', 100, '# of step to test a model')
    flags.DEFINE_integer('save_step', 1000, '# of step to save a model')
    flags.DEFINE_integer('summary_step', 2, '# of step to save the summary')
    flags.DEFINE_float('learning_rate', 2e-4, 'learning rate')
    flags.DEFINE_boolean('use_gpu', False, 'use GPU or not')
    # data
    flags.DEFINE_string('data_dir', './dataset/', 'Name of data directory')
    flags.DEFINE_integer('batch', 100, 'batch size')
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 128, 'height size')
    flags.DEFINE_integer('width', 128, 'width size')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('sample_dir', './trainsamples0/', 'Sample directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_epoch', 0, 'Reload epoch')
    flags.DEFINE_integer('test_epoch', 102000, 'Test epoch')
    flags.DEFINE_integer('test_size', 100, 'Test size')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network
    flags.DEFINE_integer('network_depth', 4, 'network depth for U-Net')
    flags.DEFINE_integer('start_channel_num', 64, 'start number of outputs')
    flags.DEFINE_integer('latent_len', 512, 'latent variable length')
    flags.DEFINE_float('sigma', 1, 'sigma value for reconstruction loss')
    flags.DEFINE_string(
        'loss_name', 'l2_loss', 'Use which conv op: l2_loss, l1_loss or msssim_loss')
    flags.DEFINE_string(
        'conv_name', 'conv2d', 'Use which conv op: conv2d or co_conv2d')
    flags.DEFINE_string(
        'deconv_name', 'deconv',
        'Use which deconv op: deconv, dilated_conv, co_dilated_conv')
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',help='actions: train, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'predict','output_train']:
        print('invalid action: ', args.action)
        print("Please input a action: train, or predict")
    else:
        model = UVAE(tf.Session(), configure())
        getattr(model, args.action)()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    tf.app.run()
