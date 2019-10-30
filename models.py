import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import urllib
import os
import tarfile
import skimage
import skimage.io
import skimage.transform



def shared_encoder(x, name='feat_ext', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
        
            scope.reuse_variables()
        with slim.arg_scope(
              [slim.conv2d, slim.fully_connected],
              weights_regularizer=slim.l2_regularizer(1e-6),
              activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm):
            net = slim.conv2d(x, 32, [5, 5],scope = 'conv1_shared_encoder')
            net = slim.max_pool2d(net, [2, 2], scope='pool1_shared_encoder')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2_shared_encoder')
            net = slim.max_pool2d(net, [2, 2], scope='pool2_shared_encoder')
            net = slim.flatten(net, scope='flat_shared_encoder')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 100, scope='shared_fc1')
    return net

#Private Target Encoder
def private_target_encoder(x, name='priviate_target_encoder', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        with slim.arg_scope(
              [slim.conv2d, slim.fully_connected],
              weights_regularizer=slim.l2_regularizer(1e-6),
              activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm):
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2],2, scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2],2, scope='pool2')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 100, scope='private_target_fc1')
    return net

#Private Source Encoder
def private_source_encoder(x, name='priviate_source_encoder', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        with slim.arg_scope(
              [slim.conv2d, slim.fully_connected],
              weights_regularizer=slim.l2_regularizer(1e-6),
              activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm):
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 100, scope='private_source_fc1')
    return net

def shared_decoder(feat,height,width,channels,reuse=False, name='shared_decoder'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        with slim.arg_scope(
              [slim.conv2d, slim.fully_connected],
              weights_regularizer=slim.l2_regularizer(1e-6),
              activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm):
            net = slim.fully_connected(feat, 600, scope='fc1_decoder')
            net = tf.reshape(net, [-1, 10, 10, 6])
        
            net = slim.conv2d(net, 32, [5, 5], scope='conv1_1_decoder')
        
            net = tf.image.resize_nearest_neighbor(net, (16, 16))
        
            net = slim.conv2d(net, 32, [5, 5], scope='conv2_1_decoder')
        
            net = tf.image.resize_nearest_neighbor(net, (32, 32))
        
            net = slim.conv2d(net, 32, [5, 5], scope='conv3_2_decoder')
        
            output_size = [height, width]
            net = tf.image.resize_nearest_neighbor(net, output_size)
        
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
              net = slim.conv2d(net, channels, activation_fn=None, scope='conv4_1_decoder')
    return net
