from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

from ops import *

#######################################################################################################################
################################################## Custom (New BN) ##################################################
#######################################################################################################################
def create_vae_individual_encoder(input, is_training, a):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("vae_individual_encoder_1"):
       output = gen_conv(input, a.nsef, a)
       layers.append(output)

    layer_specs = [
        a.nsef * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.nsef * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.nsef * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.nsef * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        # Latent representation has 8x8 dimensionality
    ]


    for out_channels in layer_specs:
        with tf.variable_scope("vae_individual_encoder_%d" % (len(layers) + 1)):
            bn = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(bn, 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, a)
            layers.append(convolved)

    # Shared part of representation uses CNN layer
    output = layers[-1]
    ch_size = output.get_shape().as_list()[-1] // 2
    r_z_shared_mean, r_z_shared_logvar = tf.split(output, [ch_size, ch_size], axis=-1)

    # Exclusive part of representation uses FC layer
    with tf.variable_scope("vae_encoder_q_z_exclusive"):
        rinput = tf.reshape(rectified, [-1, 16*16*8*a.nsef])
        q_z_exclusive = gen_fc(rinput, out_channels = a.exdim * 2)
        q_z_exclusive_mean, q_z_exclusive_logvar = tf.split(q_z_exclusive, [a.exdim, a.exdim], axis=-1)

    return q_z_exclusive_mean, q_z_exclusive_logvar, r_z_shared_mean, r_z_shared_logvar, rectified


def create_vae_shared_encoder(input1, input2, a):
    layers = []

    # encoder_1: [batch, 16, 16, ngf * 8] => [batch, 128, 128, ngf]
    with tf.variable_scope("vae_encoder_q_z_shared_1"):
       # output = gen_conv(tf.concat([input1, input2], axis=-1), a.ngf*8, a)
       output = gen_conv(tf.concat([input1, input2], axis=-1), a.nsef * 8, a)
       layers.append(output)

    ch_size = output.get_shape().as_list()[-1] // 2
    q_z_shared_mean, q_z_shared_logvar = tf.split(output, [ch_size, ch_size], axis=-1)

    return q_z_shared_mean, q_z_shared_logvar


############################## Custom (No Param Sharing) ##############################
def create_vae_individual_shared_encoder(input, is_training, a):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, nsef]
    with tf.variable_scope("vae_individual_shared_encoder_1"):
       output = gen_conv(input, a.nsef, a)
       layers.append(output)

    layer_specs = [
        a.nsef * 2, # encoder_2: [batch, 128, 128, nsef] => [batch, 64, 64, nsef * 2]
        a.nsef * 4, # encoder_3: [batch, 64, 64, nsef * 2] => [batch, 32, 32, nsef * 4]
        a.nsef * 8, # encoder_4: [batch, 32, 32, nsef * 4] => [batch, 16, 16, nsef * 8]
        a.nsef * 8, # encoder_5: [batch, 16, 16, nsef * 8] => [batch, 8, 8, nsef * 8]
        # Latent representation has 8x8 dimensionality
    ]


    for out_channels in layer_specs:
        with tf.variable_scope("vae_individual_shared_encoder_%d" % (len(layers) + 1)):
            bn = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(bn, 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, a)
            layers.append(convolved)

    # Shared part of representation uses CNN layer
    output = layers[-1]
    ch_size = output.get_shape().as_list()[-1] // 2
    r_z_shared_mean, r_z_shared_logvar = tf.split(output, [ch_size, ch_size], axis=-1)
    return r_z_shared_mean, r_z_shared_logvar, rectified


def create_vae_individual_exclusive_encoder(input, is_training, a):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, neef]
    with tf.variable_scope("vae_individual_exclusive_encoder_1"):
       output = gen_conv(input, a.neef, a)
       layers.append(output)

    layer_specs = [
        a.neef * 2, # encoder_2: [batch, 128, 128, neef] => [batch, 64, 64, neef * 2]
        a.neef * 4, # encoder_3: [batch, 64, 64, neef * 2] => [batch, 32, 32, neef * 4]
        a.neef * 8, # encoder_4: [batch, 32, 32, neef * 4] => [batch, 16, 16, neef * 8]
        # a.neef * 8, # encoder_5: [batch, 16, 16, neef * 8] => [batch, 8, 8, neef * 8]
        # Latent representation has 8x8 dimensionality
    ]


    for out_channels in layer_specs:
        with tf.variable_scope("vae_individual_exclusive_encoder_%d" % (len(layers) + 1)):
            bn = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(bn, 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, a)
            layers.append(convolved)

    # Exclusive part of representation uses FC layer
    with tf.variable_scope("vae_encoder_q_z_exclusive"):
        bn = batchnorm(layers[-1], is_training=is_training, momentum=a.bn_momentum)
        rectified = lrelu(bn, 0.2)
        rinput = tf.reshape(rectified, [-1, 16*16*8*a.neef])
        q_z_exclusive = gen_fc(rinput, out_channels = a.exdim * 2)
        q_z_exclusive_mean, q_z_exclusive_logvar = tf.split(q_z_exclusive, [a.exdim, a.exdim], axis=-1)

    return q_z_exclusive_mean, q_z_exclusive_logvar


###################################################################################################################
################################################## Custom (FLAT) ##################################################
###################################################################################################################
def create_vae_individual_encoder_flat(input, is_training, a):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("vae_individual_encoder_1"):
       output = gen_conv(input, a.nsef, a)
       layers.append(output)

    layer_specs = [
        a.nsef * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.nsef * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.nsef * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.nsef * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        # Latent representation has 8x8 dimensionality
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("vae_individual_encoder_%d" % (len(layers) + 1)):
            output = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(output, 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, a)
            layers.append(convolved)


    # Exclusive part of representation uses FC layer
    with tf.variable_scope("vae_encoder_q_z_exclusive"):
        rinput = tf.reshape(rectified, [-1, 16 * 16 * a.nsef*8])
        q_z_exclusive = gen_fc(rinput, out_channels=a.exdim*2)
        q_z_exclusive_mean, q_z_exclusive_logvar = tf.split(q_z_exclusive, [a.exdim, a.exdim], axis=-1)


    # shared part: Double
    with tf.variable_scope("vae_encoder_r_z_shared_1"):
        bn = batchnorm(layers[-1], is_training=is_training, momentum=a.bn_momentum)
        rectified = lrelu(bn, 0.2)
        rinput = tf.reshape( rectified, [-1, 8 * 8 * a.nsef*8] )
        r_z_shared_1 = gen_fc(rinput, out_channels=a.shdim * 2)

    with tf.variable_scope("vae_encoder_r_z_shared_2"):
        bn = batchnorm(r_z_shared_1, is_training=is_training, momentum = a.bn_momentum)
        rectified = lrelu(bn, 0.2)
        r_z_shared_2 = gen_fc(rectified, out_channels=a.shdim * 2)

    r_z_shared_mean, r_z_shared_logvar = tf.split(r_z_shared_2, [a.shdim, a.shdim], axis=-1)

    return q_z_exclusive_mean, q_z_exclusive_logvar, r_z_shared_mean, r_z_shared_logvar, rinput


def create_vae_shared_encoder_flat(input1, input2, is_training, a):
    layers = []

    ### Double
    # encoder_1: [batch, 16, 16, ngf * 8] => [batch, 128, 128, ngf]
    with tf.variable_scope("vae_encoder_q_z_shared_1"):
        q_z_shared_1 = gen_fc(tf.concat([input1, input2], axis=-1), out_channels=a.shdim * 2)

    with tf.variable_scope("vae_encoder_q_z_shared_2"):
        bn = batchnorm(q_z_shared_1, is_training=is_training , momentum = a.bn_momentum)
        rectified = lrelu(bn, 0.2)
        q_z_shared_2 = gen_fc(rectified, out_channels=a.shdim * 2)

    q_z_shared_mean, q_z_shared_logvar = tf.split(q_z_shared_2, [a.shdim, a.shdim], axis=-1)

    return q_z_shared_mean, q_z_shared_logvar


def create_vae_individual_shared_encoder_flat(input, is_training, a):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("vae_individual_shared_encoder_1"):
       output = gen_conv(input, a.nsef, a)
       layers.append(output)

    layer_specs = [
        a.nsef * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.nsef * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.nsef * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.nsef * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        # Latent representation has 8x8 dimensionality
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("vae_individual_shared_encoder_%d" % (len(layers) + 1)):
            output = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(output, 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, a)
            layers.append(convolved)

    # shared part: Double
    with tf.variable_scope("vae_encoder_r_z_shared_1"):
        bn = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
        rectified = lrelu(bn, 0.2)
        rinput = tf.reshape( rectified, [-1, 8 * 8 * a.nsef*8] )
        r_z_shared_1 = gen_fc(rinput, out_channels=a.shdim * 2)

    with tf.variable_scope("vae_encoder_r_z_shared_2"):
        bn = batchnorm(r_z_shared_1, is_training=is_training, momentum = a.bn_momentum)
        rectified = lrelu(bn, 0.2)
        r_z_shared_2 = gen_fc(rectified, out_channels=a.shdim * 2)

    r_z_shared_mean, r_z_shared_logvar = tf.split(r_z_shared_2, [a.shdim, a.shdim], axis=-1)

    return r_z_shared_mean, r_z_shared_logvar, rinput


def create_vae_individual_exclusive_encoder_flat(input, is_training, a):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("vae_individual_exclusive_encoder_1"):
       output = gen_conv(input, a.nsef, a)
       layers.append(output)

    layer_specs = [
        a.nsef * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.nsef * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.nsef * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        # a.nsef * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        # Latent representation has 8x8 dimensionality
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("vae_individual_exclusive_encoder_%d" % (len(layers) + 1)):
            output = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(output, 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, a)
            layers.append(convolved)


    # Exclusive part of representation uses FC layer
    with tf.variable_scope("vae_encoder_q_z_exclusive"):
        bn = batchnorm(layers[-1], is_training=is_training, momentum=a.bn_momentum)
        rectified = lrelu(bn, 0.2)
        rinput = tf.reshape(rectified, [-1, 16 * 16 * a.nsef*8])
        q_z_exclusive = gen_fc(rinput, out_channels=a.exdim*2)

    q_z_exclusive_mean, q_z_exclusive_logvar = tf.split(q_z_exclusive, [a.exdim, a.exdim], axis=-1)

    return q_z_exclusive_mean, q_z_exclusive_logvar



#######################################################################################################################
################################################## Custom: (ZS-SBIR) ##################################################
#######################################################################################################################

############################## Custom (3-layered Ex, 3-layered Shared w/ param sharing, 512) ##############################
def create_zs_vae_individual_exclusive_encoder(input, is_training, a):
    layers = []

    # encoder_1: [batch, 4096] => [batch, neef * 8]
    with tf.variable_scope("vae_individual_exclusive_encoder_1"):
       output = gen_fc(input, out_channels=a.neef * 4)
       layers.append(output)

    layer_specs = [
        a.neef * 2, # encoder_2: [batch, neef * 8] => [batch, neef * 4]
        a.exdim * 2, # encoder_3: [batch, neef * 4] => [batch, neef * 2]
    ]


    for out_channels in layer_specs:
        with tf.variable_scope("vae_individual_exclusive_encoder_%d" % (len(layers) + 1)):
            bn = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(bn, 0.2)
            # rectified = tf.nn.relu(bn)

            output = gen_fc(rectified, out_channels=out_channels)
            layers.append(output)

    output = layers[-1]
    ch_size = output.get_shape().as_list()[-1] // 2
    q_z_exclusive_mean, q_z_exclusive_logvar = tf.split(output, [ch_size, ch_size], axis=-1)

    return q_z_exclusive_mean, q_z_exclusive_logvar
    # return tf.nn.tanh(q_z_exclusive_mean), tf.nn.tanh(q_z_exclusive_logvar)


def create_zs_vae_individual_shared_encoder(input, is_training, a):
    layers = []
    rectified_layers = []

    # encoder_1: [batch, 4096] => [batch, nsef * 8]
    with tf.variable_scope("vae_individual_shared_encoder_1"):
       output = gen_fc(input, out_channels=a.nsef * 4)
       layers.append(output)

    layer_specs = [
        a.nsef * 2, # encoder_2: [batch, nsef * 8] => [batch, nsef * 4]
        a.shdim * 2, # encoder_3: [batch, nsef * 2] => [batch, nsef * 2]
    ]


    for out_channels in layer_specs:
        with tf.variable_scope("vae_individual_shared_encoder_%d" % (len(layers) + 1)):
            bn = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(bn, 0.2)
            # rectified = tf.nn.relu(bn)
            rectified_layers.append(rectified)

            output = gen_fc(rectified, out_channels=out_channels)
            layers.append(output)

    output = layers[-1]
    ch_size = output.get_shape().as_list()[-1] // 2
    r_z_shared_mean, r_z_shared_logvar = tf.split(output, [ch_size, ch_size], axis=-1)

    return r_z_shared_mean, r_z_shared_logvar, rectified_layers[0]
    # return tf.nn.tanh(r_z_shared_mean), tf.nn.tanh(r_z_shared_logvar), rectified_layers[0]


def create_zs_vae_shared_encoder(input1, input2, is_training, a):
    layers = []

    # encoder_1: [batch, nsef*8 + nsef*8] => [batch, nsef * 4]
    with tf.variable_scope("vae_shared_encoder_1"):
       output = gen_fc(tf.concat([input1, input2], axis=-1), out_channels=a.nsef*4)
       layers.append(output)

    layer_specs = [
        # a.nsef * 2, # encoder_2: [batch, nsef * 4] => [batch, nsef * 4]
        a.shdim * 2, # encoder_3: [batch, nsef * 4] => [batch, nsef * 2]
    ]


    for out_channels in layer_specs:
        with tf.variable_scope("vae_shared_encoder_%d" % (len(layers) + 1)):
            bn = batchnorm(layers[-1], is_training=is_training, momentum = a.bn_momentum)
            rectified = lrelu(bn, 0.2)
            # rectified = tf.nn.relu(bn)

            output = gen_fc(rectified, out_channels=out_channels)
            layers.append(output)

    output = layers[-1]
    ch_size = output.get_shape().as_list()[-1] // 2
    q_z_shared_mean, q_z_shared_logvar = tf.split(output, [ch_size, ch_size], axis=-1)

    return q_z_shared_mean, q_z_shared_logvar