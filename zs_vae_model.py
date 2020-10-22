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
from encoder import *
from decoder import *
# from decoderExclusive import *
# from discriminatorWGANGP import *



Model = collections.namedtuple("Model", "outputsX, outputsY,\
                                shared_mean,\
                                sR_X2Y, sR_Y2X,\
                                eR_X2Y, eR_Y2X,\
                                shared_logvar,\
                                sR_X2Y_logvar, sR_Y2X_logvar,\
                                eR_X2Y_logvar, eR_Y2X_logvar,\
                                recon_X_loss, recon_Y_loss, recon_C_loss,\
                                kl_X_loss, kl_Y_loss, kl_S_loss,\
                                kl_InterX_loss, kl_InterY_loss,\
                                joint_loss,\
                                train")

def create_zs_vae_model(inputsX, inputsY, inputsC, is_training, a):

    print('LAMBDA: ', a.LAMBDA, '    BETA: ', a.BETA)
    # Modify values if images are reduced
    IMAGE_SIZE = 256

    OUTPUT_DIM = IMAGE_SIZE*IMAGE_SIZE*3 # 256x256x3
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)


    ########## Img_Encoders ##########
    if a.separate_enc == "True":
        print("Separate Conv Encs")
        with tf.variable_scope("X_encoder"):
            q_z_x_mean, q_z_x_logvar = create_zs_vae_individual_exclusive_encoder(inputsX, is_training, a)
            r_z_x_mean, r_z_x_logvar, rectifiedX = create_zs_vae_individual_shared_encoder(inputsX, is_training, a)

        with tf.variable_scope("Y_encoder"):
            q_z_y_mean, q_z_y_logvar = create_zs_vae_individual_exclusive_encoder(inputsY, is_training, a)
            r_z_y_mean, r_z_y_logvar, rectifiedY = create_zs_vae_individual_shared_encoder(inputsY, is_training, a)

    with tf.variable_scope("S_encoder"):
        q_z_s_mean, q_z_s_logvar = create_zs_vae_shared_encoder(rectifiedX, rectifiedY, is_training, a)
        # q_z_s_mean, q_z_s_logvar = create_zs_vae_shared_encoder(inputsX, inputsY, is_training, a)



    # random noise for posterior distributions
    eps_x = tf.random_normal(q_z_x_logvar.shape)
    eps_y = tf.random_normal(q_z_y_logvar.shape)
    eps_s = tf.random_normal(q_z_s_logvar.shape)

    # random shared represenation for unconditional generation
    eps_s_common = tf.random_normal(q_z_s_logvar.shape)

    # sample z from posteriors (Think about train/test time...)
    z_x = q_z_x_mean
    z_y = q_z_y_mean
    z_s = q_z_s_mean


    # For numerical stability, make variance at most 10000.0963
    q_z_x_logvar = tf.math.minimum(q_z_x_logvar, 9.21035)
    q_z_y_logvar = tf.math.minimum(q_z_y_logvar, 9.21035)
    q_z_s_logvar = tf.math.minimum(q_z_s_logvar, 9.21035)
    r_z_x_logvar = tf.math.minimum(r_z_x_logvar, 9.21035)
    r_z_y_logvar = tf.math.minimum(r_z_y_logvar, 9.21035)


    if a.mode == "train":
        # (Make sure) stochasticity applies only to the training phase
        z_x += tf.exp(0.5 * q_z_x_logvar) * eps_x
        z_y += tf.exp(0.5 * q_z_y_logvar) * eps_y
        z_s += tf.exp(0.5 * q_z_s_logvar) * eps_s


    ########## Img_Decoders ##########
    # One copy of the decoder for the noise input, the second copy for the correct the cross-domain autoencoder
    with tf.name_scope("X_decoder_noise"):
        with tf.variable_scope("X_decoder"):
            out_channels = int(inputsX.get_shape()[-1])
            outputsX = create_zs_vae_decoder(z_s, z_x, out_channels, is_training, a)

    with tf.name_scope("Y_decoder_noise"):
        with tf.variable_scope("Y_decoder"):
            out_channels = int(inputsY.get_shape()[-1])
            outputsY = create_zs_vae_decoder(z_s, z_y, out_channels, is_training, a)

    with tf.name_scope("C_decoder_noise"):
        with tf.variable_scope("C_decoder"):
            out_channels = int(inputsC.get_shape()[-1])
            logits = create_zs_class_decoder(z_s, out_channels, is_training, a)

    ######### LOSSES
    with tf.name_scope("recon_X_loss"):
        # recon_X_loss = a.l1_weight*tf.reduce_mean(tf.abs(outputsX-inputsX))
        recon_X_loss = a.l1_weight * tf.reduce_mean((outputsX - inputsX)**2)

    with tf.name_scope("recon_Y_loss"):
        # recon_Y_loss = a.l1_weight*tf.reduce_mean(tf.abs(outputsY-inputsY))
        recon_Y_loss = a.l1_weight * tf.reduce_mean((outputsY - inputsY)**2)

    with tf.name_scope("recon_C_loss"):
        recon_C_loss = a.class_weight * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=inputsC, logits=logits))


    with tf.name_scope("kl_X_loss"):
        kl_X_loss = tf.reduce_mean(
                        tf.reduce_sum(0.5 * (-1.0 - q_z_x_logvar
                                             + q_z_x_mean**2 + tf.exp(q_z_x_logvar)),
                                      axis=1))

    with tf.name_scope("kl_Y_loss"):
        kl_Y_loss = tf.reduce_mean(
                        tf.reduce_sum(0.5 * (-1.0 - q_z_y_logvar
                                             + q_z_y_mean**2 + tf.exp(q_z_y_logvar)),
                                      axis=1))

    with tf.name_scope("kl_S_loss"):
        q_z_s_mean_flat = tf.reshape(q_z_s_mean, [a.batch_size, -1])
        q_z_s_logvar_flat = tf.reshape(q_z_s_logvar, [a.batch_size, -1])
        kl_S_loss = tf.reduce_mean(
                        tf.reduce_sum(0.5 * (-1.0 - q_z_s_logvar_flat + q_z_s_mean_flat**2 + tf.exp(q_z_s_logvar_flat)), axis=1))

    with tf.name_scope("kl_InterX_loss"):
        r_z_x_mean_flat = tf.reshape(r_z_x_mean, [a.batch_size, -1])
        r_z_x_logvar_flat = tf.reshape(r_z_x_logvar, [a.batch_size, -1])
        kl_InterX_loss = tf.reduce_mean(
                            tf.reduce_sum(
                                0.5 * (-1.0 - q_z_s_logvar_flat + r_z_x_logvar_flat
                                       + ((q_z_s_mean_flat - r_z_x_mean_flat) ** 2 + tf.exp(q_z_s_logvar_flat))/tf.exp(r_z_x_logvar_flat) ),
                                axis=1))

    with tf.name_scope("kl_InterY_loss"):
        r_z_y_mean_flat = tf.reshape(r_z_y_mean, [a.batch_size, -1])
        r_z_y_logvar_flat = tf.reshape(r_z_y_logvar, [a.batch_size, -1])
        kl_InterY_loss = tf.reduce_mean(
                            tf.reduce_sum(
                                0.5 * (-1.0 - q_z_s_logvar_flat + r_z_y_logvar_flat
                                       + ((q_z_s_mean_flat - r_z_y_mean_flat) ** 2 + tf.exp(q_z_s_logvar_flat))/tf.exp(r_z_y_logvar_flat) ),
                                axis=1))


    reg_coeff = tf.stop_gradient( 1.0 - tf.exp( -tf.cast(global_step, tf.float32) / a.ar) )
    with tf.name_scope("joint_loss"):
        joint_loss = (a.BETA + a.LAMBDA) * recon_X_loss + (a.BETA + a.LAMBDA) * recon_Y_loss \
                     + reg_coeff * (a.BETA + a.LAMBDA) * kl_X_loss \
                     + reg_coeff * (a.BETA + a.LAMBDA) * kl_Y_loss \
                     + reg_coeff * a.BETA * kl_S_loss \
                     + a.LAMBDA * kl_InterX_loss + a.LAMBDA * kl_InterY_loss \
                     + recon_C_loss


    ######### OPTIMIZERS

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope("joint_train"):
        with tf.control_dependencies(update_ops):
            joint_tvars = [var for var in tf.trainable_variables()]
            joint_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            joint_grads_and_vars = joint_optim.compute_gradients(joint_loss, var_list=joint_tvars)
            joint_train = joint_optim.apply_gradients(joint_grads_and_vars)


    # TODO: it has to be checked if ema affects training loss or not.
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([recon_X_loss, recon_Y_loss,
                               kl_X_loss, kl_Y_loss, kl_S_loss,
                               kl_InterX_loss, kl_InterY_loss,
                               recon_C_loss,
                               joint_loss])

    # global_step = tf.train.get_or_create_global_step()
    # incr_global_step = tf.assign(global_step, global_step+1)
    return Model(
        outputsX=outputsX,
        outputsY=outputsY,
        shared_mean=q_z_s_mean,
        sR_X2Y=r_z_x_mean,
        sR_Y2X=r_z_y_mean,
        eR_X2Y=q_z_x_mean,
        eR_Y2X=q_z_y_mean,
        shared_logvar=q_z_s_logvar,
        sR_X2Y_logvar=r_z_x_logvar,
        sR_Y2X_logvar=r_z_y_logvar,
        eR_X2Y_logvar=q_z_x_logvar,
        eR_Y2X_logvar=q_z_y_logvar,
        recon_X_loss=ema.average(recon_X_loss),
        recon_Y_loss=ema.average(recon_Y_loss),
        recon_C_loss=ema.average(recon_C_loss),
        kl_X_loss=ema.average(kl_X_loss),
        kl_Y_loss=ema.average(kl_Y_loss),
        kl_S_loss=ema.average(kl_S_loss),
        kl_InterX_loss=ema.average(kl_InterX_loss),
        kl_InterY_loss=ema.average(kl_InterY_loss),
        joint_loss=ema.average(joint_loss),
        train=tf.group(update_losses, incr_global_step, joint_train),
    )