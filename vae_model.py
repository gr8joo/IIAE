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


Model = collections.namedtuple("Model", "outputsX, outputsY, outputsXp, outputsYp, outputsXpp, outputsYpp, outputsXppp, outputsYppp, outputsXun, outputsYun,\
                                outputsX_fromX, outputsX_fromY,\
                                outputsY_fromY, outputsY_fromX,\
                                shared_mean,\
                                sR_X2Y, sR_Y2X,\
                                eR_X2Y, eR_Y2X,\
                                shared_logvar,\
                                sR_X2Y_logvar, sR_Y2X_logvar,\
                                eR_X2Y_logvar, eR_Y2X_logvar,\
                                im_swapped_X,im_swapped_Y,\
                                sel_auto_X,sel_auto_Y,\
                                recon_X_loss, recon_Y_loss,\
                                kl_X_loss, kl_Y_loss, kl_S_loss,\
                                kl_InterX_loss, kl_InterY_loss,\
                                joint_loss,\
                                train")

def create_vae_model(inputsX, inputsY, is_training, a):

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
            if a.dataset_name == "cars":
                q_z_x_mean, q_z_x_logvar = create_vae_individual_exclusive_encoder_flat(inputsX, is_training, a)
                r_z_x_mean, r_z_x_logvar, rectifiedX = create_vae_individual_shared_encoder_flat(inputsX, is_training, a)
            else:
                q_z_x_mean, q_z_x_logvar = create_vae_individual_exclusive_encoder(inputsX, is_training, a)
                r_z_x_mean, r_z_x_logvar, rectifiedX = create_vae_individual_shared_encoder(inputsX, is_training, a)

        with tf.variable_scope("Y_encoder"):
            if a.dataset_name == "cars":
                q_z_y_mean, q_z_y_logvar = create_vae_individual_exclusive_encoder_flat(inputsY, is_training, a)
                r_z_y_mean, r_z_y_logvar, rectifiedY = create_vae_individual_shared_encoder_flat(inputsY, is_training,
                                                                                                 a)
            else:
                q_z_y_mean, q_z_y_logvar = create_vae_individual_exclusive_encoder(inputsY, is_training, a)
                r_z_y_mean, r_z_y_logvar, rectifiedY = create_vae_individual_shared_encoder(inputsY, is_training, a)

    else:
        print("Shared Conv Encs")
        with tf.variable_scope("X_encoder"):
            if a.dataset_name == "cars":
                q_z_x_mean, q_z_x_logvar, r_z_x_mean, r_z_x_logvar, rectifiedX = create_vae_individual_encoder_flat(inputsX, is_training, a)
            else:
                q_z_x_mean, q_z_x_logvar, r_z_x_mean, r_z_x_logvar, rectifiedX = create_vae_individual_encoder(inputsX, is_training, a)

        with tf.variable_scope("Y_encoder"):
            if a.dataset_name == "cars":
                q_z_y_mean, q_z_y_logvar, r_z_y_mean, r_z_y_logvar, rectifiedY = create_vae_individual_encoder_flat(inputsY, is_training, a)
            else:
                q_z_y_mean, q_z_y_logvar, r_z_y_mean, r_z_y_logvar, rectifiedY = create_vae_individual_encoder(inputsY, is_training, a)

    with tf.variable_scope("S_encoder"):
        if a.dataset_name == "cars":
            q_z_s_mean, q_z_s_logvar = create_vae_shared_encoder_flat(rectifiedX, rectifiedY, is_training, a)
        else:
            q_z_s_mean, q_z_s_logvar = create_vae_shared_encoder(rectifiedX, rectifiedY, a)

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
    with tf.name_scope("X_decoder_noise"):
        with tf.variable_scope("X_decoder"):
            out_channels = int(inputsX.get_shape()[-1])
            outputsX = create_vae_decoder(z_s, z_x, out_channels, is_training, a)

        # Only for test time
        with tf.variable_scope("X_decoder", reuse=True):
            outputsX_fromX = create_vae_decoder(r_z_x_mean, q_z_x_mean, out_channels, is_training, a)

        with tf.variable_scope("X_decoder", reuse=True):
            outputsX_fromY = create_vae_decoder(r_z_y_mean, q_z_x_mean, out_channels, is_training, a)

        with tf.variable_scope("X_decoder", reuse=True):
            outputsXp = create_vae_decoder(r_z_y_mean, tf.random_normal(q_z_x_logvar.shape), out_channels, is_training, a)

        with tf.variable_scope("X_decoder", reuse=True):
            outputsXpp = create_vae_decoder(r_z_y_mean, tf.random_normal(q_z_x_logvar.shape), out_channels, is_training, a)

        with tf.variable_scope("X_decoder", reuse=True):
            outputsXppp= create_vae_decoder(r_z_y_mean, tf.random_normal(q_z_x_logvar.shape), out_channels, is_training, a)

        ### This tensor is for Ablation study
        with tf.variable_scope("X_decoder", reuse=True):
            outputsXun = create_vae_decoder(eps_s_common, tf.random_normal(q_z_x_logvar.shape), out_channels, is_training, a)


    with tf.name_scope("Y_decoder_noise"):
        with tf.variable_scope("Y_decoder"):
            out_channels = int(inputsY.get_shape()[-1])
            outputsY = create_vae_decoder(z_s, z_y, out_channels, is_training, a)

        # Only for test time
        with tf.variable_scope("Y_decoder", reuse=True):
            outputsY_fromY= create_vae_decoder(r_z_y_mean, q_z_y_mean, out_channels, is_training, a)

        with tf.variable_scope("Y_decoder", reuse=True):
            outputsY_fromX= create_vae_decoder(r_z_x_mean, q_z_y_mean, out_channels, is_training, a)

        with tf.variable_scope("Y_decoder",reuse=True):
            outputsYp = create_vae_decoder(r_z_x_mean, tf.random_normal(q_z_y_logvar.shape), out_channels, is_training, a)

        with tf.variable_scope("Y_decoder",reuse=True):
            outputsYpp = create_vae_decoder(r_z_x_mean, tf.random_normal(q_z_y_logvar.shape), out_channels, is_training, a)

        with tf.variable_scope("Y_decoder", reuse=True):
            outputsYppp= create_vae_decoder(r_z_x_mean, tf.random_normal(q_z_y_logvar.shape), out_channels, is_training, a)

        with tf.variable_scope("Y_decoder", reuse=True):
            outputsYun = create_vae_decoder(eps_s_common, tf.random_normal(q_z_y_logvar.shape), out_channels, is_training, a)

    ######### VISUAL ANALOGIES
    # This is only for visualization (visual analogies), not used in training loss
    with tf.name_scope("image_swapper_X"):
        im_swapped_X,sel_auto_X = create_visual_analogy(r_z_x_mean, q_z_x_mean, outputsX_fromX, inputsX,'X', is_training, a)
    with tf.name_scope("image_swapper_Y"):
        im_swapped_Y,sel_auto_Y = create_visual_analogy(r_z_y_mean, q_z_y_mean, outputsY_fromY, inputsY,'Y', is_training, a)

    ######### LOSSES
    with tf.name_scope("recon_X_loss"):
        recon_X_loss = a.l1_weight*tf.reduce_mean(tf.abs(outputsX-inputsX))

    with tf.name_scope("recon_Y_loss"):
        recon_Y_loss = a.l1_weight*tf.reduce_mean(tf.abs(outputsY-inputsY))

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
                     + a.LAMBDA * kl_InterX_loss + a.LAMBDA * kl_InterY_loss


    ######### OPTIMIZERS
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope("joint_train"):
        with tf.control_dependencies(update_ops):
            joint_tvars = [var for var in tf.trainable_variables()]
            joint_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            joint_grads_and_vars = joint_optim.compute_gradients(joint_loss, var_list=joint_tvars)
            joint_train = joint_optim.apply_gradients(joint_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([recon_X_loss, recon_Y_loss,
                               kl_X_loss, kl_Y_loss, kl_S_loss,
                               kl_InterX_loss, kl_InterY_loss,
                               joint_loss])

    return Model(
        outputsX=outputsX,
        outputsY=outputsY,
        outputsXp=outputsXp,
        outputsYp=outputsYp,
        outputsXpp=outputsXpp,
        outputsYpp=outputsYpp,
        outputsXppp=outputsXppp,
        outputsYppp=outputsYppp,
        outputsXun=outputsXun,
        outputsYun=outputsYun,
        outputsX_fromX=outputsX_fromX,
        outputsX_fromY=outputsX_fromY,
        outputsY_fromY=outputsY_fromY,
        outputsY_fromX=outputsY_fromX,
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
        im_swapped_X=im_swapped_X,
        im_swapped_Y=im_swapped_Y,
        sel_auto_X=sel_auto_X,
        sel_auto_Y=sel_auto_Y,
        recon_X_loss=ema.average(recon_X_loss),
        recon_Y_loss=ema.average(recon_Y_loss),
        kl_X_loss=ema.average(kl_X_loss),
        kl_Y_loss=ema.average(kl_Y_loss),
        kl_S_loss=ema.average(kl_S_loss),
        kl_InterX_loss=ema.average(kl_InterX_loss),
        kl_InterY_loss=ema.average(kl_InterY_loss),
        joint_loss=ema.average(joint_loss),
        train=tf.group(update_losses, incr_global_step, joint_train),
    )


def create_visual_analogy(sR, eR, auto_output, inputs, which_direction, is_training, a):
        swapScoreBKG = 0
        sR_Swap = []
        eR_Swap = []
        sel_auto = []

        for i in range(0,a.batch_size):
            if len(sR.get_shape().as_list()) < 3:
                s_curr = tf.reshape(sR[i, :], [sR.shape[1]])
            else:
                s_curr = tf.reshape(sR[i,:],[sR.shape[1],sR.shape[2],sR.shape[3]])

            # Take a random image from the batch, make sure it is different from current
            bkg_ims_idx = random.randint(0,a.batch_size-1)
            while bkg_ims_idx == i:
                bkg_ims_idx = random.randint(0,a.batch_size-1)

            ex_rnd = tf.reshape(eR[bkg_ims_idx,:],[eR.shape[1]])
            sR_Swap.append(s_curr)
            eR_Swap.append(ex_rnd)

            # Store also selected reference image for visualization
            sel_auto.append(inputs[bkg_ims_idx,:])

        with tf.variable_scope(which_direction + "_decoder", reuse=True):
                    out_channels = int(auto_output.get_shape()[-1])
                    im_swapped = create_vae_decoder(tf.stack(sR_Swap),
                                                    tf.stack(eR_Swap),
                                                    out_channels,
                                                    is_training,
                                                    a)
        return im_swapped, tf.stack(sel_auto)


