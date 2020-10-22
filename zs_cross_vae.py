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
from zs_vae_model import create_zs_vae_model
from scipy import io as sio
from sklearn.neighbors import NearestNeighbors,LSHForest
from scipy import stats

import os
import pandas as pd


from shutil import copyfile

parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", required=True, help="path to folder containing images")

########## Sketchy-specific Settings ##########
parser.add_argument("--neef", type=int, default=128, help="number of exclusive representation encoder filters in first conv layer")
parser.add_argument("--exdim", type=int, default=64, help="dimensionality of exclusive representation")

parser.add_argument("--nsef", type=int, default=128, help="number of shared represenation encoder filters in first conv layer")
parser.add_argument("--shdim", type=int, default=64, help="dimensionality of shared representation")

parser.add_argument("--ndf", type=int, default=128, help="number of decoder filters in first conv layer")

parser.add_argument("--min_epochs", default=1, help="Minimum number of training epochs to save model params")
parser.add_argument("--save_freq", type=int, default=0, help="save model every save_freq steps, 0 to disable")

########## General Settings ##########
# Added features mode to extract features for retrieval
parser.add_argument('--gpus', type=str, default='')
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--root_path", required=True, help="path to folder containing train images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=10, help="number of training epochs")


parser.add_argument("--summary_freq", type=int, default=30, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=100, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")


parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=100, help="number of images in batch")
# parser.add_argument("--batch_size", type=int, default=104, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])

parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=10.0, help="weight on L1 term for generator gradient")

# Cross-domain-disen new arugments
parser.add_argument("--LAMBDA", type=float, default=1.0, help="weight on information regularization")
parser.add_argument("--BETA", type=float, default=0.5, help="weight on ELBO")
parser.add_argument("--ar", type=float, default=2000.0, help="Annealing rate on KL for BETA-VAE")
parser.add_argument("--bn_momentum", type=float, default=0.9, help="Annealing rate on KL for BETA-VAE")
parser.add_argument("--separate_enc", default="True", help="use separable convolutions in the encoder")

# By default, we don't use classification loss.
parser.add_argument("--class_weight", type=float, default=0.0, help="weight on class reconstruction loss")
parser.add_argument("--decAct", default="None", help="Activation function applied to output of the decoder")
parser.add_argument("--split", default="1", help="Split 1 (100/25) or Split 2 (104/21)")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

if a.gpus is not '':
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus

K=100
if a.split == "2":
    K=200

CROP_SIZE = 256


def mapChange(inputArr):
    dup = np.copy(inputArr)
    for idx in range(inputArr.shape[1]):
        if (idx != 0):
            dup[:,idx] = dup[:,idx-1] + dup[:,idx]
    return np.multiply(dup, inputArr)

def getImagePath(sketchPath):
    tempArr = sketchPath.replace('sketch', 'photo').split('-')
    imagePath = ''
    for idx in range(len(tempArr) - 1):
        imagePath = imagePath + tempArr[idx] + '-'
    imagePath = imagePath[:-1]
    imagePath = imagePath + '.jpg'
    return imagePath

def getClass(path):
    return path.split('/')[-2]

def random_train_X(trainClasses, train_sketch_idx_per_class, train_image_idx_per_class):
    sketch_idx_list= []
    image_idx_list = []
    for i in range(len(trainClasses)):
        sketchIdx= random.choice(train_sketch_idx_per_class[trainClasses[i]])
        imageIdx = random.choice(train_image_idx_per_class[trainClasses[i]])

        sketch_idx_list.append(sketchIdx)
        image_idx_list.append(imageIdx)
    return np.array(sketch_idx_list), np.array(image_idx_list)

def load_examples(split, root_path):

    if int(split) == 1:
        print('Loading data split', split)
        split_path = 'split1'
    elif int(split) == 2:
        print('Loading data split', split)
        split_path = 'split2'
    else:
        print('There must be something wrong in split argument.')
        return
    # Load sketch VGG features and their_paths
    train_sketch_VGG = np.load(os.path.join(root_path, split_path, 'train_sketch_features.npy'))  # (75479, 4096)
    train_sketch_paths = np.load(os.path.join(root_path, split_path, 'train_sketch_paths.npy'))  # (75479,)
    test_sketch_VGG = np.load(os.path.join(root_path, split_path, 'test_sketch_features.npy'))  # (75479, 4096)
    test_sketch_paths = np.load(os.path.join(root_path, split_path, 'test_sketch_paths.npy'))  # (75479,)

    # Load image VGG features and their_paths
    train_image_VGG = np.load(os.path.join(root_path, split_path, 'train_image_features.npy'))  # (12500, 4096)
    train_image_paths = np.load(os.path.join(root_path, split_path, 'train_image_paths.npy'))  # (12500, )
    test_image_VGG = np.load(os.path.join(root_path, split_path, 'test_image_features.npy'))  # (12500, 4096)
    test_image_paths = np.load(os.path.join(root_path, split_path, 'test_image_paths.npy'))  # (12500, )


    ### Train data preprocessing
    # sketch
    trainClasses=[]
    train_sketch_classes = []
    train_sketch_paths = train_sketch_paths.tolist()
    train_sketch_idx_per_class = {}
    idx=0
    for sketchPath in train_sketch_paths:
        className = sketchPath.split('/')[-2]
        if className not in train_sketch_idx_per_class:
            train_sketch_idx_per_class[className] = []
            trainClasses.append(className)
        train_sketch_idx_per_class[className].append(idx)
        train_sketch_classes.append(className)
        idx += 1

    # Image
    train_image_classes = []
    train_image_paths = train_image_paths.tolist()
    train_image_idx_per_class = {}
    idx=0
    for imagePath in train_image_paths:
        className = imagePath.split('/')[-2]
        if className not in train_image_idx_per_class:
            train_image_idx_per_class[className] = []
        train_image_idx_per_class[className].append(idx)
        train_image_classes.append(className)
        idx += 1

    ### Test data preprocessing
    # sketch
    testClasses = []
    test_sketch_classes = []
    test_sketch_paths = test_sketch_paths.tolist()
    test_sketch_idx_per_class = {}
    idx=0
    for sketchPath in test_sketch_paths:
        className = sketchPath.split('/')[-2]
        if className not in test_sketch_idx_per_class:
            test_sketch_idx_per_class[className] = []
            testClasses.append(className)
        test_sketch_idx_per_class[className].append(idx)
        test_sketch_classes.append(className)
        idx += 1

    # image
    test_image_classes = []
    test_image_paths = test_image_paths.tolist()
    test_image_idx_per_class = {}
    idx=0
    for imagePath in test_image_paths:
        className = imagePath.split('/')[-2]
        if className not in test_image_idx_per_class:
            test_image_idx_per_class[className] = []
        test_image_idx_per_class[className].append(idx)
        test_image_classes.append(className)
        idx += 1

    return trainClasses, testClasses, \
            train_sketch_VGG, train_image_VGG, test_sketch_VGG, test_image_VGG,\
            train_sketch_idx_per_class, train_image_idx_per_class, test_sketch_idx_per_class,\
            test_sketch_classes, test_image_classes


def main():
    root_path = a.root_path
    feature_path = os.path.join(root_path, 'dataset/SketchyVGG')

    # Will be used for test (retrieval) time
    Sketchy_sketch_path = os.path.join(root_path, 'dataset/Sketchy/sketch/tx_000000000000')
    Sketchy_image_path = os.path.join(root_path, 'dataset/Sketchy/extended_photo')
    Retrieval_path = os.path.join(root_path, 'SUBMISSION/Sketchy')

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    print("K = ", K)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "features":
        a.checkpoint = a.output_dir
        # if a.checkpoint is None:
        #     raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))


    #################### Train / Test data Loading ####################
    trainClasses, testClasses, \
    train_sketch_VGG, train_image_VGG, test_sketch_VGG, test_image_VGG, \
    train_sketch_idx_per_class, train_image_idx_per_class,  test_sketch_idx_per_class,\
    test_sketch_classes, test_image_classes = load_examples(a.split, feature_path)

    num_train_sketch = train_sketch_VGG.shape[0]
    num_train_image = train_image_VGG.shape[0]
    num_test_sketch = test_sketch_VGG.shape[0]
    num_test_image = test_image_VGG.shape[0]

    print("Train sketch = %d" % num_train_sketch)
    print("Train image = %d" % num_train_image)
    print("Test sketch = %d" % num_test_sketch)
    print("Test image = %d" % num_test_image)

    train_steps_per_epoch = num_train_sketch // a.batch_size
    test_sketch_steps_per_epoch = num_test_sketch // a.batch_size
    test_img_steps_per_epoch = num_test_image // a.batch_size


    # inputs and targets are [batch_size, 4096]
    inputsX_placeholder = tf.placeholder(tf.float32, shape=[a.batch_size, 512], name='inputsX_placeholder')
    inputsY_placeholder = tf.placeholder(tf.float32, shape=[a.batch_size, 512], name='inputsY_placeholder')
    inputsC_placeholder = tf.placeholder(tf.float32, shape=[a.batch_size, len(trainClasses)], name='inputsC_placeholder')
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training_placeholder')

    model = create_zs_vae_model(inputsX_placeholder, inputsY_placeholder, inputsC_placeholder, is_training_placeholder, a)


    sR_X2Y = model.sR_X2Y
    sR_Y2X = model.sR_Y2X
    eR_X2Y = model.eR_X2Y
    eR_Y2X = model.eR_Y2X

    sR_X2Y_logvar = model.sR_X2Y_logvar
    sR_Y2X_logvar = model.sR_Y2X_logvar
    eR_X2Y_logvar = model.eR_X2Y_logvar
    eR_Y2X_logvar = model.eR_Y2X_logvar

    shared_mean = model.shared_mean
    shared_logvar = model.shared_logvar

    # summaries
    tf.summary.scalar("recon_X_loss", model.recon_X_loss)
    tf.summary.scalar("recon_Y_loss", model.recon_Y_loss)
    tf.summary.scalar("recon_C_loss", model.recon_C_loss)
    tf.summary.scalar("kl_X_loss", model.kl_X_loss)
    tf.summary.scalar("kl_Y_loss", model.kl_Y_loss)
    tf.summary.scalar("kl_S_loss", model.kl_S_loss)
    tf.summary.scalar("kl_InterX_loss", model.kl_InterX_loss)
    tf.summary.scalar("kl_InterY_loss", model.kl_InterY_loss)
    tf.summary.scalar("joint_loss", model.joint_loss)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
        print("parameter_count =", sess.run(parameter_count))

        ### Activate following only when retrain from a checkpoint ###
        # a.checkpoint = a.output_dir

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 500000#20000000
        if a.max_epochs is not None:
            max_steps = train_steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            X_SharedFeat = []
            Y_SharedFeat = []
            ##### Phase 1: Convert all test sketches to shared representation.
            total_num = num_test_sketch
            test_max_steps = test_sketch_steps_per_epoch
            for test_step in range(test_max_steps):
                feed_dict = {
                    inputsX_placeholder: test_sketch_VGG[test_step * a.batch_size: (test_step + 1) * a.batch_size],
                    is_training_placeholder: False}
                results = sess.run({'sR_X2Y': sR_X2Y}, feed_dict=feed_dict)

                X_SharedFeat.append(np.reshape(results['sR_X2Y'], [a.batch_size, -1]))

            last_feed_sketch = np.concatenate([test_sketch_VGG[test_max_steps * a.batch_size:],
                                               np.zeros(((test_max_steps + 1) * a.batch_size - total_num, 512))],
                                              axis=0)

            feed_dict = {inputsX_placeholder: last_feed_sketch,
                         is_training_placeholder: False}

            results = sess.run({'sR_X2Y': sR_X2Y}, feed_dict=feed_dict)

            X_SharedFeat.append(np.reshape(results['sR_X2Y'], [a.batch_size, -1]))
            X_SharedFeat = np.concatenate(X_SharedFeat, axis=0)
            X_SharedFeat = X_SharedFeat[:total_num]

            ##### Phase 2: Convert all test images to shared representation.
            total_num = num_test_image
            test_max_steps = test_img_steps_per_epoch
            for test_step in range(test_max_steps):
                feed_dict = {
                    inputsY_placeholder: test_image_VGG[test_step * a.batch_size: (test_step + 1) * a.batch_size],
                    is_training_placeholder: False}
                results = sess.run({'sR_Y2X': sR_Y2X}, feed_dict=feed_dict)

                Y_SharedFeat.append(np.reshape(results['sR_Y2X'], [a.batch_size, -1]))

            last_feed_image = np.concatenate([test_image_VGG[test_max_steps * a.batch_size:],
                                              np.zeros(((test_max_steps + 1) * a.batch_size - total_num, 512))], axis=0)

            feed_dict = {inputsY_placeholder: last_feed_image,
                         is_training_placeholder: False}

            results = sess.run({'sR_Y2X': sR_Y2X}, feed_dict=feed_dict)

            Y_SharedFeat.append(np.reshape(results['sR_Y2X'], [a.batch_size, -1]))
            Y_SharedFeat = np.concatenate(Y_SharedFeat, axis=0)
            Y_SharedFeat = Y_SharedFeat[:total_num]

            #### Phase 3: Apply K-Nearest Neighbors
            # from CVAE (ECCV 2018)
            nbrs = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute').fit(Y_SharedFeat)

            distances, indices = nbrs.kneighbors(X_SharedFeat)
            retrieved_classes = np.array(test_image_classes)[indices]
            results = np.zeros(retrieved_classes.shape)
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_classes[idx] == np.array(test_sketch_classes)[idx])
            precision_K = np.mean(results, axis=1)
            temp = [np.arange(K) for ii in range(results.shape[0])]

            mAP_K_term = 1.0 / (np.stack(temp, axis=0) + 1)
            mAP_K = np.mean(np.multiply(mapChange(results), mAP_K_term), axis=1)

            print('The mean precision@' + str(K) + 'for test sketches is ' + str(np.mean(precision_K)))
            print('The mAP@' + str(K) + 'for test_sketches is ' + str(np.mean(mAP_K)))

            #### Phase 4: map@all
            nbrs_all = NearestNeighbors(n_neighbors=Y_SharedFeat.shape[0], metric='cosine', algorithm='brute').fit(
                Y_SharedFeat)
            distances, indices = nbrs_all.kneighbors(X_SharedFeat)
            '''
            # from ZSIH (CVPR 2018)
            map_count = 0.
            average_precision = 0.
            for i in range(indices.shape[0]):
                gt_count = 0.
                precision = 0.
                for j in range(indices.shape[1]):
                    this_ind = indices[i][j]
                    if test_sketch_classes[i] == test_image_classes[this_ind]:
                        gt_count += 1.
                        precision += gt_count / (j + 1.)
                if gt_count > 0:
                    average_precision += precision / gt_count
                    map_count += 1.
            mAP_og = average_precision / map_count
            print('The mAP@all(og) for test_sketches is ' + str(mAP_og) )
            '''

            # Faster version with the exactly same (upto 1e-14) logic & result
            retrieved_classes = np.array(test_image_classes)[indices]
            results = np.zeros(retrieved_classes.shape)
            gt_count = []
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_classes[idx] == np.array(test_sketch_classes)[idx])
                gt_count.append(np.sum(results[idx], axis=-1))
                # print(gt_count[idx])

            gt_count = np.array(gt_count)
            temp = [np.arange(results.shape[1]) for ii in range(results.shape[0])]
            mAP_term = 1.0 / (np.stack(temp, axis=0) + 1)
            mAP = np.sum(np.multiply(mapChange(results), mAP_term), axis=1)
            assert gt_count.shape == mAP.shape
            mAP = mAP / gt_count
            print('The mAP@all for test_sketches is ' + str(np.mean(mAP)))

            ##### Phase 6: Retrieval
            test_sketch_paths = np.load(os.path.join(feature_path, 'split1', 'test_sketch_paths.npy'))
            test_image_paths = np.load(os.path.join(feature_path, 'split1', 'test_image_paths.npy'))
            retrieved_paths = np.array(test_image_paths)[indices]

            for label in testClasses:
                sketch_idx = random.sample(test_sketch_idx_per_class[label], 1)
                sketch_idx = sketch_idx[0]
                # sketch_idx = test_sketch_idx_per_class[label][0]

                query_path = os.path.join(Sketchy_sketch_path, test_sketch_paths[sketch_idx])
                if os.path.exists(os.path.join(query_path)):
                    # Create folder
                    os.makedirs(os.path.join(Retrieval_path, label))
                    copyfile(query_path, os.path.join(Retrieval_path, label, 'q.jpg'))

                retrieved_paths_per_class = retrieved_paths[sketch_idx][:10]
                count = 0
                for path in retrieved_paths_per_class:
                    database_path = os.path.join(Sketchy_image_path, path)
                    if os.path.exists(database_path):
                        if label == getClass(path):
                            copyfile(database_path, os.path.join(Retrieval_path, label, str(count)+'.jpg'))
                        else:
                            copyfile(database_path, os.path.join(Retrieval_path, label, str(count) + '_f.jpg'))
                    else:
                        print('Something wrong!')
                        return
                    count += 1

        else:
            start = time.time()
            # The model converges within 9~10 Epochs.
            for epoch in range(a.max_epoch):

                for step in range(train_steps_per_epoch):
                    def should(freq):
                        return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                    options = None
                    run_metadata = None
                    if should(a.trace_freq):
                        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()


                    fetches = {
                        "train": model.train,
                        "global_step": sv.global_step
                    }

                    if should(a.progress_freq):
                        fetches["recon_X_loss"] = model.recon_X_loss
                        fetches["recon_Y_loss"] = model.recon_Y_loss
                        fetches["recon_C_loss"] = model.recon_C_loss
                        fetches["kl_X_loss"] = model.kl_X_loss
                        fetches["kl_Y_loss"] = model.kl_Y_loss
                        fetches["kl_S_loss"] = model.kl_S_loss
                        fetches["kl_InterX_loss"] = model.kl_InterX_loss
                        fetches["kl_InterY_loss"] = model.kl_InterY_loss
                        fetches["joint_loss"] = model.joint_loss

                    if should(a.summary_freq):
                        fetches["summary"] = sv.summary_op

                    train_X_sketch_idx, train_X_image_idx = random_train_X(trainClasses,
                                                                           train_sketch_idx_per_class,
                                                                           train_image_idx_per_class)

                    X_sketch = np.take(train_sketch_VGG, train_X_sketch_idx, axis=0)
                    X_image = np.take(train_image_VGG, train_X_image_idx, axis=0)

                    feed_dict = {inputsX_placeholder: X_sketch,
                                 inputsY_placeholder: X_image,
                                 inputsC_placeholder: np.eye(len(trainClasses)),
                                 is_training_placeholder: True}

                    results = sess.run(fetches, feed_dict, options=options, run_metadata=run_metadata)

                    train_epoch = math.ceil(results["global_step"] / train_steps_per_epoch)
                    train_step = (results["global_step"] - 1) % train_steps_per_epoch + 1
                    # import pdb; pdb.set_trace()

                    if should(a.summary_freq):
                        print("recording summary")
                        sv.summary_writer.add_summary(results["summary"], results["global_step"])


                    if should(a.trace_freq):
                        print("recording trace")
                        sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                    if should(a.progress_freq):
                        rate = (step + 1) * a.batch_size / (time.time() - start)
                        remaining = (max_steps - step) * a.batch_size / rate
                        print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                        print("recon_X_loss", results["recon_X_loss"])
                        print("recon_Y_loss", results["recon_Y_loss"])
                        print("recon_C_loss", results["recon_C_loss"])
                        print("kl_X_loss", results["kl_X_loss"])
                        print("kl_Y_loss", results["kl_Y_loss"])
                        print("kl_S_loss", results["kl_S_loss"])
                        print("kl_InterX_loss", results["kl_InterX_loss"])
                        print("kl_InterY_loss", results["kl_InterY_loss"])

                # if True:# epoch > float(a.min_epochs):

                if sv.should_stop():
                    break

            X_SharedFeat = []
            Y_SharedFeat = []

            print("===================== Evaluation (Accuracy w.r.t. L2) =====================")
            ##### First phase: Convert all test sketches to shared representation.
            total_num = num_test_sketch
            test_max_steps = test_sketch_steps_per_epoch
            for test_step in range(test_max_steps):
                feed_dict = {
                    inputsX_placeholder: test_sketch_VGG[test_step * a.batch_size: (test_step + 1) * a.batch_size],
                    is_training_placeholder: False}
                results = sess.run({'sR_X2Y': sR_X2Y}, feed_dict=feed_dict)

                X_SharedFeat.append(np.reshape(results['sR_X2Y'], [a.batch_size, -1]))

            last_feed_sketch = np.concatenate([test_sketch_VGG[test_max_steps * a.batch_size:],
                                               np.zeros(((test_max_steps + 1) * a.batch_size - total_num, 512))],
                                              axis=0)

            feed_dict = {inputsX_placeholder: last_feed_sketch,
                         is_training_placeholder: False}

            results = sess.run({'sR_X2Y': sR_X2Y}, feed_dict=feed_dict)

            X_SharedFeat.append(np.reshape(results['sR_X2Y'], [a.batch_size, -1]))
            X_SharedFeat = np.concatenate(X_SharedFeat, axis=0)
            X_SharedFeat = X_SharedFeat[:total_num]

            ##### Second phase: Convert all test images to shared representation.
            total_num = num_test_image
            test_max_steps = test_img_steps_per_epoch
            for test_step in range(test_max_steps):
                feed_dict = {
                    inputsY_placeholder: test_image_VGG[test_step * a.batch_size: (test_step + 1) * a.batch_size],
                    is_training_placeholder: False}
                results = sess.run({'sR_Y2X': sR_Y2X}, feed_dict=feed_dict)

                Y_SharedFeat.append(np.reshape(results['sR_Y2X'], [a.batch_size, -1]))

            last_feed_image = np.concatenate([test_image_VGG[test_max_steps * a.batch_size:],
                                              np.zeros(((test_max_steps + 1) * a.batch_size - total_num, 512))], axis=0)

            feed_dict = {inputsY_placeholder: last_feed_image,
                         is_training_placeholder: False}

            results = sess.run({'sR_Y2X': sR_Y2X}, feed_dict=feed_dict)

            Y_SharedFeat.append(np.reshape(results['sR_Y2X'], [a.batch_size, -1]))
            Y_SharedFeat = np.concatenate(Y_SharedFeat, axis=0)
            Y_SharedFeat = Y_SharedFeat[:total_num]

            #### Third phase: Apply K-Nearest Neighbors
            # from CVAE (ECCV 2018)
            nbrs = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute').fit(Y_SharedFeat)

            distances, indices = nbrs.kneighbors(X_SharedFeat)
            retrieved_classes = np.array(test_image_classes)[indices]
            results = np.zeros(retrieved_classes.shape)
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_classes[idx] == np.array(test_sketch_classes)[idx])
            precision_K = np.mean(results, axis=1)
            temp = [np.arange(K) for ii in range(results.shape[0])]

            mAP_K_term = 1.0 / (np.stack(temp, axis=0) + 1)
            mAP_K = np.mean(np.multiply(mapChange(results), mAP_K_term), axis=1)

            print('The mean precision@' + str(K) + 'for test sketches is ' + str(np.mean(precision_K)))
            print('The mAP@' + str(K) + 'for test_sketches is ' + str(np.mean(mAP_K)))

            #### Fourth phase: map@all
            nbrs_all = NearestNeighbors(n_neighbors=Y_SharedFeat.shape[0], metric='cosine', algorithm='brute').fit(
                Y_SharedFeat)
            distances, indices = nbrs_all.kneighbors(X_SharedFeat)
            '''
            # from ZSIH (CVPR 2018)
            map_count = 0.
            average_precision = 0.
            for i in range(indices.shape[0]):
                gt_count = 0.
                precision = 0.
                for j in range(indices.shape[1]):
                    this_ind = indices[i][j]
                    if test_sketch_classes[i] == test_image_classes[this_ind]:
                        gt_count += 1.
                        precision += gt_count / (j + 1.)
                if gt_count > 0:
                    average_precision += precision / gt_count
                    map_count += 1.
            mAP_og = average_precision / map_count
            print('The mAP@all(og) for test_sketches is ' + str(mAP_og) )
            '''

            # Faster version with the exactly same (upto 1e-14) logic & result
            retrieved_classes = np.array(test_image_classes)[indices]
            results = np.zeros(retrieved_classes.shape)
            gt_count = []
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_classes[idx] == np.array(test_sketch_classes)[idx])
                gt_count.append(np.sum(results[idx], axis=-1))
                # print(gt_count[idx])

            gt_count = np.array(gt_count)
            temp = [np.arange(results.shape[1]) for ii in range(results.shape[0])]
            mAP_term = 1.0 / (np.stack(temp, axis=0) + 1)
            mAP = np.sum(np.multiply(mapChange(results), mAP_term), axis=1)
            assert gt_count.shape == mAP.shape
            mAP = mAP / gt_count
            print('The mAP@all for test_sketches is ' + str(np.mean(mAP)))

            print("============================== Saving Model ==============================")
            saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)


main()
