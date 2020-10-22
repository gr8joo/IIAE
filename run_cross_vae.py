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
from vae_model import create_vae_model
from scipy import io as sio

import os


parser = argparse.ArgumentParser()


########## General Settings ##########
# Added features mode to extract features for retrieval
parser.add_argument("--root_path", required=True, help="path to folder containing train images")
parser.add_argument("--dataset_name", required=True, help="path to folder containing train images")

parser.add_argument('--gpus', type=str, default='')
parser.add_argument("--mode", required=True, choices=["train", "test", "features"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")

parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")

# parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--progress_freq", type=int, default=100, help="display progress every progress_freq steps")

parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")


parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"]) # Not being used

parser.add_argument("--l1_weight", type=float, default=1000.0, help="weight on L1 term for generator gradient")
parser.add_argument("--ar", type=float, default=25000.0, help="Annealing rate on KL for BETA-VAE")
parser.add_argument("--min_epochs", default=0, help="Minimum number of training epochs to save model params")
parser.add_argument("--save_freq", type=int, default=500, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--exdim", type=int, default=8, help="dimensionality of exclusive representation")
parser.add_argument("--neef", type=int, default=32, help="number of exclusive representation encoder filters in first conv layer")
parser.add_argument("--nsef", type=int, default=32, help="number of shared represenation encoder filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of decoder filters in first conv layer")

parser.add_argument("--shdim", type=int, default=128, help="dimensionality of shared representation when encoding flat variables")

parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")


# Cross-domain-disen new arugments
parser.add_argument("--noise", type=float, default=0.1, help="Stddev for noise input into representation")
parser.add_argument("--LAMBDA", type=float, default=1.0, help="weight on information regularization")
parser.add_argument("--BETA", type=float, default=0.02, help="weight on ELBO")

parser.add_argument("--bn_momentum", type=float, default=0.9, help="Annealing rate on KL for BETA-VAE")
parser.add_argument("--separate_enc", default="True", help="use separable convolutions in the encoder")
parser.add_argument("--valid", default="True")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

if a.gpus is not '':
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus

CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputsX, inputsY, count, steps_per_epoch")


def load_examples(type, batch_size=a.batch_size):
    if type is 'train':
        if a.train_dir is None or not os.path.exists(a.train_dir):
            raise Exception("train_dir does not exist")

        input_paths = glob.glob(os.path.join(a.train_dir, "*.jpg"))
        decode = tf.image.decode_jpeg
        if len(input_paths) == 0:
            input_paths = glob.glob(os.path.join(a.train_dir, "*.png"))
            decode = tf.image.decode_png
    elif type is 'valid' or type is 'valid2':
        if a.valid_dir is None or not os.path.exists(a.valid_dir):
            raise Exception("valid_dir does not exist")

        input_paths = glob.glob(os.path.join(a.valid_dir, "*.jpg"))
        decode = tf.image.decode_jpeg
        if len(input_paths) == 0:
            input_paths = glob.glob(os.path.join(a.valid_dir, "*.png"))
            decode = tf.image.decode_png
    else:
        if a.test_dir is None or not os.path.exists(a.test_dir):
            raise Exception("train_dir does not exist")

        input_paths = glob.glob(os.path.join(a.test_dir, "*.jpg"))
        decode = tf.image.decode_jpeg
        if len(input_paths) == 0:
            input_paths = glob.glob(os.path.join(a.test_dir, "*.png"))
            decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception(type + "_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_" + type + "_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle = type=="train")
        # path_queue = tf.train.string_input_producer(input_paths, shuffle=True)

        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = preprocess(raw_input[:,:width//2,:])
            b_images = preprocess(raw_input[:,width//2:,:])

    # No longer in terms of input/target, but bidirectionally on domains X and Y
    inputsX, inputsY = [a_images, b_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope(type + "_inputX_images"):
        inputX_images = transform(inputsX)

    with tf.name_scope(type + "_inputY_images"):
        inputY_images = transform(inputsY)

    # paths_batch, inputsX_batch, inputsY_batch = tf.train.batch([paths,inputX_images,inputY_images], batch_size=a.batch_size)
    paths_batch, inputsX_batch, inputsY_batch = tf.train.batch([paths, inputX_images, inputY_images], batch_size=batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputsX=inputsX_batch,
        inputsY=inputsY_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}

        # all_kinds = ["inputsX", "outputsX", "outputsXp", "outputsXpp", "outputsX_fromX", "outputsX_fromY", "im_swapped_X", "sel_auto_X",
        #              "inputsY", "outputsY", "outputsYp", "outputsYpp", "outputsY_fromY", "outputsY_fromX", "im_swapped_Y", "sel_auto_Y"]
        all_kinds = ["inputsX", "outputsX", "outputsXp", "outputsXpp", "outputsXppp", "outputsX_fromY",
                     "im_swapped_X", "sel_auto_X",
                     "inputsY", "outputsY", "outputsYp", "outputsYpp", "outputsYppp", "outputsY_fromX",
                     "im_swapped_Y", "sel_auto_Y"]

        for kind in all_kinds:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def save_features(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "features")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        filename = name + ".mat"
        out_path = os.path.join(image_dir, filename)
        sio.savemat(out_path,{'inX':fetches["inputsX"][i],
                             'inY':fetches["inputsY"][i],
                             'shared_mean':fetches["shared_mean"][i],
                             'sR_X2Y':fetches["sR_X2Y"][i],
                             'sR_Y2X':fetches["sR_Y2X"][i],
                             'eR_X2Y':fetches["eR_X2Y"][i],
                             'eR_Y2X':fetches["eR_Y2X"][i],
                             'shared_logvar':fetches["shared_logvar"][i],
                             'sR_X2Y_logvar':fetches["sR_X2Y_logvar"][i],
                             'sR_Y2X_logvar':fetches["sR_Y2X_logvar"][i],
                             'eR_X2Y_logvar':fetches["eR_X2Y_logvar"][i],
                             'eR_Y2X_logvar':fetches["eR_Y2X_logvar"][i]})
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>inX</th><th>out(1)</th><th>out(2)</th><th>auto</th><th>swap</th><th>randomimage</th><th>inY</th><th>out(1)</th><th>out(2)</th><th>auto</th><th>swap</th><th>rnd</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        all_kinds = ["inputsX", "outputsX", "outputsXp", "outputsXpp", "outputsXppp", "outputsX_fromY",
                     "im_swapped_X", "sel_auto_X",
                     "inputsY", "outputsY", "outputsYp", "outputsYpp", "outputsYppp", "outputsY_fromX",
                     "im_swapped_Y", "sel_auto_Y"]

        for kind in all_kinds:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path



def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if a.dataset_name.lower() == "mnist" or a.dataset_name.lower() == "mnist-cdcb":
        a.dataset_name = "MNIST-CDCB"
        a.train_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'train')
        a.valid_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'valid')
        a.test_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'test')
        a.min_epochs = 6
        a.save_freq = 5000
        a.BETA = 0.2
    elif a.dataset_name.lower() == "facades" or a.dataset_name.lower() == "facade":
        a.dataset_name = "facades"
        a.train_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'train')
        a.valid_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'val')
        a.test_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'test')
        a.min_epochs = 299
        a.save_freq = 100
        a.progress_freq = 100
        a.BETA = 0.001
    elif a.dataset_name.lower() == "maps" or a.dataset_name.lower() == "map":
        a.dataset_name = "maps"
        a.train_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'train')
        a.valid_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'valid')
        a.test_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'test')
        a.min_epochs = 399
        a.save_freq = 137
        a.progress_freq = 137
        a.l1_weight = 20000.0
    elif a.dataset_name.lower() == "cars" or a.dataset_name.lower() == "car":
        a.dataset_name = "cars"
        a.train_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'train')
        a.valid_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'train') # Not going to be used!
        a.test_dir = os.path.join(a.root_path, 'dataset', a.dataset_name, 'test')
        a.save_freq = 101200 # Save every 50 epochs without validation
        a.ar = 100.0


    print('dataset_name:', a.dataset_name)
    print('BETA:', a.BETA)
    print('l1_weight:', a.l1_weight)
    print('ar:', a.ar)

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


    train_examples = load_examples(type='train')
    valid_examples = load_examples(type='valid')
    test_examples = load_examples(type='test')
    print("Train examples count = %d,   train steps per epoch %d" % (train_examples.count, train_examples.steps_per_epoch))
    print("Valid examples count = %d" % valid_examples.count)
    print("Test examples count = %d" % test_examples.count)


    inputsX_placeholder = tf.placeholder(tf.float32, shape=[a.batch_size, 256, 256, 3], name='inputsX_placeholder')
    inputsY_placeholder = tf.placeholder(tf.float32, shape=[a.batch_size, 256, 256, 3], name='inputsY_placeholder')
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training_placeholder')
    model = create_vae_model(inputsX_placeholder, inputsY_placeholder, is_training_placeholder, a)

    # undo colorization splitting on images that we use for display/output
    inputsX = deprocess(inputsX_placeholder)
    inputsY = deprocess(inputsY_placeholder)
    outputsX = deprocess(model.outputsX)
    outputsY = deprocess(model.outputsY)
    outputsXp = deprocess(model.outputsXp)
    outputsYp = deprocess(model.outputsYp)
    outputsXpp = deprocess(model.outputsXpp)
    outputsYpp = deprocess(model.outputsYpp)
    outputsXppp = deprocess(model.outputsXppp)
    outputsYppp = deprocess(model.outputsYppp)
    outputsX_fromX = deprocess(model.outputsX_fromX)
    outputsX_fromY = deprocess(model.outputsX_fromY)
    outputsY_fromY = deprocess(model.outputsY_fromY)
    outputsY_fromX = deprocess(model.outputsY_fromX)
    im_swapped_X = deprocess(model.im_swapped_X)
    im_swapped_Y = deprocess(model.im_swapped_Y)
    sel_auto_X = deprocess(model.sel_auto_X)
    sel_auto_Y = deprocess(model.sel_auto_Y)

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


    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputsX"):
        converted_inputsX = convert(inputsX)

    with tf.name_scope("convert_inputsY"):
        converted_inputsY = convert(inputsY)

    with tf.name_scope("convert_outputsX"):
        converted_outputsX = convert(outputsX)

    with tf.name_scope("convert_outputsY"):
        converted_outputsY = convert(outputsY)

    with tf.name_scope("convert_outputsXp"):
        converted_outputsXp = convert(outputsXp)

    with tf.name_scope("convert_outputsYp"):
        converted_outputsYp = convert(outputsYp)

    with tf.name_scope("convert_outputsXpp"):
        converted_outputsXpp= convert(outputsXpp)

    with tf.name_scope("convert_outputsYpp"):
        converted_outputsYpp= convert(outputsYpp)

    with tf.name_scope("convert_outputsXppp"):
        converted_outputsXppp= convert(outputsXppp)

    with tf.name_scope("convert_outputsYppp"):
        converted_outputsYppp= convert(outputsYppp)

    with tf.name_scope("convert_outputsX_fromX"):
        converted_outputsX_fromX = convert(outputsX_fromX)

    with tf.name_scope("convert_outputsX_fromY"):
        converted_outputsX_fromY = convert(outputsX_fromY)

    with tf.name_scope("convert_outputsX_fromX"):
        converted_outputsY_fromY = convert(outputsY_fromY)

    with tf.name_scope("convert_outputsX_fromY"):
        converted_outputsY_fromX = convert(outputsY_fromX)

    with tf.name_scope("convert_im_swapped_X"):
        converted_im_swapped_X = convert(im_swapped_X)

    with tf.name_scope("convert_im_swapped_Y"):
        converted_im_swapped_Y = convert(im_swapped_Y)

    with tf.name_scope("convert_sel_auto_X"):
        converted_sel_auto_X= convert(sel_auto_X)

    with tf.name_scope("convert_sel_auto_Y"):
        converted_sel_auto_Y= convert(sel_auto_Y)

    with tf.name_scope("encode_images"):
        display_fetches = {
            # "paths": examples.paths,
            "inputsX": tf.map_fn(tf.image.encode_png, converted_inputsX, dtype=tf.string, name="inputX_pngs"),
            "inputsY": tf.map_fn(tf.image.encode_png, converted_inputsY, dtype=tf.string, name="inputY_pngs"),
            "outputsX": tf.map_fn(tf.image.encode_png, converted_outputsX, dtype=tf.string, name="outputX_pngs"),
            "outputsY": tf.map_fn(tf.image.encode_png, converted_outputsY, dtype=tf.string, name="outputY_pngs"),
            "outputsXp": tf.map_fn(tf.image.encode_png, converted_outputsXp, dtype=tf.string, name="outputXp_pngs"),
            "outputsYp": tf.map_fn(tf.image.encode_png, converted_outputsYp, dtype=tf.string, name="outputYp_pngs"),
            "outputsXpp": tf.map_fn(tf.image.encode_png, converted_outputsXpp, dtype=tf.string, name="outputXpp_pngs"),
            "outputsYpp": tf.map_fn(tf.image.encode_png, converted_outputsYpp, dtype=tf.string, name="outputYpp_pngs"),
            "outputsXppp": tf.map_fn(tf.image.encode_png, converted_outputsXppp, dtype=tf.string, name="outputXpp_pngs"),
            "outputsYppp": tf.map_fn(tf.image.encode_png, converted_outputsYppp, dtype=tf.string, name="outputYpp_pngs"),
            # "outputsXun": tf.map_fn(tf.image.encode_png, converted_outputsXun, dtype=tf.string, name="outputXpp_pngs"),
            # "outputsYun": tf.map_fn(tf.image.encode_png, converted_outputsYun, dtype=tf.string, name="outputYpp_pngs"),
            # "outputsX_fromX": tf.map_fn(tf.image.encode_png, converted_outputsX_fromX, dtype=tf.string, name="outputX_fromX_pngs"),
            "outputsX_fromY": tf.map_fn(tf.image.encode_png, converted_outputsX_fromY, dtype=tf.string, name="outputX_fromY_pngs"),
            # "outputsY_fromY": tf.map_fn(tf.image.encode_png, converted_outputsY_fromY, dtype=tf.string, name="outputY_fromY_pngs"),
            "outputsY_fromX": tf.map_fn(tf.image.encode_png, converted_outputsY_fromX, dtype=tf.string, name="outputY_fromX_pngs"),
            "im_swapped_X": tf.map_fn(tf.image.encode_png, converted_im_swapped_X, dtype=tf.string, name="im_swapped_X_pngs"),
            "im_swapped_Y": tf.map_fn(tf.image.encode_png, converted_im_swapped_Y, dtype=tf.string, name="im_swapped_Y_pngs"),
            "sel_auto_X": tf.map_fn(tf.image.encode_png, converted_sel_auto_X, dtype=tf.string, name="sel_auto_X_pngs"),
            "sel_auto_Y": tf.map_fn(tf.image.encode_png, converted_sel_auto_Y, dtype=tf.string, name="sel_auto_Y_pngs"),
        }
    with tf.name_scope("extract_features"):
        features_fetches = {
            # "paths": test_examples.paths,
            "inputsX": converted_inputsX,
            "sR_X2Y": sR_X2Y,
            "eR_X2Y": eR_X2Y,
            "sR_X2Y_logvar": sR_X2Y_logvar,
            "eR_X2Y_logvar": eR_X2Y_logvar,
            "inputsY": converted_inputsY,
            "sR_Y2X": sR_Y2X,
            "eR_Y2X": eR_Y2X,
            "sR_Y2X_logvar": sR_Y2X_logvar,
            "eR_Y2X_logvar": eR_Y2X_logvar,
            "shared_mean": shared_mean,
            "shared_logvar": shared_logvar,
        }

    # summaries
    with tf.name_scope("X1_input_summary"):
        tf.summary.image("inputsX", converted_inputsX, max_outputs=1)

    with tf.name_scope("Y1_input_summary"):
        tf.summary.image("inputsY", converted_inputsY, max_outputs=1)

    with tf.name_scope("X_output_summary"):
        tf.summary.image("outputsX", converted_outputsX, max_outputs=1)

    with tf.name_scope("Y_output_summary"):
        tf.summary.image("outputsY", converted_outputsY, max_outputs=1)

    with tf.name_scope("X_cond_sample_summary"):
        tf.summary.image("outputsXp", converted_outputsXp, max_outputs=1)

    with tf.name_scope("Y_cond_sample_summary"):
        tf.summary.image("outputsYp", converted_outputsYp, max_outputs=1)

    with tf.name_scope("X_uncond_sample_summary"):
        tf.summary.image("outputsXpp", converted_outputsXpp, max_outputs=1)

    with tf.name_scope("Y_uncond_sample_summary"):
        tf.summary.image("outputsYpp", converted_outputsYpp, max_outputs=1)

    with tf.name_scope("X_fromX_summary"):
        tf.summary.image("outputsX_fromX", converted_outputsX_fromX, max_outputs=1)

    with tf.name_scope("X_fromY_summary"):
        tf.summary.image("outputsX_fromY", converted_outputsX_fromY, max_outputs=1)

    with tf.name_scope("Y_fromY_summary"):
        tf.summary.image("outputsY_fromY", converted_outputsY_fromY, max_outputs=1)

    with tf.name_scope("Y_fromX_summary"):
            tf.summary.image("outputsY_fromX", converted_outputsY_fromX, max_outputs=1)

    with tf.name_scope("swapped_1Y_summary"):
        tf.summary.image("im_swapped_Y", converted_im_swapped_Y,max_outputs=1)
        tf.summary.image("sel_auto_Y", converted_sel_auto_Y,max_outputs=1)

    with tf.name_scope("swapped_2X_summary"):
        tf.summary.image("im_swapped_X", converted_im_swapped_X,max_outputs=1)
        tf.summary.image("sel_auto_X", converted_sel_auto_X,max_outputs=1)

    tf.summary.scalar("recon_X_loss", model.recon_X_loss)
    tf.summary.scalar("recon_Y_loss", model.recon_Y_loss)
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

        max_steps = 1012000#20000000
        if a.max_epochs is not None:
            max_steps = train_examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(test_examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                inputs = sess.run([test_examples.paths, test_examples.inputsX, test_examples.inputsY])
                feed_dict = {inputsX_placeholder: inputs[1],
                             inputsY_placeholder: inputs[2],
                             is_training_placeholder: False}
                results = sess.run(display_fetches, feed_dict)
                results["paths"] = inputs[0]
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)

        elif a.mode == "features":
            max_steps = min(test_examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                inputs = sess.run([test_examples.paths, test_examples.inputsX, test_examples.inputsY])
                feed_dict = {inputsX_placeholder: inputs[1],
                             inputsY_placeholder: inputs[2],
                             is_training_placeholder: False}
                results = sess.run(features_fetches, feed_dict)
                results["paths"] = inputs[0]
                print(inputs[0])
                save_features(results)

        else:
            # training
            start = time.time()
            old_sum = 0

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)
                def should_save(freq, train_step):
                    return freq > 0 and ((train_step+1) % freq == 0 or step == max_steps - 1)

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
                    fetches["kl_X_loss"] = model.kl_X_loss
                    fetches["kl_Y_loss"] = model.kl_Y_loss
                    fetches["kl_S_loss"] = model.kl_S_loss
                    fetches["kl_InterX_loss"] = model.kl_InterX_loss
                    fetches["kl_InterY_loss"] = model.kl_InterY_loss
                    fetches["joint_loss"] = model.joint_loss

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                inputs = sess.run([train_examples.paths, train_examples.inputsX, train_examples.inputsY])
                feed_dict = {inputsX_placeholder: inputs[1],
                             inputsY_placeholder: inputs[2],
                             is_training_placeholder: True}

                results = sess.run(fetches, feed_dict, options=options, run_metadata=run_metadata)
                results["global_step"] = sess.run(sv.global_step) - 1
                train_epoch = math.ceil(results["global_step"] / train_examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % train_examples.steps_per_epoch + 1

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    # train_epoch = math.ceil(results["global_step"] / train_examples.steps_per_epoch)
                    # train_step = (results["global_step"] - 1) % train_examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("recon_X_loss", results["recon_X_loss"])
                    print("recon_Y_loss", results["recon_Y_loss"])
                    print("kl_X_loss", results["kl_X_loss"])
                    print("kl_Y_loss", results["kl_Y_loss"])
                    print("kl_S_loss", results["kl_S_loss"])
                    print("kl_InterX_loss", results["kl_InterX_loss"])
                    print("kl_InterY_loss", results["kl_InterY_loss"])

                if  train_epoch >= float(a.min_epochs) and should_save(a.save_freq, results["global_step"]):
                    if a.dataset_name == 'cars':
                        print("============================== Saving Model ==============================")
                        saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)
                        continue

                    X_SharedFeat = []
                    Y_SharedFeat = []
                    X_ExFeat = []
                    Y_ExFeat = []

                    print("===================== Evaluation (Accuracy w.r.t. L2) =====================")
                    eval_max_steps = min(valid_examples.steps_per_epoch, max_steps)
                    for eval_step in range(eval_max_steps):
                        inputs = sess.run([valid_examples.paths, valid_examples.inputsX, valid_examples.inputsY])
                        feed_dict = {inputsX_placeholder: inputs[1],
                                     inputsY_placeholder: inputs[2],
                                     is_training_placeholder: False}
                        results = sess.run(features_fetches, feed_dict)

                        X_SharedFeat.append(np.reshape(results['sR_X2Y'], [a.batch_size, -1]))
                        Y_SharedFeat.append(np.reshape(results['sR_Y2X'], [a.batch_size, -1]))

                        X_ExFeat.append(np.reshape(results['eR_X2Y'], [a.batch_size, -1]))
                        Y_ExFeat.append(np.reshape(results['eR_Y2X'], [a.batch_size, -1]))

                    total_num = valid_examples.count

                    ##### Shared Representation Preprocess #####
                    X_SharedFeat = np.concatenate(X_SharedFeat, axis=0)
                    X_SharedFeat = X_SharedFeat[:total_num]
                    Y_SharedFeat = np.concatenate(Y_SharedFeat, axis=0)
                    Y_SharedFeat = Y_SharedFeat[:total_num]

                    # X to Y
                    X_SharedFeatRepeat = np.repeat(X_SharedFeat, total_num, axis=0)
                    X_SharedFeatRepeat = np.reshape(X_SharedFeatRepeat, [total_num, total_num, -1])
                    Y_SharedFeatTile = np.tile(Y_SharedFeat, [total_num, 1])
                    Y_SharedFeatTile = np.reshape(Y_SharedFeatTile, [total_num, total_num, -1])

                    X_SharedDiff = (X_SharedFeatRepeat - Y_SharedFeatTile) ** 2
                    X_SharedDist = np.sum(X_SharedDiff, axis=-1)
                    X_SharedDistMinIdx = np.argmin(X_SharedDist, axis=-1)

                    del X_SharedFeatRepeat
                    del Y_SharedFeatTile
                    del X_SharedDiff
                    del X_SharedDist

                    # Y to X
                    X_SharedFeatTile = np.tile(X_SharedFeat, [total_num, 1])
                    X_SharedFeatTile = np.reshape(X_SharedFeatTile, [total_num, total_num, -1])
                    Y_SharedFeatRepeat = np.repeat(Y_SharedFeat, total_num, axis=0)
                    Y_SharedFeatRepeat = np.reshape(Y_SharedFeatRepeat, [total_num, total_num, -1])

                    Y_SharedDiff = (Y_SharedFeatRepeat - X_SharedFeatTile) ** 2
                    Y_SharedDist = np.sum(Y_SharedDiff, axis=-1)
                    Y_SharedDistMinIdx = np.argmin(Y_SharedDist, axis=-1)

                    del X_SharedFeatTile
                    del Y_SharedFeatRepeat
                    del Y_SharedDiff
                    del Y_SharedDist

                    del X_SharedFeat
                    del Y_SharedFeat

                    ##### Exclusive Representation Preprocess #####
                    X_ExFeat = np.concatenate(X_ExFeat, axis=0)
                    X_ExFeat = X_ExFeat[:total_num]
                    Y_ExFeat = np.concatenate(Y_ExFeat, axis=0)
                    Y_ExFeat = Y_ExFeat[:total_num]

                    # X to Y
                    X_ExFeatRepeat = np.repeat(X_ExFeat, total_num, axis=0)
                    X_ExFeatRepeat = np.reshape(X_ExFeatRepeat, [total_num, total_num, -1])
                    Y_ExFeatTile = np.tile(Y_ExFeat, [total_num, 1])
                    Y_ExFeatTile = np.reshape(Y_ExFeatTile, [total_num, total_num, -1])

                    X_ExDiff = (X_ExFeatRepeat - Y_ExFeatTile) ** 2
                    X_ExDist = np.sum(X_ExDiff, axis=-1)
                    X_ExDistMinIdx = np.argmin(X_ExDist, axis=-1)

                    del X_ExFeatRepeat
                    del Y_ExFeatTile
                    del X_ExDiff
                    del X_ExDist

                    # Y to X
                    X_ExFeatTile = np.tile(X_ExFeat, [total_num, 1])
                    X_ExFeatTile = np.reshape(X_ExFeatTile, [total_num, total_num, -1])
                    Y_ExFeatRepeat = np.repeat(Y_ExFeat, total_num, axis=0)
                    Y_ExFeatRepeat = np.reshape(Y_ExFeatRepeat, [total_num, total_num, -1])

                    Y_ExDiff = (Y_ExFeatRepeat - X_ExFeatTile) ** 2
                    Y_ExDist = np.sum(Y_ExDiff, axis=-1)
                    Y_ExDistMinIdx = np.argmin(Y_ExDist, axis=-1)

                    del X_ExFeatTile
                    del Y_ExFeatRepeat
                    del Y_ExDiff
                    del Y_ExDist

                    del X_ExFeat
                    del Y_ExFeat


                    # Compute num of correct retrieval
                    X_SharedPositive = 0
                    Y_SharedPositive = 0
                    X_ExPositive = 0
                    Y_ExPositive = 0
                    for i in range(total_num):
                        if X_SharedDistMinIdx[i] == i:
                            X_SharedPositive += 1
                        if Y_SharedDistMinIdx[i] == i:
                            Y_SharedPositive += 1
                        if X_ExDistMinIdx[i] == i:
                            X_ExPositive += 1
                        if Y_ExDistMinIdx[i] == i:
                            Y_ExPositive += 1

                    print("X_Shared_Accuracy:", float(X_SharedPositive) / float(total_num), '          Y_Shared_Accuracy:', float(Y_SharedPositive) / float(total_num))
                    print("X_Ex_Accuracy:", float(X_ExPositive) / float(total_num), '          Y_Ex_Accuracy:', float(Y_ExPositive) / float(total_num))

                    if X_SharedPositive + Y_SharedPositive > old_sum and X_ExPositive <= 1 and Y_ExPositive <= 1:
                        old_sum = X_SharedPositive + Y_SharedPositive
                        print("============================== Saving Model ==============================")
                        saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    print(inputs[0])
                    break

main()
