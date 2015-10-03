from __future__ import print_function

import lasagne
import theano
from theano import tensor as T
from lasagne.objectives import squared_error
from lasagne.regularization import l2, apply_penalty
from lasagne.nonlinearities import sigmoid
import sys
import numpy as np
import imp
import time
import os

from skimage import io
from skimage import img_as_float

from pyscript.pyscript import ArffToArgs

import helper

class Container(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

symbols = None
args = None
SEED = 0

def load_image(filename):
    img = io.imread(filename)
    img = img_as_float(img) # don't need to do float32 conversion
    img = np.asarray( [ img ] )
    return img

def print_network(output_layer):
    layers = lasagne.layers.get_all_layers(output_layer)
    desc = []
    for i in range(0, len(layers)):
        layer = layers[i]
        print (layer, layer.output_shape)

def get_net():
    # https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    l_in = lasagne.layers.InputLayer( shape=(None, 1, 28, 28) )
    l_gaussian = lasagne.layers.GaussianNoiseLayer(l_in, sigma=0.3)
    l_conv1 = lasagne.layers.Conv2DLayer(l_gaussian, filter_size=(5,5), num_filters=20/2, nonlinearity=sigmoid)
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2))
    l_deconv1 = lasagne.layers.InverseLayer(l_pool1, l_pool1)
    l_deconv2 = lasagne.layers.InverseLayer(l_deconv1, l_conv1)

    return l_in, l_pool1, l_deconv2


def prepare():

    X = T.tensor4('X')

    input_layer, conv_layer, output_layer = get_net()
    print_network(output_layer)

    all_params = lasagne.layers.get_all_params(output_layer)

    loss_fn = squared_error
    
    prediction = lasagne.layers.get_output(output_layer, X)

    loss_flat = loss_fn(prediction, X).flatten()
    loss = loss_flat.mean()

    return Container(
        { "X": X, "input_layer":input_layer, "output_layer": output_layer, "all_params": all_params, 
        "loss": loss, "loss_flat": loss_flat, "conv_layer": conv_layer }
    )

def get_batch(filenames, start, end):
    batch = np.asarray(
        [ load_image(x) for x in filenames[start:end] ],
        dtype="float32"
    )
    return batch


def train(arg):
    global args, symbols, best_weights, SEED
    args = arg

    filenames = [ (args["dir"] + os.path.sep + args["attr_values"]["filename"][int(x)]) \
        for x in args["X_train"].flatten().tolist() ]

    symbols = prepare()

    alpha = args["alpha"]
    momentum = 0.9

    if alpha != -1:
        if "rmsprop" not in args:
            updates = lasagne.updates.momentum(symbols.loss, symbols.all_params, alpha, momentum)
        else:
            updates = lasagne.updates.rmsprop(symbols.loss, symbols.all_params, alpha)
    else:
        updates = lasagne.updates.adagrad(symbols.loss, symbols.all_params, 1.0)

    iter_train = theano.function(
        [symbols.X],
        [symbols.loss, symbols.loss_flat],
        updates=updates
    )

    if "batch_size" in args:
        bs = args["batch_size"]
    else:
        bs = 128

    best_valid_accuracy = -1
    for e in range(0, args["epochs"]):

        np.random.seed(SEED)
        np.random.shuffle(filenames)

        SEED += 1
        np.random.seed(SEED)

        sys.stderr.write("Epoch #%i:\n" % e)
        batch_train_losses = []
        batch_train_accuracies = []
        batch_train_alt_losses = []
        for b in range(0, len(filenames)):
            if b*bs >= len(filenames):
                break
            X_train_batch = get_batch(filenames, b*bs, (b+1)*bs)
            #print (X_train_batch.shape)

            #sys.stderr.write("  Batch #%i (%i-%i)\n" % ((b+1), (b*bs), ((b+1)*bs) ))
            loss, loss_flat = iter_train( X_train_batch )
            batch_train_losses.append(loss)
            print (loss, loss_flat, sum(loss_flat > 0))
        helper.plot_conv_activity( symbols.conv_layer, X_train_batch[1:2] )

        sys.stderr.write( "  train_loss = %f\n" % \
            (np.mean(batch_train_losses)) )

    current_weights = lasagne.layers.get_all_param_values(symbols.output_layer)

    return (print_network(symbols.output_layer), current_weights)

if __name__ == '__main__':

    f = ArffToArgs()
    f.set_input("../mnist/mnist.meta.arff")
    args = f.get_args()
    f.close()
    args["lambda"] = 0
    args["alpha"] = 0.1
    args["epochs"] = 10
    args["dir"] = "../mnist/data"

    weights = train(args)
