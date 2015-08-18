import gzip
import cPickle as pickle

import lasagne
import theano
from theano import tensor as T
from lasagne.objectives import categorical_crossentropy as x_ent
from lasagne.regularization import l2, apply_penalty
import sys
import numpy as np
import imp
import time
import os

import gzip
import cPickle as pickle

from skimage import io
from skimage import img_as_float

class Container(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

symbols = None
args = None
SEED = 0
X_bank = None

def load_image(filename):
    img = io.imread(filename)
    img = img_as_float(img) # don't need to do float32 conversion
    # if it's an rgb image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.asarray( [ img[..., 0], img[..., 1], img[..., 2] ] )
    else: # if it's a bw image
        img = np.asarray( [ img ] )
    return img

def print_network(output_layer):
    layers = lasagne.layers.get_all_layers(output_layer)
    desc = []
    for i in range(0, len(layers)):
        layer = layers[i]
        if isinstance(layer, lasagne.layers.InputLayer):
            desc.append("%s" % "InputLayer")
        elif isinstance(layer, lasagne.layers.Conv2DLayer):
            desc.append("%s, filter_size=%s, num_filters=%s, nonlinearity=%s" % \
                ("Conv2D", layer.filter_size, layer.num_filters, layer.nonlinearity.__name__))
        elif isinstance(layer, lasagne.layers.MaxPool2DLayer):
            desc.append("%s, pool_size=%s" % ("MaxPool2D", layer.pool_size))
        elif isinstance(layer, lasagne.layers.DenseLayer):
            desc.append("%s, num_units=%s" % ("DenseLayer", layer.num_units))
        else:
            desc.append("%s" % "UnknownLayer")
    return "\n".join(desc)
    #sys.stderr.write("Number of parameters: %i\n" % lasagne.layers.count_params(output_layer))

def lenet_skinny():
    # https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    l_in = lasagne.layers.InputLayer( shape=(None, 1, 28, 28) )
    l_conv1 = lasagne.layers.Conv2DLayer(l_in, filter_size=(5,5), num_filters=20/2)
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2))
    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, filter_size=(5,5), num_filters=50/2)
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2))
    l_hidden = lasagne.layers.DenseLayer(
        l_pool2,
        num_units=500/2,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform()
    )
    return l_out


def prepare():

    X = T.tensor4('X')
    y = T.ivector('y')

    output_layer = lenet_skinny()

    all_params = lasagne.layers.get_all_params(output_layer)

    loss_fn = x_ent
    
    prediction = lasagne.layers.get_output(output_layer, X)
    loss = loss_fn(prediction, y).mean() + \
        args["lambda"]*apply_penalty(lasagne.layers.get_all_params(output_layer, regularizable=True), l2 )

    label_vector = lasagne.layers.get_output(output_layer, X)
    pred = T.argmax( label_vector, axis=1 )
    accuracy = T.mean( T.eq(pred,y) )

    return Container(
        { "X": X, "y": y, "output_layer": output_layer, "all_params": all_params,
        "loss": loss, "label_vector": label_vector, "pred": pred, "accuracy": accuracy
        }
    )

def get_batch(filenames, start, end):
    if X_bank == None:
        batch = np.asarray(
            [ load_image(x) for x in filenames[start:end] ],
            dtype="float32"
        )
        return batch
    else:
        return X_bank[start : end]


def train(arg):
    global args, symbols, best_weights, SEED, X_bank
    args = arg

    print args["X_train"]

    filenames = [ (args["dir"] + os.path.sep + args["attr_values"]["filename"][int(x)]) \
        for x in args["X_train"].flatten().tolist() ]

    if "cache" in args:
        print "Loading all images into memory"
        #X_bank = np.asarray( [load_image(x) for x in filenames], dtype="float32" )
        raise NotImplementedError()

    y_train = args["y_train"]
    y_train = np.asarray(y_train.flatten(), dtype="int32")

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
        [symbols.X, symbols.y],
        [symbols.label_vector, symbols.pred, symbols.loss, symbols.accuracy],
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
        np.random.seed(SEED)
        np.random.shuffle(y_train)

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
            y_train_batch = y_train[b*bs : (b+1)*bs]

            #sys.stderr.write("  Batch #%i (%i-%i)\n" % ((b+1), (b*bs), ((b+1)*bs) ))
            v, _, loss, acc = iter_train( X_train_batch, y_train_batch )
            batch_train_losses.append(loss)
            batch_train_accuracies.append(acc)

        sys.stderr.write( "  train_loss, train_accuracy = %f, %f\n" % \
            (np.mean(batch_train_losses), np.mean(batch_train_accuracies)) )

    current_weights = lasagne.layers.get_all_param_values(symbols.output_layer)

    return (print_network(symbols.output_layer), current_weights)

def describe(arg, model):
    return model[0]

def test(arg, model):

    args = arg

    symbols = prepare()
    lasagne.layers.set_all_param_values(symbols.output_layer, model[1])
    iter_test = theano.function(
        [symbols.X],
        symbols.label_vector
    )

    filenames = [ (args["dir"] + os.path.sep + x) for x in args["X_test"].flatten().tolist() ]
    X_test = np.asarray( [load_image(x) for x in filenames], dtype="float32" )

    if "batch_size" in args:
        bs = args["batch_size"]
    else:
        bs = 128

    preds = iter_test(X_test).tolist()

    return preds

if __name__ == '__main__':

    f = gzip.open("other/mnist.meta.pkl.gz")
    args = pickle.load(f)
    args["lambda"] = 0
    args["alpha"] = 0.01
    args["epochs"] = 10
    #args["cache"] = True
    args["dir"] = "data"
    f.close()

    weights = train(args)

    args["X_test"] = args["X_train"]

    preds = test(args, weights)

    for pred in preds:
        print pred


