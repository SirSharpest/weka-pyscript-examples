import lasagne
import theano
from theano import tensor as T
from lasagne.objectives import squared_error
from lasagne.regularization import l2
from lasagne.regularization import regularize_network_params as reg
import sys
import numpy as np
import imp
import time
import os

from pyscript.pyscript import ArffToArgs, uses

"""
args: train, test, valid, num_instances, num_classes, num_attributes
"""

class Container(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

symbols = None
args = None
best_weights = None
iter_test = None
SEED = 0

def compare_to_logistic():
    num_attributes = len( args["attributes"] ) - 1
    l_in = lasagne.layers.InputLayer( shape=(None, num_attributes) )
    l_out = lasagne.layers.DenseLayer(
        l_in,
        num_units=args["num_classes"],
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    return "Number of params for logistic(): %i" % lasagne.layers.count_params(l_out)

def squared_error_net():
    num_attributes = len( args["attributes"] ) - 1
    l_in = lasagne.layers.InputLayer( shape=(None, num_attributes) )
    l_hidden = lasagne.layers.DenseLayer(
        l_in,
        num_units=args["num_classes"]-1,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    l_flatten = lasagne.layers.FlattenLayer(l_out, outdim=1)
    print "Number of params for squared_error_net(): %i" % lasagne.layers.count_params(l_flatten)
    print compare_to_logistic()
    return l_flatten


def get_num_hidden_units(p, k):
    x = float(p*k + k - 1) / (p + 2)
    #return (p*x + x + x + 1)
    return int(round(x))

def squared_error_net_adaptive():
    num_attributes = len( args["attributes"] ) - 1
    l_in = lasagne.layers.InputLayer( shape=(None, num_attributes) )
    l_hidden = lasagne.layers.DenseLayer(
        l_in,
        num_units=get_num_hidden_units(num_attributes, args["num_classes"]),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    l_flatten = lasagne.layers.FlattenLayer(l_out, outdim=1)
    print "Number of params for squared_error_net_adaptive(): %i" % lasagne.layers.count_params(l_flatten)
    print compare_to_logistic()
    return l_flatten

def prepare():

    X = T.fmatrix('X')
    y = T.ivector('y')

    if "adaptive" not in args:
        output_layer = squared_error_net()
    else:
        output_layer = squared_error_net_adaptive()

    all_params = lasagne.layers.get_all_params(output_layer)

    loss_fn = squared_error
    label_vector = lasagne.layers.get_output(output_layer, X)
    loss = loss_fn(label_vector, y).mean()

    pred = T.maximum(0, T.minimum( T.round(label_vector), args["num_classes"]-1 ) )
    accuracy = T.mean( T.eq( pred, y ) )

    return Container(
        { "X": X, "y": y, "output_layer": output_layer, "all_params": all_params,
        "loss": loss, "label_vector": label_vector, "pred": pred, "accuracy": accuracy
        }
    )

def train(arg):

    global args, symbols, best_weights, SEED
    args = arg
    symbols = prepare()

    debug = False
    if "debug" in args and args["debug"]:
        debug = True

    if not os.path.exists("/tmp/exp/"):
        os.makedirs("/tmp/exp")
    if "kappa" in args or "hybrid" in args:
        g = open( "/tmp/exp/kappa.txt", "wb" )
    else:
        g = open( "/tmp/exp/normal.txt", "wb" )
    g.write("loss,alt_loss\n")

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
    iter_valid = theano.function(
        [symbols.X, symbols.y],
        [symbols.label_vector, symbols.pred, symbols.loss, symbols.accuracy]
    )

    args["X_train"] = np.asarray(args["X_train"], dtype="float32")
    args["y_train"] = np.asarray(args["y_train"].flatten(), dtype="int32")

    idxs = [x for x in range(0, args["X_train"].shape[0])]
    X_train = args["X_train"][idxs]
    y_train = args["y_train"][idxs]

    sys.stderr.write(str(X_train.shape)+"\n")
    sys.stderr.write(str(y_train.shape)+"\n")

    if "batch_size" in args:
        bs = args["batch_size"]
    else:
        bs = 128

    best_valid_accuracy = -1
    for e in range(0, args["epochs"]):

        np.random.seed(SEED)
        np.random.shuffle(X_train)
        np.random.seed(SEED)
        np.random.shuffle(y_train)

        SEED += 1
        np.random.seed(SEED)

        sys.stderr.write("Epoch #%i:\n" % e)
        batch_train_losses = []
        batch_train_accuracies = []
        for b in range(0, X_train.shape[0]):
            if b*bs >= X_train.shape[0]:
                break

            #sys.stderr.write("  Batch #%i (%i-%i)\n" % ((b+1), (b*bs), ((b+1)*bs) ))
            vector, pred, loss, acc = iter_train( X_train[b*bs : (b+1)*bs], y_train[b*bs : (b+1)*bs] )
            if debug:
                print vector[0:10].tolist(), pred[0:10].tolist()

            batch_train_losses.append(loss)
            batch_train_accuracies.append(acc)

        batch_valid_losses = []
        batch_valid_accuracies = []

        best_weights = lasagne.layers.get_all_param_values(symbols.output_layer)

        sys.stderr.write( "  train_loss, train_accuracy = %f, %f\n" % \
            (np.mean(batch_train_losses), np.mean(batch_train_accuracies)) )
        g.write( str(np.mean(batch_train_losses)) + "\n" )

    g.close()

    return best_weights

def describe(arg, weights):
    return "blank"

def test(arg, weights):

    global args, symbols, iter_test
    args = arg

    symbols = prepare()
    lasagne.layers.set_all_param_values(symbols.output_layer, weights)
    iter_test = theano.function(
        [symbols.X],
        symbols.pred
    )

    args["X_test"] = np.asarray(args["X_test"], dtype="float32")

    if "batch_size" in args:
        bs = args["batch_size"]
    else:
        bs = 128

    X_test = args["X_test"]

    preds = iter_test(X_test).tolist()

    new = []
    for pred in preds:
        new.append( np.eye(args["num_classes"])[pred].tolist() )
    return new

if __name__ == '__main__':
    x = ArffToArgs()
    x.set_input("data/cpu_act.arff")
    x.set_class_index("last")
    x.set_impute(True)
    x.set_binarize(True)
    x.set_standardize(True)
    x.set_arguments("adaptive=True;alpha=0.01;lambda=0;epochs=500;rmsprop=True")
    args = x.get_args()
    #args["debug"] = True

    args["X_test"] = np.asarray(args["X_train"], dtype="float32")

    model = train(args)

    test(args, model)