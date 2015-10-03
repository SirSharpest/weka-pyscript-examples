import lasagne
import theano
from theano import tensor as T
from lasagne.objectives import categorical_crossentropy as x_ent
from lasagne.objectives import squared_error
from lasagne.regularization import l2
from lasagne.regularization import regularize_network_params as reg
import sys
import numpy as np
import imp
import time
import os
import gzip
import cPickle as pickle

from pyscript.pyscript import ArffToArgs

"""
args: train, test, valid, num_instances, num_classes, num_attributes
"""

class Container(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

symbols = None
args = None
best_weights = None
first_time_test = True
iter_test = None
SEED = 0

def soft_kappa_loss(pred_vector, actual_vector):
    return (T.dot(pred_vector, T.arange(0,args["num_classes"])) - actual_vector)**2

def get_hybrid_loss(a,b):
    def this_hybrid_loss(pred_vector, actual_vector):
        if a == 1 and b == 0:
            return x_ent(pred_vector, actual_vector)
        elif a == 0 and b == 1:
            return soft_kappa_loss(pred_vector, actual_vector)
        else:
            return a*x_ent(pred_vector, actual_vector) + b*soft_kappa_loss(pred_vector, actual_vector)
    return this_hybrid_loss

def expectation(q):
    res = 0
    for i in range(0, len(q)):
        res += (i*q[i])
    return res

def expectations_rounded(qs):
    pred_labels = []
    for j in range(0, len(qs)):
        pred_label = int(round(expectation(qs[j])))
        pred_labels.append(pred_label)
    return pred_labels

def logistic():
    l_in = lasagne.layers.InputLayer( shape=(None, len(args["attributes"])-1 ) )
    l_out = lasagne.layers.DenseLayer(
        l_in,
        num_units=args["num_classes"],
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform()
    )
    return l_out

def network():
    l_in = lasagne.layers.InputLayer( shape=(None, len(args["attributes"])-1 ) )
    l_hidden = lasagne.layers.DenseLayer(
        l_in,
        num_units=args["num_hidden_units"],
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform()
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=args["num_classes"],
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform()
    )
    return l_out

"""
Functions for doing the regression (squared error loss) experiments.
"""

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
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform()
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=1,
        nonlinearity=lasagne.nonlinearities.linear,
        W=lasagne.init.GlorotUniform()
    )
    l_flatten = lasagne.layers.FlattenLayer(l_out, outdim=1)
    print "Number of params for squared_error_net_adaptive(): %i" % lasagne.layers.count_params(l_flatten)
    print compare_to_logistic()
    return l_flatten

def prepare():

    X = T.fmatrix('X')
    y = T.ivector('y')

    assert not ("regression" in args and "logistic" in args)

    if "regression" in args:
        output_layer = squared_error_net_adaptive()
    else:
        output_layer = logistic()

    all_params = lasagne.layers.get_all_params(output_layer)

    if "regression" in args:
        prob_vector = lasagne.layers.get_output(output_layer, X)
        loss = squared_error(prob_vector, y).mean()
        pred = T.maximum(0, T.minimum( T.round(prob_vector), args["num_classes"]-1 ) )
        accuracy = T.mean( T.eq( pred, y ) )
    else:
        a = args["a"]
        b = args["b"]
        loss_fn = get_hybrid_loss(a,b)
        prob_vector = lasagne.layers.get_output(output_layer, X)
        loss = loss_fn(prob_vector, y).mean()
        pred = T.argmax( prob_vector, axis=1 )
        accuracy = T.mean( T.eq(pred,y) )

    return Container(
        { "X": X, "y": y, "output_layer": output_layer, "all_params": all_params,
        "loss": loss, "pred": pred, "accuracy": accuracy,
        "prob_vector": prob_vector
        }
    )

def train(arg):
    global args, symbols, best_weights, SEED
    args = arg
    symbols = prepare()

    debug = False
    if "debug" in args and args["debug"]:
        debug = True

    alpha = theano.shared( args["alpha"] )
    momentum = 0.9

    if alpha != -1:
        if "rmsprop" in args:
            updates = lasagne.updates.rmsprop(symbols.loss, symbols.all_params, alpha)
        elif "adagrad" in args:
            updates = lasagne.updates.adagrad(symbols.loss, symbols.all_params, alpha)
        else:
            updates = lasagne.updates.nesterov_momentum(symbols.loss, symbols.all_params, alpha, momentum)

    iter_train = theano.function(
        [symbols.X, symbols.y],
        [symbols.prob_vector, symbols.pred, symbols.loss, symbols.accuracy],
        updates=updates
    )
    iter_valid = theano.function(
        [symbols.X, symbols.y],
        [symbols.prob_vector, symbols.pred, symbols.loss, symbols.accuracy]
    )

    args["X_train"] = np.asarray(args["X_train"], dtype="float32")
    args["y_train"] = np.asarray(args["y_train"].flatten(), dtype="int32")

    if "batch_size" in args:
        bs = args["batch_size"]
    else:
        bs = 128

    best_valid_accuracy = -1
    last_loss = -100000
    eps = 1e-6
    tol_counter = 0
    tolerance = 5
    np.random.seed(SEED)
    for e in range(0, args["epochs"]):

        # if we're doing sgd/msgd
        if bs < X_train.shape[0]:
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
            vectors, pds, loss, acc = iter_train( X_train[b*bs : (b+1)*bs], y_train[b*bs : (b+1)*bs] )
            if debug:
                print vectors[0:10], pds[0:10]
            if "expectation" in args:
                acc = sum( np.equal(expectations_rounded(vectors), y_train[b*bs : (b+1)*bs]) )
            batch_train_losses.append(loss)
            batch_train_accuracies.append(acc)

        print( "  train_loss, train_accuracy, = %f, %f\n" % \
            (np.mean(batch_train_losses), np.mean(batch_train_accuracies)) )

        if abs(np.mean(batch_train_losses) - last_loss) < eps:
            tol_counter += 1
            if tol_counter == tolerance:
                break
        else:
            last_loss = np.mean(batch_train_losses)
            tol_counter = 0

    best_weights = lasagne.layers.get_all_param_values(symbols.output_layer)

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
        symbols.prob_vector
    )

    args["X_test"] = np.asarray(args["X_test"], dtype="float32")

    if "batch_size" in args:
        bs = args["batch_size"]
    else:
        bs = 128

    X_test = args["X_test"]

    preds = iter_test(X_test).tolist()
    if "expectation" in args:
        new = []
        for pred in preds:
            label = int( round(expectation(pred)) )
            new.append( np.eye(args["num_classes"])[label].tolist() )
        return new
    else:    
        return preds

if __name__ == '__main__':
    x = ArffToArgs()
    x.set_input("data/auto_price.arff")
    x.set_class_index("last")
    x.set_impute(True)
    x.set_binarize(True)
    x.set_standardize(True)
    x.set_arguments("a=1;b=0;logistic=True;alpha=0.1;rmsprop=True;epochs=10000;batch_size=1000000")
    #x.set_arguments("a=1;b=0;regression=True;alpha=0.1;rmsprop=True;epochs=10000;batch_size=1000000")

    #x.set_arguments("a=1;b=0;logistic=True;alpha=0.1;adagrad=True;epochs=50000;batch_size=1000000")
    args = x.get_args()
    args["debug"] = False

    args["X_test"] = np.asarray(args["X_train"], dtype="float32")

    model = train(args)
