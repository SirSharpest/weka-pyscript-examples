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

    # decay/schedules are optional and mutually exclusive
    assert not ("decay" in args and "schedule" in args)
    decay = -1
    schedule = -1
    if "decay" in args:
        decay = args["decay"]
    if "schedule" in args:
        schedule = args["schedule"]

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

    X_train = args["X_train"]
    y_train = args["y_train"]

    best_valid_accuracy = -1
    last_loss = -100000
    #eps = 1e-6
    eps = 0.01
    tol_counter = 0
    tolerance = 100
    np.random.seed(SEED)
    for e in range(0, args["epochs"]):

        sys.stderr.write("Epoch #%i:\n" % e)
        vectors, pds, loss, acc = iter_train( X_train, y_train )
        if debug:
            print vectors[0:10], pds[0:10]

        print( "  train_loss, train_accuracy, = %f, %f\n" % (loss, acc) )

        """
        if "decay" in args:
            # decay formula: alpha_new = alpha / (1 + epoch*decay_rate)
            alpha.set_value( args["alpha"] / (1 + ((e+1)*decay)) )
            print "new alpha: %f" % alpha.get_value()
        """
        if "schedule" in args:
            if (e+1) % schedule == 0:
                alpha.set_value( alpha.get_value() / 2 )
                print "new alpha: %f" % alpha.get_value()
        if abs(loss - last_loss) < eps: # if they're similar
            tol_counter += 1
            # if the loss has been the "same" for the past `tolerence` iterations
            # then stop
            if tol_counter == tolerance:
                break
        else:
            tol_counter = 0
            last_loss = loss

    best_weights = lasagne.layers.get_all_param_values(symbols.output_layer)

    return best_weights

def describe(arg, weights):
    return "blank"

def test(arg, weights):

    global args, symbols, iter_test
    args = arg

    symbols = prepare()
    lasagne.layers.set_all_param_values(symbols.output_layer, weights)

    if "regression" in args:
        iter_test = theano.function(
            [symbols.X],
            symbols.pred
        )
    else:
        iter_test = theano.function(
            [symbols.X],
            symbols.prob_vector
        )

    args["X_test"] = np.asarray(args["X_test"], dtype="float32")
    X_test = args["X_test"]

    preds = iter_test(X_test).tolist()

    if "regression" in args:
        new = []
        for pred in preds:
            new.append( np.eye(args["num_classes"])[pred].tolist() )
        return new
    else:
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
    #x.set_input("data/auto_price.arff")
    if len(sys.argv) != 3:
        sys.argv.append("data/2dplanes.arff")
        sys.argv.append("kappa")
    x.set_input( sys.argv[1] )
    print "Training on: %s" % sys.argv[1]
    x.set_class_index("last")
    x.set_impute(True)
    x.set_binarize(True)
    x.set_standardize(True)
    if sys.argv[2] == "kappa":
        #x.set_arguments("expectation=True;a=1;b=0;logistic=True;alpha=0.1;rmsprop=True;epochs=5000")
        x.set_arguments("expectation=True;a=1;b=0;logistic=True;alpha=0.1;schedule=500;epochs=5000")
    elif sys.argv[2] == "regression":
        #x.set_arguments("regression=True;alpha=0.1;rmsprop=True;epochs=5000")
        x.set_arguments("regression=True;alpha=0.1;schedule=500;epochs=5000")
    else:
        print "error!"
    args = x.get_args()
    args["debug"] = False

    args["X_test"] = np.asarray(args["X_train"], dtype="float32")

    model = train(args)
