import lasagne
import theano
from theano import tensor as T
from lasagne.objectives import categorical_crossentropy as x_ent
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

def prepare():

    X = T.fmatrix('X')
    y = T.ivector('y')

    if "logistic" not in args:
        output_layer = network()
    else:
        output_layer = logistic()

    all_params = lasagne.layers.get_all_params(output_layer)

    a = args["a"]
    b = args["b"]
    loss_fn = get_hybrid_loss(a,b)

    prob_vector = lasagne.layers.get_output(output_layer, X)

    loss = loss_fn(prob_vector, y).mean() + args["lambda"]*reg(output_layer, l2)
    alt_loss = x_ent(prob_vector, y).mean() + args["lambda"]*reg(output_layer, l2)

    pred = T.argmax( prob_vector, axis=1 )
    accuracy = T.mean( T.eq(pred,y) )

    return Container(
        { "X": X, "y": y, "output_layer": output_layer, "all_params": all_params,
        "loss": loss, "pred": pred, "accuracy": accuracy, "alt_loss": alt_loss,
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
        [symbols.prob_vector, symbols.pred, symbols.loss, symbols.accuracy],
        updates=updates
    )
    iter_valid = theano.function(
        [symbols.X, symbols.y],
        [symbols.prob_vector, symbols.pred, symbols.loss, symbols.accuracy]
    )

    iter_alt_loss = theano.function(
        [symbols.X, symbols.y],
        symbols.alt_loss
    )

    args["X_train"] = np.asarray(args["X_train"], dtype="float32")
    args["y_train"] = np.asarray(args["y_train"].flatten(), dtype="int32")

    idxs = [x for x in range(0, args["X_train"].shape[0])]
    X_train = args["X_train"][idxs]
    y_train = args["y_train"][idxs]

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
        batch_train_alt_losses = []
        for b in range(0, X_train.shape[0]):
            if b*bs >= X_train.shape[0]:
                break
            #sys.stderr.write("  Batch #%i (%i-%i)\n" % ((b+1), (b*bs), ((b+1)*bs) ))
            vectors, _, loss, acc = iter_train( X_train[b*bs : (b+1)*bs], y_train[b*bs : (b+1)*bs] )

            if "expectation" in args:
                acc = sum( np.equal(expectations_rounded(vectors), y_train[b*bs : (b+1)*bs]) )

            batch_train_losses.append(loss)
            batch_train_accuracies.append(acc)

            batch_train_alt_losses.append( iter_alt_loss(X_train[b*bs : (b+1)*bs], y_train[b*bs : (b+1)*bs]) )

        batch_valid_losses = []
        batch_valid_accuracies = []

        #kappa = weighted_kappa(all_preds, y_valid.tolist())
        kappa = 0

        batch_valid_losses = [0]
        valid_mean = 0
        best_weights = lasagne.layers.get_all_param_values(symbols.output_layer)

        sys.stderr.write( "  train_loss, train_accuracy, = %f, %f\n" % \
            (np.mean(batch_train_losses), np.mean(batch_train_accuracies)) )
        g.write( str(np.mean(batch_train_losses)) + "," + str(np.mean(batch_train_alt_losses)) + "\n" )

    g.close()

    return best_weights

def describe(arg, weights):
    return "blank"

def test(arg, weights):

    global args, first_time_test, symbols, iter_test
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
    x.set_input("data/cpu_act.arff")
    x.set_class_index("last")
    x.set_impute(True)
    x.set_binarize(True)
    x.set_standardize(True)
    x.set_arguments("a=1;b=0;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=500;rmsprop=True")
    args = x.get_args()
    args["debug"] = True

    args["X_test"] = np.asarray(args["X_train"], dtype="float32")

    model = train(args)
