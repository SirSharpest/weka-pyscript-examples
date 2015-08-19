from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne import PrintLog

import gzip
import cPickle as pickle
import os

from skimage import io
from skimage import img_as_float

import numpy as np

import sys

import re

from cStringIO import StringIO

"""
Interesting tutorial:
http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

def remove_colour(st):
    """
    http://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
    """
    ansi_escape = re.compile(r'\x1b[^m]*m')
    return ansi_escape.sub('', st)

def load_image(filename):
    img = io.imread(filename)
    img = img_as_float(img) # don't need to do float32 conversion
    # if it's an rgb image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.asarray( [ img[..., 0], img[..., 1], img[..., 2] ] )
    else: # if it's a bw image
        img = np.asarray( [ img ] )
    return img

class Capturing(list):
    """
    HACKY: This is to allow us to capture the output of the fit()
    method when we train our network. Code retrieved from:
    http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

class FilenameToImageBatchIterator(BatchIterator):
    def __init__(self, filenames, *args, **kwds):
        super(FilenameToImageBatchIterator, self).__init__(*args, **kwds)
        self.filenames = filenames
    def transform(self, Xb, yb):
        filenames = np.asarray( [ self.filenames[int(x)] for x in Xb.flatten().tolist() ] )
        Xb_actual = np.asarray( [ load_image(x) for x in filenames ], dtype="float32" )
        return Xb_actual, yb

def get_net(filenames):
    return NeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        input_shape = (None, 1, 28, 28),
        hidden_num_units = 100,
        output_nonlinearity = softmax,
        output_num_units = 10,
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        batch_iterator_train = FilenameToImageBatchIterator(filenames, batch_size=128),
        batch_iterator_test = FilenameToImageBatchIterator(filenames, batch_size=128),
        verbose=1,
        max_epochs=1
    )  

def train(args):
    filenames = [ (args["dir"] + os.path.sep + elem) for elem in args["attr_values"]["filename"] ] 
    y_train = np.asarray(args["y_train"].flatten(), dtype="int32")
    net1 = get_net(filenames)
    X_train = args["X_train"]
    with Capturing() as output:
        model = net1.fit(X_train, y_train)
    return { "results": remove_colour("\n".join(output)),
        "params": net1.get_all_params_values() }

def describe(args, model):
    return model["results"]

def test(args, model):
    filenames = [ (args["dir"] + os.path.sep + elem) for elem in args["attr_values"]["filename"] ]
    net1 = get_net(filenames)
    net1.initialize()
    net1.load_params_from(model["params"])
    X_test = args["X_test"]
    return net1.predict_proba(X_test).tolist()

if __name__ == "__main__":

    f = gzip.open("other/mnist.meta.pkl.gz")
    args = pickle.load(f)
    args["dir"] = "data"
    f.close()
    dd =  train(args)
    args["X_test"] = args["X_train"]
    test(args, dd)
