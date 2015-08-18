#!/bin/bash

rm mnist.meta.pkl.gz

java weka.Run weka.classifiers.pyscript.PyScriptClassifier \
    -fn ~/github/weka-pyscript/scripts/zeror.py \
    -df mnist.meta.pkl.gz \
    -t ../mnist.meta.arff \
    -no-cv > /dev/null

mv mnist.meta.pkl.gz.train mnist.meta.pkl.gz
rm mnist.meta.pkl.gz.test