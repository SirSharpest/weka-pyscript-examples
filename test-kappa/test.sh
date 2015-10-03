#!/bin/bash

NUM_EPOCHS=100
for file in `cd data; find *.arff`; do
    echo data/$file
    #java -Xmx4g weka.Run .PyScriptClassifier \
    #    -script train.py \
    #    -args "a=1;b=0;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=${NUM_EPOCHS};rmsprop=True" \
    #    -impute -binarize -standardize \
    #    -t data/$file \
    #    -c last \
    #    -no-cv

    # test regression
    java -Xmx4g weka.Run .PyScriptClassifier \
        -script train-regression.py \
        -args "adaptive=True;alpha=0.01;lambda=0;epochs=500;rmsprop=True" \
        -impute -binarize -standardize \
        -t data/$file \
        -c last \
        -no-cv
done