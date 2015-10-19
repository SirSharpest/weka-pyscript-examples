#!/bin/bash

NUM_EPOCHS=100
for file in `cd data; find *.arff`; do
    echo data/$file
    #java -Xmx4g weka.Run .PyScriptClassifier \
    #    -script train.py \
    #    -args "expectation=True;a=1;b=0;logistic=True;alpha=0.1;schedule=500;epochs=5000;" \
    #    -impute -binarize -standardize \
    #    -t data/$file \
    #    -c last \
    #    -no-cv

    # test regression
    #java -Xmx4g weka.Run .PyScriptClassifier \
    #    -script train.py \
    #    -args "regression=True;alpha=0.1;rmsprop=True;epochs=5000" \
    #    -impute -binarize -standardize \
    #    -t data/$file \
    #    -c last \
    #    -no-cv

    python train.py data/$file kappa
    #python train.py data/$file regression
done
