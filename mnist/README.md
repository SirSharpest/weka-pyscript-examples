This repository is an example of how the MNIST dataset can be efficiently trained in WEKA using Python. The ARFF file in this example is a "meta ARFF" file - rather than having an ARFF file with attributes corresponding to pixels in the image, we simply have a string attribute that stores the location of each image, which is more memory-efficient because we won't have the MNIST dataset sitting in the JVM when all the model training is actually happening in Python. This example also trains MNIST in an incremental fashion, which makes things more memory-efficient at the cost of speed.

To get started, we have to extract the .zip file containing the digits:

```
unzip data.zip
```

Then, simply run:

```
java weka.Run weka.classifiers.pyscript.PyScriptClassifier \
    -pc python \
    -fn mnist-lasagne.py \
    -xp "'lambda'=0,'alpha'=0.01,'epochs'=10,'dir'='data'" \
    -yp "'lambda'=0,'alpha'=0.01,'epochs'=10,'dir'='data'" \
    -t mnist.meta.arff -no-cv
```

