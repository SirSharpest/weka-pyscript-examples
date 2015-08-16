This repository is an example of how the MNIST dataset can be efficiently trained in WEKA using Python. The ARFF file in this example is a "meta ARFF" file - rather than having an ARFF file with attributes corresponding to pixels in the image, we simply have a string attribute that stores the location of each image, which is more memory-efficient because Python will load those images into memory incrementally and train the network.

```
unzip data.zip
```
