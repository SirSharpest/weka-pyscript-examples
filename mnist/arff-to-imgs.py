import numpy as np
import scipy
from scipy import misc

f = open("mnist-dense.arff")
for line in f:
    if "@data" in line:
        break

i = 0
for line in f:
    print i
    line = [ int(x) for x in line.rstrip().split(",") ]
    pixels = line[0: len(line)-1]
    class_label = line[ len(line)-1 ]

    arr = np.asarray(pixels)
    arr = arr.reshape( (28, 28) )

    misc.toimage(arr, cmin=0, cmax=255).save("data/" + str(i) + "_c" + str(class_label) + ".gif")

    i += 1



f.close()
