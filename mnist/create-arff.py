import sys
import glob
import os

print "@relation mnist"
print "@attribute filename string"
print "@attribute class {0,1,2,3,4,5,6,7,8,9}"
print "@data"

gifs = glob.glob("data/*.gif")
for filename in gifs:
    filename = os.path.basename(filename)
    class_label = filename.split("_")[1].replace("c","").replace(".gif","")
    #print "%s,%s" % (filename[0]+".gif", class_label)
    print filename + "," + class_label
