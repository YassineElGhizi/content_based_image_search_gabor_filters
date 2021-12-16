import os
import numpy as np
import glob
import cv2
import argparse
from my_tools.gabor import GaborDescriptor

def train():
	params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}
	gd = GaborDescriptor(params)
	gaborKernels = gd.kernels()
	output_file = 'index.csv'
	c = 1
	all_files = os.listdir('static/images/')  ##path relative to server.py

	for imagePath in all_files:
		imageId = imagePath[imagePath.rfind("/")+1:]
		image = cv2.imread("./static/images/"+imagePath)

		features = gd.gaborHistogram(image,gaborKernels)
		features = [str(f) for f in features]
		print("c = {}".format(c))
		c += 1
		with open(output_file, 'a', encoding="utf8") as f:
			f.write("%s,%s\n" % ("static/images/"+imageId, ",".join(features)))
			f.close()



def train_one(imagepath):

	params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}
	gd = GaborDescriptor(params)
	gaborKernels = gd.kernels()
	output_file = 'index.csv'
	image = cv2.imread(imagepath)

	features = gd.gaborHistogram(image,gaborKernels)
	features = [str(f) for f in features]

	print("output_file = {}".format(output_file))
	with open(output_file, 'a', encoding="utf8") as f:
		f.write("%s,%s\n" % (imagepath, ",".join(features)))
		f.close()
