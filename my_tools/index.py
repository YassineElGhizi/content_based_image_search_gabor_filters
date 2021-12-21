import os
import cv2
from my_tools.gabor import GaborDescriptor

def index(params):
	# creating a gaborDescripto instance and its kernels
	gd = GaborDescriptor(params)
	gaborKernels = gd.kernels()
	output_file = 'index.csv'
	c = 1
	all_files = os.listdir('static/images/')  ##path relative to server.py

	#For each image in the database we will extract the Gabor kernels based vector features and saving it in a csv file
	for imagePath in all_files:
		imageId = imagePath[imagePath.rfind("/")+1:]
		image = cv2.imread("./static/images/"+imagePath)

		features = gd.gaborHistogram(image,gaborKernels)
		features = [str(f) for f in features]
		# print("c = {}".format(c))
		c += 1
		with open(output_file, 'a', encoding="utf8") as f:
			f.write("%s,%s\n" % ("static/images/"+imageId, ",".join(features)))
			f.close()



def index_one(imagepath , params):
	# creating a gaborDescripto instance and its kernels
	gd = GaborDescriptor(params)
	gaborKernels = gd.kernels()

	output_file = 'index.csv'
	image = cv2.imread(imagepath)

	# For the uploaded image ,we will extract and return the Gabor kernels based vector features and also saving it in a csv file
	features = gd.gaborHistogram(image,gaborKernels)
	feats = [str(f) for f in features]
	with open(output_file, 'a', encoding="utf8") as f:
		f.write("%s,%s\n" % (imagepath, ",".join(feats)))
		f.close()
	return  features