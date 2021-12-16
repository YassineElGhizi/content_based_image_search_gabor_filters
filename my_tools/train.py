# from histogrameFeatures import HistogrameFeatures
from my_tools.histogrameFeatures import HistogrameFeatures
import cv2
import os

def train():
    images = []
    output_file = 'index.csv'
    hsv_init = (8,12,3)

    #initializing the class HistogrameFeatures
    histfeat = HistogrameFeatures(hsv_init)
    print("type of features = {}".format(histfeat))

    #getting all the images
    # all_files = os.listdir('../images/')
    all_files = os.listdir('static/images/') ##because this unction are gonna be used from server.py
    print("all_files = {}".format(all_files))
    for f in all_files:
        # if '.png' in f:
            # images.append('../images/' + str(f))
        images.append('static/images/' + str(f))


    #saving the features in a csv file
    for image in images:
        tmp_image = cv2.imread(image)
        features = histfeat.features(tmp_image)
        #writing the features to a csv file
        features = [str(f) for f in features]
        with open(output_file , 'a' , encoding="utf8") as f:
            f.write("%s,%s\n" % (image, ",".join(features)))
            f.close()
