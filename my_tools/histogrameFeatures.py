import numpy as np
import cv2

class HistogrameFeatures:
    def __init__(self , bins):
        """bins of the 3D histograme"""
        self.bins = bins

    def features(self ,image):
        """converting image to a HSV color space"""
        features = list()

        image = cv2.cvtColor(image ,cv2.COLOR_BGR2HSV)
        # Grid number of the image center
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w / 2), int(h / 2))
        #deviding the image into four segemtns
        image_four_segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]

        # constructing an elliptical mask representing the center of the image
        ellipMask = self.ellipceMask(w,h,image,cX,cY)

        for (startX, endX, startY, endY) in image_four_segments:
            # constructing a mask for each segment of the image minus the centre of the ellipce
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extracting a color histogram from the image and affecting the value to the featuers list
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extracting the color histogram from the elliptical region and update the features list
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        return features

    def ellipceMask(self, w, h, image , cX , cY):
        (ellipX, ellipY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (ellipX, ellipY), 0, 0, 360, 255, -1)
        return ellipMask

    def histogram(self, image, mask):
        """extracting a 3D color histogram from the masked region of the image"""
        hist = cv2.calcHist(
            [image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        return hist

