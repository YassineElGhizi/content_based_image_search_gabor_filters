import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import cv2


class GaborDescriptor:
	def __init__(self,params):
		#Class Constructor
		self.theta = params['theta']
		self.frequency = params['frequency']
		self.sigma = params['sigma']
		self.n_slice = params['n_slice']

	# a fucntion that return a list of gabor kernels
	def kernels(self):
		kernels = []
		for theta in range(self.theta):
			theta = theta/4. * np.pi
			for frequency in self.frequency:
				for sigma in self.sigma:
					kernel = gabor_kernel(frequency,theta=theta,sigma_x=sigma,sigma_y=sigma)
					kernels.append(kernel)
		return kernels


	def gaborHistogram(self,image,gabor_kernels):
		#extracting the Image height ,widht and the number of colors
		height,width,channel = image.shape

		"""deviding the image into four parts ,applying on each part the _gabor() : (64 ,) [because we flattened 32 gabor filter with the maean and variance ],
		also storing the results in the hist variable:numpy.ndarray with the shape of (2, 2, 64) becuase the (64 ,) distrubuted on a squares as follows :
		[0][0] ,[0][1] ,[1][0] ,[1][1].
		Finnaly devieding each cube value on the total values ,  and returning the flattened shape of the hist variable as (256,)
		"""
		hist = np.zeros((self.n_slice,self.n_slice,2*len(gabor_kernels))) #2*len because we need to store both the mean and the variance
		h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
		w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
		for hs in range(len(h_silce)-1):
			for ws in range(len(w_slice)-1):
				img_r = image[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
				# from matplotlib import pyplot as plt
				# plt.imshow(img_r, interpolation='nearest')
				# plt.show()
				hist[hs][ws] = self._gabor(img_r,gabor_kernels)

		hist /= np.sum(hist)
		return hist.flatten()

	def _power(self,image,kernel):
		image = (image - image.mean()) / image.std()
		f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 + ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
		feats = np.zeros(2, dtype=np.double)
		feats[0] = f_img.mean()
		feats[1] = f_img.var()
		return feats

	def _gabor(self,image,gabor_kernels):
		#converting image to COLOR_BGR2GRAY color space
		gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		results = []
		#for each kernel in the 32 gabor kernels
		for kernel in gabor_kernels:
			#extracting the mean and variance
			results.append(self._power(gray_img,kernel))

		hist = np.array(results)
		# hist.shape -> (2 , 32) [mean and variance for 32 gabor kernel]
		hist = hist / np.sum(hist, axis=0)
		"""returning the flattend transpose of numpy.ndarray:(2, 32) after devidied every value by of the sum of each row in the hist 
		variable ,the shape of the return is (64,) for each part of the query image
		"""
		return hist.T.flatten() # .T -> transpose && shape == (2 , 32)
