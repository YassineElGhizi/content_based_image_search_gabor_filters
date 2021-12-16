import numpy as np
import csv

class Search:
	def __init__(self, path):
		#storing the cvs file's path
		self.path = path

	def search(self, queryFeatures, limit=101):
		results = dict()
		#opening the csv file
		with open(self.path) as f:
			reader = csv.reader(f)
			#for each element in the csv file
			for row in reader:
				#separiting the the image Name from features, and computing the chi-squared distance.
				features = [float(x) for x in row[1:]]
				d = self.chi_squared_distance(features, queryFeatures)
				results[row[0]] = d

			f.close()

		#dictionarry sort
		results = sorted(
			[(v,k) for (k,v) in results.items()]
		)
		return results[:limit]

	def chi_squared_distance(self, histA, histB, eps=1e-10):
		d = 0.5 * np.sum(
			[((a-b)**2)/(a+b+eps) for (a,b) in zip(histA, histB)]
		)
		return d