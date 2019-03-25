import numpy
import os
import sys
import util
from urlparse import urljoin
import gzip
import struct
import operator
import numpy as np
from scipy.io import loadmat
def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]
class USPS:
	base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'
	data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

	def __init__(self,path=None,shuffle=True,output_size=[28,28],output_channel=1,split='train',select=[]):
		self.image_shape=(16,16,1)
		self.label_shape=()	
		self.path=path
		self.shuffle=shuffle
		self.output_size=output_size
		self.output_channel=output_channel
		self.split=split
		self.select=select
		self.download()
		self.pointer=0
		self.load_dataset(self.select)
	def download(self):
		data_dir = self.path
        	if not os.path.exists(data_dir):
            		os.mkdir(data_dir)
        	for filename in self.data_files.values():
            		path = self.path+'/'+filename
            		if not os.path.exists(path):
                		url = urljoin(self.base_url, filename)
                		util.maybe_download(url, path)
        def shuffle_data(self):
		images = self.images[:]
        	labels = self.labels[:]
        	self.images = []
        	self.labels = []

        	idx = np.random.permutation(len(labels))
        	for i in idx:
            		self.images.append(images[i])
            		self.labels.append(labels[i])
	def _read_datafile(self,path):
		"""Read the proprietary USPS digits data file."""
        	labels, images = [], []
        	with gzip.GzipFile(path) as f:
        	    for line in f:
        	        vals = line.strip().split()
        	        labels.append(float(vals[0]))
        	        images.append([float(val) for val in vals[1:]])
        	labels = np.array(labels, dtype=np.int32)
        	labels[labels == 10] = 0  # fix weird 0 labels
        	images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        	images = (images + 1) / 2
        	return images, labels
	def load_dataset(self,select):
		abspaths = {name: self.path+'/'+path
                	for name, path in self.data_files.items()}

		if self.split=='train':
			train_images,train_labels = self._read_datafile(abspaths['train'])
			self.images = train_images[select]
        		self.labels = train_labels[select]
			if len(select)==0:	
				self.images=train_images
				self.labels=train_labels
        	elif self.split=='test':
			test_images,test_labels = self._read_datafile(abspaths['test'])
			self.images=test_images[select]
			self.labels=test_labels[select]
			if len(select)==0:	
				self.images=test_images
				self.labels=test_labels
	def reset_pointer(self):
		self.pointer=0
		if self.shuffle:
			self.shuffle_data()	

	def class_next_batch(self,num_per_class):
		batch_size=10*num_per_class
		classpaths=[]
		ids=[]
		for i in xrange(10):
			classpaths.append([])
		for j in xrange(len(self.labels)):
			label=self.labels[j]
			classpaths[label].append(j)
	        for i in xrange(10):
			ids+=np.random.choice(classpaths[i],size=num_per_class,replace=False).tolist()
		selfimages=np.array(self.images)
		selflabels=np.array(self.labels)
		return np.array(selfimages[ids]),get_one_hot(selflabels[ids],10)

	def next_batch(self,batch_size):
		if self.pointer+batch_size>=len(self.labels):
			self.reset_pointer()
		images=self.images[self.pointer:(self.pointer+batch_size)]
		labels=self.labels[self.pointer:(self.pointer+batch_size)]
		self.pointer+=batch_size
		return np.array(images),get_one_hot(labels,10)	

def main():
	svhn=USPS(path='data/usps',split='train',select=[2,3,45])
	print len(svhn.images)
	svhn=USPS(path='data/usps',split='test')
	print len(svhn.images)
	a,b=svhn.next_batch(1)
	#print a
	print b

if __name__=='__main__':
	main()
