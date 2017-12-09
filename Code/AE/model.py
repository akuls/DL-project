import PIL
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import numpy as np
import random
from torchvision import transforms
import sys
sys.path.append('../../Config')
from constants import HAVE_CUDA, BUCKET_SIZE, EMBEDDING_DIM, SIDELENGTH

if HAVE_CUDA:
	import torch.cuda as cuda

class AutoEncoder(nn.Module):
	"""docstring for AutoEncoder"""
	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.encode1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=64, out_channels=256, kernel_size=2, stride=1),
			nn.ReLU(True)
		)
		self.encoder = nn.Sequential(
			self.encode1,
			nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
		)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=1, stride=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=6, stride=2, padding=5),
			nn.ReLU(True)
		)

	def forward(self, x):
		y = self.encoder(x)
		x_cap = self.decoder(y)
		return x_cap

	def Encode(self,x):
		return self.encoder(x)

	def Decode(self,x):
		return self.decoder(x)

	def get_feature_vector(self,x):
		return self.encode1(x)

	def get_intermediate_vector(self,x):
		return np.squeeze(self.Encode(x))

class AutoEncoder_experimental(nn.Module):
	"""docstring for AutoEncoder"""
	def __init__(self):
		super(AutoEncoder_experimental, self).__init__()
		self.encode1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=64, out_channels=256, kernel_size=2, stride=1),
			nn.ReLU(True)
		)
		self.encoder = nn.Sequential(
			self.encode1,
			nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
		)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=1, stride=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=6, stride=2, padding=5),
			nn.ReLU(True)
		)

	def forward(self, x):
		N,C,H,W = x.size()
		#print "input received", x
		y = self.encoder(x)
		#print "intermediate", y
		x_cap = self.decoder(y)
		#print "resultant", x_cap
		return x_cap

	def Encode(self,x):
		return self.encoder(x)

	def Decode(self,x):
		return self.decoder(x)

	def get_feature_vector(self,x):
		return self.encode1(x)

	def get_intermediate_vector(self,x):
		return np.squeeze(self.Encode(x))

