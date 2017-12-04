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


class FeedForward(nn.Module):
	"""docstring for FeedForward"""
	def __init__(self):
		super(FeedForward, self).__init__()
		
		self.network = nn.Sequential(
		nn.Linear(200, 300),
		nn.BatchNorm1d(300),
		nn.ReLU(),

		nn.Linear(300, 400),
		nn.BatchNorm1d(400),
		nn.ReLU(),
		
		nn.Linear(400, 300),
		nn.BatchNorm1d(300),
		nn.ReLU(),

		nn.Linear(300, 200),
		nn.BatchNorm1d(200),
		nn.ReLU(),

		nn.Linear(200, 100),
		nn.BatchNorm1d(100),
		nn.ReLU(),
		
		nn.Linear(100, 1),
		nn.Sigmoid()
		)

	def forward(self, x):
		y_pred = self.network(x)
		return y_pred
