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
		self.fc1 = nn.Linear(28 * 28, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, 10)


	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x)