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
sys.path.append('../Data Handler')
from constants import HAVE_CUDA, BUCKET_SIZE, EMBEDDING_DIM, SIDELENGTH

if HAVE_CUDA:
	import torch.cuda as cuda


class FeedForward(nn.Module):
	"""docstring for FeedForward"""
	def __init__(self,embedding_dim=100,num_users=39387):
		super(FeedForward, self).__init__()
		self.user_embed = nn.Embedding(num_users,embedding_dim)
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
		nn.BatchNorm1d(1),
		nn.Sigmoid()
		)

	def forward(self, item_vec, user_idx):
		akul
		user_vec = np.squeeze(self.user_embed(user_idx))
		x = self.combine_user_data(user_vec,item_vec)
		# print x
		y_pred = self.network(x)
		return y_pred

	def get_embeddding(self):
		return self.user_embed

	def set_user_embed(self,embed):
		self.user_embed = embed

	def combine_user_data(self, user, item):
		res = torch.cat((user, item),1)
		return res

	def print_user_data(self,user_idx):
		user_vec = np.squeeze(self.user_embed(user_idx))
		print user_vec