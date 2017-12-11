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


class JointNet(nn.Module):
	"""docstring for JointNet"""
	def __init__(self, embedding_dim=256, num_users=39387):
		super(JointNet, self).__init__()
		self.user_embed = nn.Embedding(num_users, embedding_dim)
		
		#Image network
		self.item_CNN = nn.Sequential(
		nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
		nn.ReLU(True),
		#Result- 48*48
		nn.MaxPool2d(kernel_size=2, stride=2),
		#Resut- 24*24

		nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1),
		nn.ReLU(True),
		#Result- 22*22

		nn.Conv2d(in_channels=256, out_channels=1, kernel_size=7, stride=1),
		nn.BatchNorm2d(1),
		# nn.ReLU(True)
		# nn.Tanh()
		#Result- 16*16
		)

		#User network
		self.user_FCN = nn.Sequential(
		nn.Linear(embedding_dim, 512),
		nn.BatchNorm1d(512),
		nn.ReLU(),

		nn.Linear(512, 256),
		nn.BatchNorm1d(256),
		)

		#Recommender network
		self.user_item_FCN = nn.Sequential(
		# nn.Linear(512, 1024),
		# nn.BatchNorm1d(1024),
		# nn.ReLU(),

		# nn.Linear(1024, 512),
		# nn.BatchNorm1d(512),
		# nn.ReLU(),

		nn.Linear(512, 256),
		nn.BatchNorm1d(256),
		nn.ReLU(),

		nn.Linear(256, 1),
		# nn.BatchNorm1d(1),
		# nn.ReLU(),

		# nn.Linear(64, 1),
		nn.BatchNorm1d(1),
		nn.Sigmoid()
		)

	def forward(self, images, user_idx):
		user_vec = np.squeeze(self.user_embed(user_idx))

		y_item_CNN = self.item_CNN(images)
		y_item_CNN = y_item_CNN.view(y_item_CNN.size()[0], y_item_CNN.size()[1]*y_item_CNN.size()[2]*y_item_CNN.size()[3])
		# print 'Item net output size', y_item_CNN.size()

		# print '+++++++++++++++++++++++++++++++++++++++++ Before Combine +++++++++++++++++++++++++++++++++++++'
		# print y_item_CNN.data[0]
		y_user_FCN = self.user_FCN(user_vec)
		# print 'User net output size', y_user_FCN.size()

		x_user_item_FCN = self.combine_user_data(y_user_FCN, y_item_CNN)
		# print 'User_item input size', x_user_item_FCN.size()

		# print '+++++++++++++++++++++++++++++++++++++++++ After Combine +++++++++++++++++++++++++++++++++++++'
		# print x_user_item_FCN.data[0]
		y_pred = self.user_item_FCN(x_user_item_FCN)
		#y_pred = self.cosine_similarity(y_user_FCN, y_item_CNN)
		return y_pred

	def get_embeddding(self):
		return self.user_embed

	def set_user_embed(self, embed):
		self.user_embed = embed

	def combine_user_data(self, user, item):
		res = torch.cat((user, item),1)
		return res

	def print_user_data(self, user_idx):
		user_vec = np.squeeze(self.user_embed(user_idx))
		print user_vec

	def cosine_similarity(self, v1, v2):
		v1 = v1 / v1.norm(2, 1).clamp(min=1e-12).expand_as(v1)
		v2 = v2 / v2.norm(2,1).clamp(min=1e-12).expand_as(v2)
		res = torch.sum(v1*v2, dim=1)
		res += 1.0
		res /= 2.0
		return res

class DeepJointNet(nn.Module):
	"""docstring for JointNet"""
	def __init__(self, embedding_dim=256, num_users=39387):
		super(DeepJointNet, self).__init__()
		self.user_embed = nn.Embedding(num_users, embedding_dim)
		
		#Image network
		self.item_CNN = nn.Sequential(
		nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
		nn.BatchNorm2d(32),
		nn.ReLU(True),
		#Result- 46*46
		nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1),
		nn.BatchNorm2d(64),
		nn.ReLU(True),
		#Result- 43*43
		nn.MaxPool2d(kernel_size=3, stride=1),
		#Resut- 41*41

		nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=1),
		nn.BatchNorm2d(256),
		nn.ReLU(True),
		#Result- 38*38
		
		nn.MaxPool2d(kernel_size=2, stride=2),
		#Result- 19*19

		nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1),
		nn.BatchNorm2d(1),
		# nn.ReLU(True)
		# nn.Tanh()
		#Result- 16*16
		)

		#User network
		self.user_FCN = nn.Sequential(
		nn.Linear(embedding_dim, 512),
		nn.BatchNorm1d(512),
		nn.ReLU(),

		nn.Linear(512, 512),
		nn.BatchNorm1d(512),
		nn.ReLU(),

		nn.Linear(512, 256),
		nn.BatchNorm1d(256),
		)

		#Recommender network
		self.user_item_FCN = nn.Sequential(
		nn.Linear(512, 1024),
		nn.BatchNorm1d(1024),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(1024, 512),
		nn.BatchNorm1d(512),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(512, 256),
		nn.BatchNorm1d(256),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(256, 128),
		nn.BatchNorm1d(128),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(128, 64),
		nn.BatchNorm1d(64),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(64, 32),
		nn.BatchNorm1d(32),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(32, 1),
		nn.BatchNorm1d(1),
		nn.Sigmoid()
		)

	def forward(self, images, user_idx):
		user_vec = np.squeeze(self.user_embed(user_idx))

		y_item_CNN = self.item_CNN(images)
		y_item_CNN = y_item_CNN.view(y_item_CNN.size()[0], y_item_CNN.size()[1]*y_item_CNN.size()[2]*y_item_CNN.size()[3])
		# print 'Item net output size', y_item_CNN.size()

		# print '+++++++++++++++++++++++++++++++++++++++++ Before Combine +++++++++++++++++++++++++++++++++++++'
		# print y_item_CNN.data[0]
		y_user_FCN = self.user_FCN(user_vec)
		# print 'User net output size', y_user_FCN.size()

		x_user_item_FCN = self.combine_user_data(y_user_FCN, y_item_CNN)
		# print 'User_item input size', x_user_item_FCN.size()

		# print '+++++++++++++++++++++++++++++++++++++++++ After Combine +++++++++++++++++++++++++++++++++++++'
		# print x_user_item_FCN.data[0]
		y_pred = self.user_item_FCN(x_user_item_FCN)
		#y_pred = self.cosine_similarity(y_user_FCN, y_item_CNN)
		return y_pred

	def get_embeddding(self):
		return self.user_embed

	def set_user_embed(self, embed):
		self.user_embed = embed

	def combine_user_data(self, user, item):
		res = torch.cat((user, item),1)
		return res

	def print_user_data(self, user_idx):
		user_vec = np.squeeze(self.user_embed(user_idx))
		print user_vec

	def cosine_similarity(self, v1, v2):
		v1 = v1 / v1.norm(2, 1).clamp(min=1e-12).expand_as(v1)
		v2 = v2 / v2.norm(2,1).clamp(min=1e-12).expand_as(v2)
		res = torch.sum(v1*v2, dim=1)
		res += 1.0
		res /= 2.0
		return res
