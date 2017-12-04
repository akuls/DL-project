import torch.optim as optim
import sys
import os
import time
import numpy as np
import collections
from torchvision import transforms
import PIL
from PIL import Image
import os
import torch
import torch.nn as nn
sys.path.append('../AE/')
import model as md
import torch.autograd as ag
sys.path.append('../../Config')
from constants import *
import random

def generate_user_id_file():
	datafile = open("../../Data/pairs.txt","r")
	user_file = open("../../Data/user_ids.txt", "w")

	for l in datafile:
		user_id, items, empty = l.split(" ||| ")
		user_file.write(user_id + "\n")

	user_file.close()
	datafile.close()

def generate_index_file_for(source, dest):
	input_file = open(source, "r")
	output_file = open(dest, "w")
	
	index = 0
	for line in input_file:
		output_file.write(line.rstrip() + "," + str(index) + "\n")
		index += 1

	output_file.close()
	input_file.close()	

def get_dicts_from_text(filename):
	input_file = open(filename, "r")

	data = {}
	for line in input_file:
		key, value = line.split(',')
		data[key] = value.rstrip()
	return data
	pass

def generate_train_test_split(user_dict, item_dict, train_pct=0.8):
	
	datafile = open("../../Data/pairs.txt","r")

	user_item_train = open("../../Data/user_item_train.txt","w")
	user_item_test = open("../../Data/user_item_test.txt","w")

	for l in datafile:
		user, items, t = l.split(" ||| ")
		items = items.split(',')
		
		for i in range(len(items)):
			items[i] = item_dict[items[i]]

		tr_end = int(train_pct*len(items))
		if(tr_end == len(items)):
			tr_end -=1
		
		train = items[:tr_end]
		test = items[tr_end:]

		user_item_train.write(user_dict[user] + " " + ",".join(train) +"\n")
		user_item_test.write(user_dict[user] + " " + ",".join(test) + "\n")

	user_item_train.close()
	user_item_test.close()
	pass

def get_dict_from_index_mapping(filename):

	datafile = open(filename,"r")
	data = {}
	for l in datafile:
		user_idx, item_idxs = l.split(' ')
		items = item_idxs.split(',')
		items[len(items)-1] = items[len(items)-1].rstrip()
		data[user_idx] = items
	return data	

def get_ids_from_file(filename):
	datafile = open(filename,"r")
	data = []
	for l in datafile:
		Id, Idx = l.split(',')
		data.append(Id)
	return data

def image_id_to_variable(item_id):
	tt = transforms.ToTensor()
	image = Image.open("../../Data/Resize_images_50/"+item_id.rstrip()+".jpg")			
	item_image = ag.Variable(tt(image)).view(1,-1,SIDELENGTH,SIDELENGTH)
	image.close()
	return item_image

def image_ids_to_variable(item_ids):
	i = 0
	for item_id in item_ids:
		item_variable = image_id_to_variable(item_id)
		if i == 0:
			image_variables = item_variable
			i += 1
		else:
			image_variables = torch.cat((image_variables,item_variable),0)
	return image_variables


def get_image_vectors(model,image_ids=None):
	"""
	model :: an autoencoder model that has a separate encoder and decoder function
	image_ids :: The labels (item_ids) of the images
	"""
	if image_ids == None:
		image_ids = get_ids_from_file("../../Data/item_to_index.txt")[0:1000]
	image_variables = image_ids_to_variable(image_ids)
	image_vectors = (model.get_intermediate_vector(image_variables)).view(len(image_ids),-1)
	return image_vectors

def get_user_vectors(filename="",embedding_dim=100,num_users=39387):
	if os.path.isfile(filename):
		return torch.load(filename)
	else:
		return torch.nn.Embedding(num_users,embedding_dim)

def get_random_from_tuple_list(data, batch_size=0):
	indexes = random.sample(range(0, len(data)-1), batch_size)
	return [data[k] for k in indexes]

def save_user_vectors(user_vectors,filename):
	torch.save(user_vectors,filename)


def loadAE(filename=None):
	# Load AutoEncoder
	if os.path.isfile(filename):
		AE = torch.load(filename)
	else:
		AE = md.AutoEncoder()
	return AE

def loadOptimizer(MODEL, filename=None):
	if filename is not None and os.path.isfile(filename):
		optimizer = torch.load(filename)
	else:
		optimizer = optim.Adam(MODEL.parameters(), lr=0.001)

	return optimizer

if __name__ == '__main__':
	#One time run
	# generate_user_id_file()

	# generate_index_file_for("../../Data/user_ids.txt", "../../Data/user_to_index.txt")
	# generate_index_file_for("../../Data/item_ids.txt", "../../Data/item_to_index.txt")

	# generate_train_test_split(get_dicts_from_text('../../Data/user_to_index.txt'), get_dicts_from_text('../../Data/item_to_index.txt'))

	# train_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")
	# test_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")

	# item_ids_in_order_of_idx = get_ids_from_file("../../Data/item_to_index.txt")
	# user_ids_in_order_of_idx = get_ids_from_file("../../Data/user_to_index.txt")
	# print len(user_ids_in_order_of_idx)

	AE = loadAE('../AE/Checkpoints/auto_encoder2')