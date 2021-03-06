import torch.optim as optim
import sys
import os
import time
import numpy as np
print np.__path__
import collections
from torchvision import transforms
import PIL
from PIL import Image
import os
import torch
import torch.nn as nn
sys.path.append('../AE/')
sys.path.append('../RCMDR/')
import model as md
import rcmdr_model as rmd
import torch.autograd as ag
sys.path.append('../../Config')
from constants import *
import random
import struct
if HAVE_CUDA:
	import torch.cuda as cuda

def format_time(t):
	out_str = ""
	s = int(t%60)
	out_str = str(s) +"s"
	t = t/60
	if t==0:
		return out_str
	m = int(t%60)
	out_str = str(m) +"m " + out_str
	t = t/60
	if t==0:
		return out_str
	h = int(t)
	out_str = str(h) +"h " + out_str
	return out_str
	

def time_remaining(start_time, total_iterations, completed_iterations):
	elapsed_time = time.time()-start_time
	print "Time Elapsed: ", format_time(elapsed_time)
	remaining_time = elapsed_time*(total_iterations-completed_iterations)/completed_iterations
	print "Time Remaining: ",format_time(remaining_time)

def generate_user_id_file():
	"""
	One time run function
	"""
	datafile = open("../../Data/pairs.txt","r")
	user_file = open("../../Data/user_ids.txt", "w")

	for l in datafile:
		user_id, items, empty = l.split(" ||| ")
		user_file.write(user_id + "\n")

	user_file.close()
	datafile.close()

def generate_index_file_for(source, dest):
	"""
	One time run function
	"""
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
	"""
	One time run function
	"""
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
	"""
	filename: test or train indexes file
	returns a dict of {user_index:[item_indexes]}
	"""
	datafile = open(filename,"r")
	data = {}
	for l in datafile:
		user_idx, item_idxs = l.split(' ')
		items = item_idxs.split(',')
		items[len(items)-1] = items[len(items)-1].rstrip()
		data[user_idx] = items
	return data	

def get_ids_from_file(filename):
	"""
	filename: user or item's id->index file 
	For eg: 
	BFCCCXXX003, 0
	BDCGTRXX003, 1
	DGF1CXXX003, 2
	returns a list of image or user ids in order of indexes
	"""
	datafile = open(filename,"r")
	data = []
	for l in datafile:
		Id, Idx = l.split(',')
		data.append(Id)
	return data

def image_id_to_variable(item_id):
	"""
	input: item id
	output: Torch variable of that image
	"""
	tt = transforms.ToTensor()
	image = Image.open("../../Data/Resize_images_50/"+item_id.rstrip()+".jpg")			
	item_image = ag.Variable(tt(image)).view(1,-1,SIDELENGTH,SIDELENGTH)
	return item_image

def image_ids_to_variable(item_ids):
	"""
	input: list of image ids
	output: list of torch variables for all the images
	"""
	image_variables = ag.Variable(torch.zeros(len(item_ids),3,50,50))
	for i in range(len(item_ids)):
		sh = image_id_to_variable(item_ids[i]).shape
		print sh
		item_variable = torch.squeeze(image_id_to_variable(item_ids[i]))
		image_variables[i] = item_variable
		# print item_variable
		# break
		# if i == 0:
		# 	image_variables = item_variable
		# 	i += 1
		# else:
		# 	image_variables = torch.cat((image_variables,item_variable),0)
	return image_variables


def get_image_vectors(model, image_ids=None, filename=""):
	"""
	model :: an autoencoder model that has a separate encoder and decoder function
	image_ids :: The labels (item_ids) of the images
	"""
	if os.path.isfile(filename):
		return torch.load(filename)
	else:
		if image_ids == None:
			image_ids = get_ids_from_file("../../Data/item_to_index.txt")
		pt = time.time()
		data_len = len(image_ids)
		for i in range(data_len/1000):
			image_variables = image_ids_to_variable(image_ids[1000*(i):1000*(i+1)])
			image_vectors = (model.get_intermediate_vector(image_variables)).view(1000,-1)
			torch.save(image_vectors,"../../Data/image_vectors_"+str(i+1))
			print i
			et = time.time()
			print et-pt
			# image_vectors = torch.cat((image_vectors,image_vectors_temp),0)
		image_variables = image_ids_to_variable(image_ids[1000*(data_len/1000):data_len])
		image_vectors = (model.get_intermediate_vector(image_variables)).view(data_len-1000*(data_len/1000),-1)
		torch.save(image_vectors,"../../Data/image_vectors_"+str(data_len/1000+1))
		et = time.time()
		print et-pt
		tf = 0
		image_vectors = ag.Variable(torch.zeros(data_len,100))
		for filename in os.listdir("../../Data/"):
			if filename.startswith("image_vectors_"): 
				file_id =  int(filename[14:])
				# print file_id
				image_vectors[1000*(file_id-1):min(1000*file_id,data_len)] = torch.load("../../Data/"+filename)
		# print image_vectors
	return image_vectors

def get_user_vectors(filename="",embedding_dim=100,num_users=39387):
	"""
	input- the filename from where we load user vectors
	embedding_dim- Length of user vector
	num_users- number of users
	"""
	if os.path.isfile(filename):
		return torch.load(filename)
	else:
		return torch.nn.Embedding(num_users,embedding_dim)

def get_random_from_tuple_list(data, batch_size=0):
	"""
	data- [list of data]
	batch_size- number of random samples needed
	"""
	# random.seed(1)
	indexes = random.sample(range(0, len(data)-1), batch_size)
	return [data[k] for k in indexes]

def save_user_vectors(user_vectors,filename):
	torch.save(user_vectors,filename)


def loadAE(filename=None):
	# Load AutoEncoder
	if filename is not None and os.path.isfile(filename):
		AE = torch.load(filename)
	else:
		AE = md.AutoEncoder()

	if HAVE_CUDA == True:
		AE = AE.cuda()

	return AE

def loadrec_net(filename=None):
	# Load AutoEncoder
	if filename is not None and os.path.isfile(filename):
		rec_net = torch.load(filename)
	else:
		rec_net = rmd.FeedForward()

	if HAVE_CUDA == True:
		rec_net = rec_net.cuda()

	return rec_net

def loadOptimizer(MODEL, filename=None):
	optimizer = optim.Adam(MODEL.parameters(), lr=0.01)
	if filename is not None and os.path.isfile(filename):
		optimizer.load_state_dict(torch.load(filename))
	# else:
	# 	optimizer = optim.Adam(MODEL.parameters(), lr=0.001)

	return optimizer

def loadJointTrainingNet(filename=None):
	# Load AutoEncoder
	if filename is not None and os.path.isfile(filename):
		rec_net = torch.load(filename)
	else:
		rec_net = rmd.JointNet()

	if HAVE_CUDA == True:
		rec_net = rec_net.cuda()

	return rec_net

def loadDeepJointTrainingNet(filename=None):
	# Load AutoEncoder
	if filename is not None and os.path.isfile(filename):
		rec_net = torch.load(filename)
	else:
		rec_net = rmd.DeepJointNet()

	if HAVE_CUDA == True:
		rec_net = rec_net.cuda()

	return rec_net

def loadDeepRELUJointTrainingNet(filename=None):
	# Load AutoEncoder
	if filename is not None and os.path.isfile(filename):
		rec_net = torch.load(filename)
	else:
		rec_net = rmd.DeepRELUJointNet()

	if HAVE_CUDA == True:
		rec_net = rec_net.cuda()

	return rec_net

def loadDeepItemJointTrainingNet(filename=None):
	# Load AutoEncoder
	if filename is not None and os.path.isfile(filename):
		rec_net = torch.load(filename)
	else:
		rec_net = rmd.DeepItemJointNet()

	if HAVE_CUDA == True:
		rec_net = rec_net.cuda()

	return rec_net

def loadDeepRELUItemJointTrainingNet(filename=None):
	# Load AutoEncoder
	if filename is not None and os.path.isfile(filename):
		rec_net = torch.load(filename)
	else:
		rec_net = rmd.DeepRELUItemJointNet()

	if HAVE_CUDA == True:
		rec_net = rec_net.cuda()

	return rec_net

def get_ids_to_index_dict_from_file(filename):
	"""
	filename: user or item's id->index file 
	For eg: 
	BFCCCXXX003, 0
	BDCGTRXX003, 1
	DGF1CXXX003, 2
	returns a list of image or user ids in order of indexes
	"""
	datafile = open(filename,"r")
	data = {}
	for l in datafile:
		Id, Idx = l.split(',')
		data[Id] = Idx.rstrip()
	return data

def store_CNN_image_vecs_in_torch_file():
	"""
	Reads data from the binary file and returns the 
	"""
	image_id_to_idx = get_ids_to_index_dict_from_file("../../Data/item_to_index.txt")

	#Load all the image vectors into memory
	image_vectors = ag.Variable(torch.zeros(data_len, 4096))
	
	f = open("../../Data/CNN_image_features/raw_feat.b", 'rb')
	while True:
		image_id = f.read(10)
		if image_id == '':
			break
		#Initialize to torch variable/tensor
		feature = []
		for i in range(4096):
			feature.append(struct.unpack('f', f.read(4)))
		if image_id in image_id_to_idx.keys():
			image_vectors[image_id_to_idx[image_id]] = feature
	f.close()

	#Save the image vectors as torch variable
	torch.save(image_vectors,"../../Data/CNN_image_features/indexed_CNN_feat")

def get_cnn_image_vectors():
	return torch.load("../../Data/CNN_image_features/indexed_CNN_feat")

def run_random_baseline(num_items=23033, K=10):
	image_ids = get_ids_from_file("../../Data/item_to_index.txt")
	train_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")
	test_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")
	hits = 0
	num_users = len(test_dict.keys())

	for user_idx in test_dict.keys():
		pred_out = random.sample(range(0, num_items), K)
		
		for item in pred_out:
			if str(item) in train_dict[user_idx] or str(item) in test_dict[user_idx]:
				hits += 1

	print 'Average HR@', K, 'using random recommendation is', float(hits)/(K*num_users) 

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

	# AE = loadAE('../AE/Checkpoints/auto_encoder2')

	#One time run for CNN features to generate indexed feature vectors
	# store_CNN_image_vecs_in_torch_file()

	# Random baseline
	run_random_baseline(K=10)
