import PIL
from PIL import Image
import os, sys
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import numpy as np
import model as md
import random
from collections import defaultdict
from torchvision import transforms
import time
sys.path.append('../../Config')
from constants import HAVE_CUDA, BUCKET_SIZE, EMBEDDING_DIM, SIDELENGTH

if HAVE_CUDA:
	import torch.cuda as cuda


def get_item_id_buckets():
	itemsfile = open("../../Data/item_ids.txt","r")
	item_bucket = []
	all_item_buckets = []
	num_items=0

	for line in itemsfile:
		num_items+=1
		itm = line.split("\n")[0]
		item_bucket.append(itm)

		if num_items%BUCKET_SIZE == 0:
			all_item_buckets.append(item_bucket)
			item_bucket = []

	if len(item_bucket)>0:
		all_item_buckets.append(item_bucket)

	return all_item_buckets

def get_user_item_pair_buckets():
	datafile = open("../../Data/pairs.txt","r")	#Change this to train/test as the data is divided
	data = []
	user_id=0
	pdata = defaultdict(list)
	users_to_id = {}
	for l in datafile:
		user, items, t = l.split(" ||| ")
		pdata[user] = items.split(",")

		users_to_id[str(user)] = user_id
		user_id+=1

		if user_id%BUCKET_SIZE == 0:
			data.append(pdata)
			pdata = defaultdict(list)
	if len(pdata)>0:
		data.append(pdata)
	# print len(data)
	# for i in range(len(data)):
	# 	print len(data[i])
	# 	break

	return data, users_to_id

def load_classes(users_to_id):
	
	# torch.nn.Module.dump_patches = True

	# Load the model if available
	if os.path.isfile(os.getcwd()+"/Checkpoints/img_model"):
		img_model = torch.load(os.getcwd()+"/Checkpoints/img_model")
	else:
		img_model = md.ExtractImageVectors(EMBEDDING_DIM)
	
	# Load user vectors
	if os.path.isfile(os.getcwd()+"/Checkpoints/user_vts"):
		user_vts = torch.load(os.getcwd()+"/Checkpoints/user_vts")
	else:
		user_vts = nn.Embedding(len(users_to_id),EMBEDDING_DIM)#,max_norm = 1.0)

	# Load AutoEncoder
	if os.path.isfile(os.getcwd()+"/Checkpoints/auto_encoder"):
		AE = torch.load(os.getcwd()+"/Checkpoints/auto_encoder")
	else:
		AE = md.AutoEncoder()

	if os.path.isfile(os.getcwd()+"/Checkpoints/optm"):
		optimizer = torch.load(os.getcwd()+"/Checkpoints/optm")
	else:
		optimizer = optim.Adam(AE.parameters(), lr=0.001)

	return img_model, user_vts, AE, optimizer

def begin_training(num_epochs = 10, print_every = 100):

	#Get item id buckets and user item pair buckets
	all_item_buckets = get_item_id_buckets()
	data, users_to_id = get_user_item_pair_buckets()

	print 'All item buckets = ', len(all_item_buckets)
	print 'All user-item buckets = ', len(data)
	print 'Total number of users = ', len(users_to_id)

	#Load all necessary classes
	img_model, user_vts, AE, optimizer = load_classes(users_to_id)

	#Begin training
	start_time = time.time()
	iteration = 0
	total_loss = 0

	#Run for num_epochs number of epochs
	while(iteration<num_epochs):
		iteration+=1

		# Selecting a data bucket
		bucket_index = random.randint(0, len(all_item_buckets)-1)
		print 'Iteration', iteration, 'picked bucket->', bucket_index
		# b_no = 1
		# Optimizer
		
		# Train autoencoder
		total_loss += md.trainAE(all_item_buckets[bucket_index], AE, optimizer)
		break
		# md.trainAE(itemsbin[b_no],AE,optimizer)

		# Checkpointing
		torch.save(AE,os.getcwd()+"/Checkpoints/auto_encoder")
		# torch.save(optimizer,os.getcwd()+"/Checkpoints/optm")

		# Train the current batch
		# md.trainmodel1(data[b_no],items,user_vts,users_to_ix ,img_model,optimizer)
		if iteration % print_every == 0:
			print "Time elapsed ======================== ",(time.time()-start_time)/60
			print "Total loss ========================== ",total_loss/100
			total_loss = 0
		# break

	#Record finish time
	end_time = time.time()
	print 'Total training time for', num_epochs, 'epochs = ', (end_time-start_time)/60, 's'

if __name__ == '__main__':
	print 'Beginning to train AE'
	begin_training(num_epochs = 10)
	print 'Training AE completed'