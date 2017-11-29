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

def begin_training(num_epochs = 1, print_every = 10):

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
	num_buckets = len(all_item_buckets)
	loss_arr = []

	#Run for num_epochs number of epochs
	while(iteration<num_epochs*num_buckets):
		iteration+=1

		# Selecting a data bucket
		bucket_index = random.randint(0, num_buckets-1)
		# bucket_index = 0
		# print 'Iteration', iteration, 'picked bucket->', bucket_index
		# b_no = 1
		# Optimizer
		
		# Train autoencoder
		loss = md.trainAE(all_item_buckets[bucket_index], AE, optimizer)
		total_loss += loss
		loss_arr.append(loss)
		# md.trainAE(itemsbin[b_no],AE,optimizer)

		# Checkpointing
		torch.save(AE,os.getcwd()+"/Checkpoints/auto_encoder")
		# break
		# torch.save(optimizer,os.getcwd()+"/Checkpoints/optm")

		# Train the current batch
		# md.trainmodel1(data[b_no],items,user_vts,users_to_ix ,img_model,optimizer)
		if iteration % print_every == 0:
			print "Time elapsed ======================== ",(time.time()-start_time)
			print "Total loss ========================== ",total_loss/print_every
			total_loss = 0
		# break

	#Record finish time
	end_time = time.time()
	print 'Total training time for', num_epochs, 'epochs = ', (end_time-start_time), 's'
	return loss_arr

def transform_images(x=None):
	tt = transforms.ToTensor()
	if x == None:
		x = get_item_id_buckets()[0]
	if os.path.isfile(os.getcwd()+"/Checkpoints/auto_encoder"):
		AE = torch.load(os.getcwd()+"/Checkpoints/auto_encoder")
	else:
		AE = md.AutoEncoder()
	for item_id in x:
		print item_id
		item_image = Image.open("../../Data/Resize_images_50/"+item_id.rstrip()+".jpg")			
		item_image = ag.Variable(tt(item_image)).view(1,-1,SIDELENGTH,SIDELENGTH)
		break
	y = AE(item_image)
	print item_image
	print y
	return y


def main():
	print 'Beginning to train AE'
	loss_val = begin_training(num_epochs = 10)
	# np.save("loss.npy",loss_val)
	# print np.load("loss.npy")
	# out = transform_images()
	# out = ((out.view(3,50,50)).data.numpy()*255).astype(np.uint8)
	# out = np.swapaxes(out,0,2)
	# out = np.swapaxes(out,0,1)
	# print out.shape
	# im = Image.fromarray(out,"RGB")
	# im.save("your_file.png")

	print 'Training AE completed'

if __name__ == '__main__':
	main()