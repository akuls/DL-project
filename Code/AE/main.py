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
sys.path.append('../Data Handler')
from constants import HAVE_CUDA, BUCKET_SIZE, EMBEDDING_DIM, SIDELENGTH
from utils import *

if HAVE_CUDA:
	import torch.cuda as cuda

def run_AE(AE, optimizer, data, image_variables, batch_size, num_epochs, criterion, print_every =100,checkpoint_name = "auto_encoder"):
	
	training_size = len(data)
	total_loss = 0.0
	print 'Total training data', training_size
	num_batches_per_epoch = training_size/batch_size
	tot_iters = num_epochs*num_batches_per_epoch
	start_time = time.time()

	for iteration in range(tot_iters):
		# Obtain batch data
		image_idxs = torch.LongTensor(random.sample(range(0, training_size-1), batch_size))
		# image_idxs = torch.LongTensor([0,0,0,0])
		if HAVE_CUDA == True:
			image_idxs = image_idxs.cuda()
		batch_data = image_variables[image_idxs]
		batch_data.view(-1,1,SIDELENGTH,SIDELENGTH)

		optimizer.zero_grad()
		#Training a full batch
		pred_target = AE(batch_data)
		loss = 0.0
		loss = criterion(pred_target, batch_data)
		total_loss += loss.data[0]
		loss.backward()
		optimizer.step()
		
		# Print loss after ever batch of training
		if (iteration+1) % print_every == 0 or (iteration+1) == tot_iters:
			print "============================================"
			print type(pred_target.data)
			print iteration+1, "of ", tot_iters
			time_remaining(start_time, tot_iters, iteration+1)
			print "Total loss === ",total_loss/print_every
			total_loss = 0.0
			torch.save(AE,os.getcwd()+"/Checkpoints/"+checkpoint_name)
			torch.save(optimizer.state_dict(),os.getcwd()+"/Checkpoints/optim_"+checkpoint_name)


def train_AE(batch_size=32, num_epochs=10, criterion=nn.MSELoss(), print_every = 10,checkpoint_name="auto_encoder"):
	
	AE = loadAE_exp('Checkpoints/'+ checkpoint_name)
	optimizer = loadOptimizer(AE,filename='Checkpoints/optim_'+ checkpoint_name)
	data = get_ids_from_file("../../Data/item_to_index.txt")
	image_variables = image_ids_to_variable(data)
	if HAVE_CUDA == True:
		criterion = criterion.cuda()
		image_variables = image_variables.cuda()

	run_AE(AE, optimizer, data, image_variables, batch_size, num_epochs, criterion, print_every =print_every,checkpoint_name = checkpoint_name)


def transform_images(x=None):
	tt = transforms.ToTensor()
	if x == None:
		x = get_item_id_buckets()[0]
	if os.path.isfile(os.getcwd()+"/Checkpoints/auto_encoder2"):
		AE = torch.load(os.getcwd()+"/Checkpoints/auto_encoder2")
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

def get_Image_Feature_maps():
	AE = torch.load(os.getcwd()+"/Checkpoints/auto_encoder")
	itemsfile = open("../../Data/item_ids.txt","r")
	tt = transforms.ToTensor()
	num_items=0
	tot_out = None

	for line in itemsfile:
		num_items+=1
		item_id = line.split("\n")[0]
		item_image = Image.open("../../Data/Resize_images_50/"+item_id.rstrip()+".jpg")			
		item_image = ag.Variable(tt(item_image)).view(1,-1,SIDELENGTH,SIDELENGTH)
		out = AE.get_vector(item_image)
		out = ((np.squeeze(out)).data.numpy()*255)
		if tot_out ==None:
			tot_out = out
			# print "shit"
		else:
			tot_out = np.concatenate((tot_out,out),axis=0)
		# print tot_out.shape
		if num_items == 1000:
			print num_items
			break
	return tot_out	

def main():
	print 'Beginning to train AE'
	# train_AE(num_epochs = 100, print_every=100)
	train_AE(num_epochs = 100, print_every=100, checkpoint_name="auto_encoder_exp")
	# loss_val = begin_training(num_epochs = 10,print_every=10)
	# np.save("loss.npy",loss_val)
	# print np.load("loss.npy")
	# out = transform_images()
	# out = ((out.view(3,50,50)).data.numpy()*255).astype(np.uint8)
	# out = np.swapaxes(out,0,2)
	# out = np.swapaxes(out,0,1)
	# print out.shape
	# im = Image.fromarray(out,"RGB")
	# im.save("your_file.png")
	# temp = get_Image_Feature_maps()
	# np.save("temp.npy",temp)
	# AE = md.AutoEncoder()
	# AE = loadAE(os.getcwd()+"/Checkpoints/auto_encoder2")
	# ivc = get_image_vectors(AE)
	# torch.save(ivc,"../../Data/image_vectors")
	# ivc1 = torch.load("../../Data/image_vectors")
	# print ivc1
	# image_ids = get_item_id_buckets()[0]
	# i_vt = get_Image_Vectors(AE,image_ids)
	# print i_vt
	print 'Training AE completed'

if __name__ == '__main__':
	main()
