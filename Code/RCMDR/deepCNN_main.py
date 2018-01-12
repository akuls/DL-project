import random
import sys
import torch
sys.path.append('../Data Handler')
from utils import *
import rcmdr_model as rcmdr_model
import torch.nn.functional as F
if HAVE_CUDA:
	import torch.cuda as cuda

def compute_PRF_HR(topK_list, ground_truth_dict, test_dict, topK):
	"""
	topN_list- list containing the topK entries for each user in the list
	ground_truth_dict- dictionary that contains (user_idx, item_idx) -> 1 or 0
	topK - integer indicating how many topK entries we are considering per user

	Computes precision, recall, MAP, HR@10
	"""
	
	n = len(topK_list)
	
	num_users = n/topK
	total_hits = 0

	for i in range(n):
		u, val, v = topK_list[i]
		if(ground_truth_dict[(u, v)] == 1):
			total_hits += 1
	return total_hits/(topK*float(num_users)),total_hits


def compute_metrics(pred, triples, test_dict, topK=10):
	"""
	pred- the predicted output from the network
	triples- triple list containing (user_idx, item_idx, ground_truth)
	"""
	final_triple = [] #This will have (user_idx, pred_score, item_idx)
	ground_truth_dict = {} #This will map (user_idx, item_idx) to (ground_truth of 1 or 0)

	n = len(triples)
	for i in range(n):
		u, v, ground_truth_dict[(u, v)] = triples[i]
		final_triple.append((u, pred.data[i][1], v))
		
	#Sort them while keeping user's data together
	sorted_triple = sorted(final_triple, reverse=True)

	#Iterate this and for every new user, pick the first topK items
	topK_list = []
	prev_user = -1
	i = 0
	while(i<n):
		u, _, _ = sorted_triple[i]
		if(u != prev_user):
			# print 'New user at position', i 
			prev_user = u
			topK_list.extend(sorted_triple[i:i+topK])
			i += topK

		else:
			i += 1
	# print topK_list[0:topK]
	#At this point, topN_list has topK for all users sorted in descending order
	#Use this and the ground truth dictionary to compute metrics for each/all users
	# return compute_PRF_HR(topK_list, ground_truth_dict, test_dict, topK)
	aa,bb = compute_PRF_HR(topK_list, ground_truth_dict, test_dict, topK)
	return aa, bb, topK_list#remove this later

def get_data_for_rcmdr(cnn_item_vecs, index_triples,item_bool=False):
	"""
	cnn_item_vecs- item vectors from the autoencoder
	index_triples- triple list containing (user_idx, item_idx, ground_truth)

	This function prepares data for the rcmdr_net and return user_data, item_data and targets
	"""
	item_data = None
	user_data = None
	targets = None
	flag = 0

	for triple in index_triples:

		u = triple[0]
		v = triple[1]
		val = triple[2]

		#Get real valued vectors for these
		user = ag.Variable(torch.LongTensor([u])).view(1,1)
		itemid = ag.Variable(torch.LongTensor([v])).view(1,1)
		item = cnn_item_vecs[v].view(1, 3, 50, 50)
		# print int(val)
		target = ag.Variable(torch.LongTensor([int(val)]), requires_grad=False) # Change to Float for MSE

		if(flag == 0):
			item_data = item
			user_data = user
			itemid_data = itemid
			targets = target
			flag = 1
		else:
			item_data = torch.cat((item_data, item),0)
			itemid_data = torch.cat((itemid_data, itemid),0)
			user_data = torch.cat((user_data, user),0)
			targets = torch.cat((targets, target),0)

	if item_bool:
		return item_data, user_data, targets, itemid_data
	return item_data, user_data, targets

def add_negative_samples_train(tuple_list, data_dict, total_items, num_negative=0):
	"""
	tuple_list- The list of (user_idx, item_idx, 1)
	data_dict- The list containing (user_idx) ->[item_idxs bought]. 
	We need this to add items not in this list as negative samples for each user.
	total_items- Total number of items
	num_negative- number of negative samples per positive sample
	"""
	# random.seed(1)
	all_triples =[]
	for pair in tuple_list:
		u = pair[0]
		v = pair[1]

		all_triples.append((int(u), int(v), 1.0))

		if(num_negative>0):
			existing_item_idxs = [int(idx) for idx in data_dict[u]]
			neg_indexes = random.sample(range(0, total_items-1), num_negative*4)
			done = 0
			#Keep adding until done
			while done != num_negative:
				x = neg_indexes.pop(0)
				if(x not in existing_item_idxs):
					all_triples.append((int(u), int(x), 0.0))
					done += 1
		
	return all_triples

def add_negative_samples_test(tuple_list, data_dict, total_items, test_dict, num_negative=0):
	"""
	tuple_list- The list of (user_idx, item_idx, 1)
	data_dict- dict containing (user_idx) ->[item_idxs bought]. 
	We need this to add items not in this list as negative samples for each user.
	total_items- Total number of items
	test_dict- dict of (user_idx) -> [item_idxs for test]
	num_negative- number of negative samples per positive sample
	"""
	# random.seed(1)
	user_done = {}

	all_triples =[]
	for pair in tuple_list:
		
		u = pair[0]
		v = pair[1]

		if u not in user_done:
			# if int(u)%1000 == 0:
				# print 'At user idx', u
			user_done[u] = True
			all_triples.append((int(u), int(v), 1.0))

			if(num_negative>0):
				required = num_negative - len(test_dict[u])
				existing_item_idxs = [int(idx) for idx in data_dict[u]]
				neg_indexes = random.sample(range(0, total_items-1), required*4)
				done = 0
				#Keep adding until done
				while done != required:
					x = neg_indexes.pop(0)
					if(x not in existing_item_idxs):
						all_triples.append((int(u), int(x), 0.0))
						done += 1
		else:
			all_triples.append((int(u), int(v), 1.0))
	return all_triples

def run_network(rec_net, optimizer, item_vecs, batch_size, mode, num_negative, num_epochs, data_dict=None, criterion=None, print_every =100,checkpoint_name = "Recommender_Network"):
	"""
	rec_net- recommender net
	item_vecs - Tensor of shape (N, D)
	batch_size- input batch size for training
	mode- train or test
	num_negative- number of negative samples for each postive sample
	num_epochs- total epochs to train
	data_dict-  (user_idx)- [list of item_idx]
	Runs the specified network with all the parameters in the mode specified
	"""

	if mode is None:
		print 'No mode given'
		return

	else:

		#Create tuple list from dict
		data_tuples = []
		for user, items in data_dict.iteritems():
			for v in items:
				data_tuples.append((user, v))

		#Create loss criterion
		if criterion is None:
			criterion = nn.MSELoss()

		if HAVE_CUDA:
			criterion = criterion.cuda()

		if mode == 'train':

			training_size = len(data_tuples)
			total_loss = 0.0
			print 'Total training tuples', training_size
			num_batches_per_epoch = training_size/batch_size
			tot_iters = num_epochs*num_batches_per_epoch
			start_time = time.time()
			num_items = item_vecs.size()[0]

			for iteration in range(tot_iters):
				train_batch = get_random_from_tuple_list(data_tuples, batch_size)
				# print 'Number of positive samples', len(train_batch)
				train_batch = add_negative_samples_train(train_batch, data_dict, num_items, num_negative)
				# print 'Number of positive+negative samples', len(train_batch)
				item_data, user_data, target = get_data_for_rcmdr(item_vecs, train_batch)
				# item_data, user_data, target, itemid_data = get_data_for_rcmdr(item_vecs, train_batch,True)
				
				# print 'All data collected', item_data.size(), user_data.size(), target.size()
				if HAVE_CUDA:
					item_data = item_data.cuda()
					# itemid_data = itemid_data.cuda()
					user_data = user_data.cuda()
					target = target.cuda()

				optimizer.zero_grad()
				#Training a full batch
				# print item_data,itemid_data,user_data
				pred_target = rec_net(item_data, user_data)
				# pred_target = rec_net(item_data, itemid_data, user_data)
				loss = 0.0
				loss = criterion(pred_target, target)
				total_loss += loss.data[0]
				loss.backward()
				optimizer.step()

				# Print loss after ever batch of training
				if (iteration+1) % print_every == 0 or (iteration+1) == tot_iters:
					print "============================================"
					print iteration+1, "of ", tot_iters
					time_remaining(start_time, tot_iters, iteration+1)
					print "Total loss === ",total_loss/print_every
					print 'Pred is', torch.squeeze(pred_target).data[0:6]
					# print 'Truth is', np.squeeze(target).data[0:6]
					# print "Mismatch = ", round(np.squeeze(pred_target).data.numpy())-target
					total_loss = 0.0
					torch.save(rec_net,os.getcwd()+"/Checkpoints/"+checkpoint_name)
					torch.save(optimizer.state_dict(),os.getcwd()+"/Checkpoints/optim_"+checkpoint_name)
			##############################END OF TRAIN###################################

		elif mode == 'test':
			print 'In test block'
			#Read train dict for negative sampling
			train_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")
			rec_net.eval()
			#Merge with test dict for negative sampling
			merged = {}
			for user_idx, item_idxs in train_dict.iteritems():
				merged[user_idx] = train_dict[user_idx] + data_dict[user_idx]

			print 'Adding negative samples to', len(data_tuples)
			test_batch = add_negative_samples_test(data_tuples, merged, item_vecs.size()[0], data_dict, num_negative)
			print 'After adding', len(test_batch)
			print test_batch[:20]
			print 'Running in batches of size', batch_size
			
			# all_pred = ag.Variable(torch.FloatTensor(len(test_batch),1).zero_())
			#if HAVE_CUDA:
			#	all_pred = all_pred.cuda()
			start_time = time.time()
			batch_size *= num_negative
			HR = 0.0
			total_hits = 0
			for i in range(0, len(test_batch), batch_size):
				#item_data, user_data, target = get_data_for_rcmdr(item_vecs, test_batch[i:i+batch_size])
				item_data, user_data, target, itemid_data = get_data_for_rcmdr(item_vecs, test_batch[i:i+batch_size],True)
				if HAVE_CUDA:
					item_data = item_data.cuda()
					itemid_data = itemid_data.cuda()
					user_data = user_data.cuda()
					target = target.cuda()
				#print 'Correct till here with total items/user', item_data.size(), user_data.size()
				
				#Run forward pass to get the results
				#pred_target = rec_net(item_data, user_data)
				pred_target = rec_net(item_data, itemid_data, user_data)
				loss = criterion(pred_target, target)
				#print 'Test time loss is', loss.data[0]
				#print 'Pred target shape', pred_target.size()

				HR_add, th, _ = compute_metrics(pred_target, test_batch[i:i+batch_size], data_dict, topK=10)
				HR += HR_add
				total_hits += th
				#print HR_add 
				# all_pred[i:i+batch_size] = pred_target

				#if(i>100000):
				#	break
			HR /= (len(test_batch)/batch_size)
			print "Time taken to predict: ", time.time()-start_time
			print "Hit rate is", HR
			print "Total Hits", total_hits
			# compute_metrics(all_pred, test_batch, topK=10)
			pass

def run_recommender(batch_size=None, mode=None, num_epochs=None, num_negative=0, criterion=None, print_every = 10,checkpoint_name="Recommender_Network"):
	if mode is None:
		print 'No mode given'
		return

	else:
		#Call util to get the vectors and optimizer
		# rec_net = loadJointTrainingNet(os.getcwd()+"/Checkpoints/"+checkpoint_name)
		rec_net = loadDeepRELUItemJointTrainingNet(os.getcwd()+"/Checkpoints/"+checkpoint_name)
		criterion = nn.CrossEntropyLoss() # Only for RELU model
		optimizer = loadOptimizer(rec_net, os.getcwd()+"/Checkpoints/optim_"+checkpoint_name)

		print 'Loading raw image vectors'
		pt = time.time()
		if os.path.exists('../../Data/images_as_variables'):
			item_images = torch.load('../../Data/images_as_variables')
		else:
			item_images = image_ids_to_variable(get_ids_from_file('../../Data/item_to_index.txt'))
			torch.save(item_images, '../../Data/images_as_variables')
		et = time.time()
		print 'Takes', et-pt, 'seconds to load', item_images.size(), 'image variables'

		if mode == 'train':
			data_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")

		elif mode == 'test':
			data_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")

		if HAVE_CUDA:
			criterion = criterion.cuda()
			item_images = item_images.cuda()
		
		run_network(rec_net, optimizer, item_images, batch_size, mode, num_negative, num_epochs, data_dict=data_dict, criterion=criterion, print_every = print_every, checkpoint_name=checkpoint_name)

def run_saliency(image_ids = None, user_id = None, checkpoint_name="Recommender_Network"):
	#Call util to get the vectors and optimizer
	# rec_net = loadJointTrainingNet(os.getcwd()+"/Checkpoints/"+checkpoint_name)
	rec_net = loadDeepJointTrainingNet(os.getcwd()+"/Checkpoints/"+checkpoint_name)
	optimizer = loadOptimizer(rec_net, os.getcwd()+"/Checkpoints/optim_"+checkpoint_name)
	# rec_net.eval()

	print 'Loading raw image vectors'
	pt = time.time()
	item_data = image_ids_to_variable(image_ids)
	# item_data.requires_grad = True
	user_data = ag.Variable(torch.LongTensor([user_id for i in range(len(image_ids))]))
	target = ag.Variable(torch.FloatTensor([1 for i in range(len(image_ids))]))
	# criterion = F.MSELoss()
	if HAVE_CUDA:
		# criterion = criterion.cuda()
		item_data = item_data.cuda()
		user_data = user_data.cuda()
		target = target.cuda()

	# print rec_net(item_data, user_data)

	for x in rec_net.parameters():
		tx = x.clone()
		break
	optimizer.zero_grad()
	pred_target = rec_net(item_data, user_data)
	loss = torch.nn.functional.mse_loss(pred_target,target)
	# print loss.data[0]
	print loss.grad
	loss.backward()
	print loss.grad
	optimizer.step()
	print pred_target.requires_grad, target.requires_grad
	print pred_target.grad
	print user_data.requires_grad
	for x in rec_net.parameters():
		print x.requires_grad
		break
	# pred_target.backward(torch.Tensor([1 for i in range(len(image_ids))]).view(-1,1))
	# print item_data.grad.data

def run_random_test(batch_size=32, num_negative=50):
	test_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")
	train_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")

	print 'Got data'
	#Create tuple list from dict
	data_tuples = []
	for user, items in test_dict.iteritems():
		for v in items:
			data_tuples.append((user, v))

	print 'Merging'
	#Merge with test dict for negative sampling
	merged = {}
	for user_idx, item_idxs in train_dict.iteritems():
		merged[user_idx] = train_dict[user_idx] + test_dict[user_idx]

	print 'Now adding negative samples'
	test_batch = add_negative_samples_test(data_tuples, merged, 23033, test_dict, num_negative)
	print 'Added negative samples and now length is', len(test_batch)

	HR = 0.0
	start_time = time.time()
	batch_size *= num_negative
	for i in range(0, len(test_batch), batch_size):
		# Randomly predict 
		pred_target = np.random.uniform(0.0, 1.0, (batch_size, 1))
		x, _ = compute_metrics(pred_target, test_batch[i:i+batch_size], test_dict, topK=2)
		# print 'i', x
		HR += x

	HR /= (len(test_batch)/batch_size)
	print "Time taken to predict: ", time.time()-start_time
	print "Hit rate is", HR

if __name__ == '__main__':
	# run_recommender(batch_size=32, mode="train", num_epochs=200, num_negative=5, print_every=100, criterion=nn.MSELoss(),checkpoint_name="Deep_Joint_Net_Recommender_BN_200")
	run_recommender(batch_size=4, mode="test", num_negative=100, criterion=nn.MSELoss(),checkpoint_name="Deep_RELU_Item_Joint_Net_Recommender_Max")
	# run_random_test(batch_size=32, num_negative=100)
	# items = ["B0007UDXF2","B000GZQHKG"]
	# user = 13822
	# run_saliency(items,user,"Deep_Net")

