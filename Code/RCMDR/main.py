import random
import sys
import torch
sys.path.append('../Data Handler')
from utils import *
import rcmdr_model as rcmdr_model

if HAVE_CUDA:
	import torch.cuda as cuda

def combine_user_data(user, item):
	# print 'User size, item size is', user.size(), item.size()
	item = item.view(1, item.size()[0])
	# item.requires_grad = False
	# print 'User size, item size is', user.size(), item.size()
	res = torch.cat((user, item),1)
	# print 'Res size is', res.size()
	return res

def get_data_for_rcmdr(ae_item_vecs, user_vecs, index_triples):
	
	data = None
	targets = None
	flag = 0

	for triple in index_triples:

		u = triple[0]
		v = triple[1]
		val = triple[2]

		#Get real valued vectors for these
		u = ag.Variable(torch.LongTensor([u]))
		user = user_vecs(u)
		item = ae_item_vecs[v]

		#Combine it with the data
		user_item = combine_user_data(user, item)
		target = ag.Variable(torch.FloatTensor([val]), requires_grad=False)

		if(flag == 0):
			data = user_item
			targets = target
			flag = 1
		else:
			data = torch.cat((data, user_item),0)
			targets = torch.cat((targets, target),0)

	return data, targets

def add_negative_samples(train_batch, data_dict, total_items, num_negative=0):

	all_triples =[]
	for pair in train_batch:
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

def run_network(AE, item_vecs, user_vecs, batch_size, mode, num_negative, num_epochs, data_dict=None, criterion=None):
	
	if mode is None:
		print 'No mode given'
		return

	elif mode == 'train':
		rec_net = rcmdr_model.FeedForward()
		optimizer = loadOptimizer(MODEL=rec_net)
		if criterion is None:
			criterion = nn.MSELoss()

		train_tuples = []
		for user, items in data_dict.iteritems():
			for v in items:
				train_tuples.append((user, v))

		training_size = len(train_tuples)

		print 'Total training tuples', training_size
		num_batches_per_epoch = training_size*(num_negative+1)/batch_size

		for epoch in range(num_epochs*num_batches_per_epoch):
			train_batch = get_random_from_tuple_list(train_tuples, batch_size)
			# print 'Number of positive samples', len(train_batch)
			train_batch = add_negative_samples(train_batch, data_dict, item_vecs.size()[0], num_negative)
			print 'Number of positive+negative samples', len(train_batch)
			data, target = get_data_for_rcmdr(item_vecs, user_vecs, train_batch)
			# data_optimizer = optim.SGD([data],lr=0.001)
			# temp_data = data.clone()	
			# print 'All correct'
			# print 'Data size', data.size()
			# print 'Target size', target.size()
			# for overfit in range(10):
				# print overfit
				# temp_data = data.clone()
			optimizer.zero_grad()
			# data_optimizer.zero_grad()
			pred_target = rec_net(data)
			print data.requires_grad
			print data.grad
			loss = 0.0
			loss = criterion(pred_target, target)
			loss.backward()
			optimizer.step()
			# data_optimizer.step()
			print data

				# Print loss after ever batch of training
			print epoch, 'batches of training done. Loss = ', loss.data[0]
			if epoch == 0:
				break

		pass

	elif mode == 'test':
		test_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")
		rcmdr = md.FeedForward()
		pass

def run_recommender(batch_size=None, mode=None, num_epochs=None, num_negative=0, criterion=None):
	if mode is None:
		print 'No mode given'
		return

	else:
		#Call util to get the vectors and optimizer
		AE = loadAE('../AE/Checkpoints/auto_encoder2')
		
		print 'Loading item item_vecs'
		pt = time.time()
		ae_item_vecs = get_image_vectors(AE,filename="../../Data/image_vectors")
		et = time.time()
		print 'Takes', et-pt, 'seconds'
		
		user_vecs = get_user_vectors()

		if mode == 'train':
			data_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")

		elif mode == 'test':
			data_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")

		
		run_network(AE, ae_item_vecs, user_vecs, batch_size, mode, num_negative, num_epochs, data_dict=data_dict, criterion=criterion)


if __name__ == '__main__':

	run_recommender(batch_size=32, mode="train", num_epochs=10, num_negative=5, criterion=nn.MSELoss())