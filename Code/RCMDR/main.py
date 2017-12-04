import random
import sys
sys.path.append('../Data Handler')
from utils import *
import rcmdr_model as rcmdr_md

if HAVE_CUDA:
	import torch.cuda as cuda

def combine_user_data(user, item):
	res = torch.cat((user, item),1)
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
		# print u
		u = ag.Variable(torch.LongTensor([u]))
		user = user_vecs(u)
		print ae_item_vecs.size()
		item = ae_item_vecs[v]

		#Combine it with the data
		user_item = combine_user_data(user, item)
		target = ag.Variable(torch.FloatTensor(val))
		
		if(flag == 0):
			data = user_item
			targets = target
			flag = 1
		else:
			data = torch.cat((data,user_item),0)
			targets = torch.cat((targets,target),0)

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

def run_network(AE, item_vecs, user_vecs, batch_size, mode, num_negative, num_epochs, data_dict=None):
	
	if mode is None:
		print 'No mode given'
		return

	elif mode == 'train':
		rcmdr = rcmdr_md.FeedForward()
		optimizer = loadOptimizer(MODEL=rcmdr)

		train_tuples = []
		for user, items in data_dict.iteritems():
			for v in items:
				train_tuples.append((user, v))

		training_size = len(train_tuples)

		print 'Total training tuples', training_size
		num_batches_per_epoch = training_size*(num_negative+1)/batch_size

		for epoch in range(num_epochs*num_batches_per_epoch):
			train_batch = get_random_from_tuple_list(train_tuples, batch_size)
			train_batch = add_negative_samples(train_batch, data_dict, item_vecs.size()[0], num_negative)
			data, target = get_data_for_rcmdr(item_vecs, user_vecs, train_batch)
				
			print 'All correct'
			break
	        # optimizer.zero_grad()

	        # rcmdr_out = rcmdr(data)
	        # loss = criterion(rcmdr_out, target)
	        # loss.backward()
	        # optimizer.step()

	        #Print loss after ever batch of training

		pass

	elif mode == 'test':
		test_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")
		rcmdr = md.FeedForward()
		pass

def run_recommender(batch_size=None, mode=None, num_epochs=None, num_negative=0):
	if mode is None:
		print 'No mode given'
		return

	else:
		#Call util to get the vectors and optimizer
		AE = loadAE('../AE/Checkpoints/auto_encoder2')
		print 'loading'
		pt = time.time()
		ae_item_vecs = get_image_vectors(AE)
		et = time.time()

		print 'Takes', et-pt
		user_vecs = get_user_vectors()

		if mode == 'train':
			data_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")

		elif mode == 'test':
			data_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")

		
		run_network(AE, ae_item_vecs, user_vecs, batch_size, mode, num_negative, num_epochs, data_dict=data_dict)


if __name__ == '__main__':

	run_recommender(batch_size=32, mode="train", num_epochs=10, num_negative=5)