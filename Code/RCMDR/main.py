import random
import sys
import torch
sys.path.append('../Data Handler')
from utils import *
import rcmdr_model as rcmdr_model

if HAVE_CUDA:
	import torch.cuda as cuda



def get_data_for_rcmdr(ae_item_vecs, index_triples):
	
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
		item = ae_item_vecs[v].view(1,100)
		target = ag.Variable(torch.FloatTensor([val]), requires_grad=False)

		if(flag == 0):
			item_data = item
			user_data = user
			targets = target
			flag = 1
		else:
			item_data = torch.cat((item_data, item),0)
			user_data = torch.cat((user_data, user),0)
			targets = torch.cat((targets, target),0)

	return item_data, user_data, targets

def add_negative_samples(train_batch, data_dict, total_items, num_negative=0):

	random.seed(1)
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

def run_network(rec_net, AE, item_vecs, batch_size, mode, num_negative, num_epochs, data_dict=None, criterion=None, print_every =100,checkpoint_name = "Recommender_Network"):
	
	if mode is None:
		print 'No mode given'
		return

	elif mode == 'train':
		optimizer = loadOptimizer(MODEL=rec_net)
		if criterion is None:
			criterion = nn.MSELoss()

		train_tuples = []
		for user, items in data_dict.iteritems():
			for v in items:
				train_tuples.append((user, v))

		training_size = len(train_tuples)
		total_loss = 0.0
		print 'Total training tuples', training_size
		num_batches_per_epoch = training_size*(num_negative+1)/batch_size
		tot_iters = num_epochs*num_batches_per_epoch
		start_time = time.time()

		for iteration in range(tot_iters):
			train_batch = get_random_from_tuple_list(train_tuples, batch_size)
			# print 'Number of positive samples', len(train_batch)
			train_batch = add_negative_samples(train_batch, data_dict, item_vecs.size()[0], num_negative)
			# print 'Number of positive+negative samples', len(train_batch)
			item_data, user_data, target = get_data_for_rcmdr(item_vecs, train_batch)
			
			optimizer.zero_grad()
			
			pred_target = rec_net(item_data,user_data)
			loss = 0.0
			loss = criterion(pred_target, target)
			total_loss += loss.data[0]
			loss.backward()
			optimizer.step()
			
			# Print loss after ever batch of training
			if iteration % print_every == 0:
				print "============================================"
				print iteration, "of ", tot_iters
				time_remaining(start_time, tot_iters, iteration+1)
				print "Total loss === ",total_loss/print_every
				print np.squeeze(pred_target).data[0:6]
				# print "Mismatch = ", round(np.squeeze(pred_target).data.numpy())-target
				total_loss = 0
				torch.save(rec_net,os.getcwd()+"/Checkpoints/"+checkpoint_name)
			
		pass

	elif mode == 'test':
		test_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")
		rcmdr = md.FeedForward()
		pass

def run_recommender(batch_size=None, mode=None, num_epochs=None, num_negative=0, criterion=None, print_every = 10,checkpoint_name="Recommender_Network"):
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
		
		if os.path.isfile(os.getcwd()+"/Checkpoints/"+checkpoint_name):
			rec_net = torch.load(os.getcwd()+"/Checkpoints/"+checkpoint_name)
		else:
			rec_net = rcmdr_model.FeedForward()
			


		if mode == 'train':
			data_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")

		elif mode == 'test':
			data_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")

		
		run_network(rec_net, AE, ae_item_vecs, batch_size, mode, num_negative, num_epochs, data_dict=data_dict, criterion=criterion, print_every = print_every,checkpoint_name=checkpoint_name)


if __name__ == '__main__':

	run_recommender(batch_size=32, mode="train", num_epochs=10, num_negative=5, criterion=nn.MSELoss(),checkpoint_name="Recommender_Network_New")