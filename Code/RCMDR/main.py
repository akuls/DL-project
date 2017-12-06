import random
import sys
import torch
sys.path.append('../Data Handler')
from utils import *
import rcmdr_model as rcmdr_model

if HAVE_CUDA:
	import torch.cuda as cuda

def compute_PRF_HR(topN_list, ground_truth_dict, topK):

	test_dict = get_dict_from_index_mapping("../../Data/user_item_test.txt")
	n = len(topK_list)
	prev_user = -1
	tp = 0
	fp = 0
	start = 0
	user_precision = []
	user_recall = []
	user_AP = []
	temp_ap = 0.0
	MAP = 0.0
	num_users = 0
	total_hits = 0

	for i in range(n):
		u, val, v = topK_list[i]

		if u != prev_user:
			if(prev_user != -1):
				print 'HR@', topK, 'for user', prev_user, '= ', float(tp)/float(topK)
				user_AP.append((prev_user, temp_ap))
				user_precision.append((prev_user, float(tp)/(float(tp)+float(fp))))
				user_recall.append((prev_user, float(fp)/float(len(test_dict[prev_user]))))
				MAP += temp_ap
				num_users += 1
				total_hits += tp

			prev_user = u
			tp = fp = 0
			start = i
			temp_ap = 0.0

		if(ground_truth_dict[(u, v)] == 1):
			tp += 1
		else:
			fp += 1

		#Running ratio of tp/total_seen_till_now for each user
		temp_ap += float(tp)/float(i-start+1)

	#Last user outside the loop	
	print 'HR@', topK, 'for user', prev_user, '= ', float(tp)/float(topK)
	user_AP.append((prev_user, temp_ap))
	user_precision.append((prev_user, float(tp)/(float(tp)+float(fp))))
	user_recall.append((prev_user, float(fp)/float(len(test_dict[prev_user]))))
	MAP += temp_ap
	num_users += 1

	MAP /= float(num_users)
	print 'MAP is', MAP

	print 'Average HR@', topK, 'per user is', total_hits/float(num_users)


def compute_metrics(pred, triples, topK=10):

	final_triple = [] #This will have (user_idx, pred_score, item_idx)
	ground_truth_dict = {} #This will map (user_idx, item_idx) to (ground_truth of 1 or 0)

	n = len(triples)
	for i in range(n):
		u, v, ground_truth_dict[(u, v)] = triples[i]
		final_triple.append((u, pred.data[i], v))
		
	#Sort them while keeping user's data together
	sorted_triple = sorted(final_triple, reverse=True)

	#Iterate this and for every new user, pick the first topK items
	topK_list = []
	prev_user = -1
	i = 0
	while(i<n):
		u, _, _ = sorted_triple[i]
		if(u != prev_user):
			print 'New user at position', i 
			prev_user = u
			topK_list.extend(sorted_triple[i:i+topK])
			i += topK

		else:
			i += 1

	#At this point, topN_list has topK for all users sorted in descending order
	#Use this and the ground truth dictionary to compute metrics for each/all users
	compute_PRF_HR(topK_list, ground_truth_dict, topK)

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

def add_negative_samples(tuple_list, data_dict, total_items, num_negative=0):

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

def run_network(rec_net, AE, item_vecs, batch_size, mode, num_negative, num_epochs, data_dict=None, criterion=None, print_every =100,checkpoint_name = "Recommender_Network"):
	
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

		if mode == 'train':
			optimizer = loadOptimizer(MODEL=rec_net)

			training_size = len(data_tuples)
			total_loss = 0.0
			print 'Total training tuples', training_size
			num_batches_per_epoch = training_size*(num_negative+1)/batch_size
			tot_iters = num_epochs*num_batches_per_epoch
			start_time = time.time()

			for iteration in range(tot_iters):
				train_batch = get_random_from_tuple_list(data_tuples, batch_size)
				# print 'Number of positive samples', len(train_batch)
				train_batch = add_negative_samples(train_batch, data_dict, item_vecs.size()[0], num_negative)
				# print 'Number of positive+negative samples', len(train_batch)
				item_data, user_data, target = get_data_for_rcmdr(item_vecs, train_batch)
				
				optimizer.zero_grad()
				
				pred_target = rec_net(item_data, user_data)
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
			##############################END OF TRAIN###################################

		elif mode == 'test':
			print 'In test block'
			#Read train dict for negative sampling
			train_dict = get_dict_from_index_mapping("../../Data/user_item_train.txt")

			#Merge with test dict for negative sampling
			merged = {}
			for user_idx, item_idxs in train_dict.iteritems():
				merged[user_idx] = train_dict[user_idx] + data_dict[user_idx]

			print 'Adding negative samples to', len(data_tuples)
			test_batch = add_negative_samples(data_tuples, merged, item_vecs.size()[0], num_negative)
			print 'After adding', len(test_batch)
			print test_batch[:20]
			print 'Getting actual item vecs into place'
			item_data, user_data, target = get_data_for_rcmdr(item_vecs, test_batch)

			print 'Correct till here with total items/user', item_data.size(), user_data.size()
			#Run forward pass to get the results
			# pred_target = rec_net(item_data, user_data)
			# loss = criterion(pred_target, target)

			# print 'Test time loss is', loss.data[0]

			# compute_metrics(pred_target, test_batch, topK=num_negative/10)
			pass

def run_recommender(batch_size=None, mode=None, num_epochs=None, num_negative=0, criterion=None, print_every = 100,checkpoint_name="Recommender_Network"):
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

	run_recommender(batch_size=32, mode="test", num_epochs=10, num_negative=5, criterion=nn.MSELoss())