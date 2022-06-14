'''
Notes so far:

- Most of these split don't change much
- Some of these split dramatically worsen results
- Sometimes, random splits are the best (even better than state-of-the-art results).
- Why are random splits so good? And how (if) can they be improved?
'''


import argparse
import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', '-p', type=str, default='data/AWA1', help='path to the dataset folder')
	parser.add_argument('--split', '-s', type=str, default='rnd', help='type of split to use for the new dataset generation')	# 'rnd', 'gcs', 'ccs', 'mas', 'mcs', 'pca'
	parser.add_argument('--inverse', '-i', action='store_true', help='use inverse split')
	parser.add_argument('--pca_components', '-c', type=int, default=10, help='number of PCA components to use')
	parser.add_argument('--save', '-S', action='store_true', help='save the new dataset')
	return parser.parse_args()


''' Random Split (RND)
Control split
'''
def random_split(features, labels, attributes, inverse):
	old_seen = torch.from_numpy(np.unique(seen_labels))
	old_unseen = torch.from_numpy(np.unique(unseen_labels))
	old_classes = np.concatenate((old_seen, old_unseen))
	np.random.shuffle(old_classes)
	new_seen = old_classes[:int(n_seen_classes)]
	new_unseen = old_classes[int(n_seen_classes):]
	return new_seen, new_unseen, attributes


''' Greedy Class Split (GCS)
Tries to avoid the "horse with stripes without stripes images" scenario by keeping as much semantic information as possible among the seen classes.
In the binary definition of the semantic space, the value 1 indicates the presence of an attribute in an image, while the value 0 indicates its absence.
This means that ones are more useful than zeros, so we maximize the former in the seen classes split.
In other words, for each class, we simply sum the values of its signature vector and we sort the classes by these sums in descending order.
Consequently, we select the first Ns classes as seen classes, and the other Nu as unseen classes.
'''
def greedy_class_split(features, labels, attributes, inverse):
	# for each class, sum the values of its signature vector
	sums = np.sum(attributes, axis=1)
	# sorted_sums = np.sort(sums)
	sorted_sums = np.argsort(sums)
	new_seen = sorted_sums[:n_seen_classes] if inverse else sorted_sums[n_unseen_classes:]
	new_unseen = sorted_sums[n_seen_classes:] if inverse else sorted_sums[:n_unseen_classes]
	return new_seen, new_unseen, attributes


''' Clustered Class Split (CCS)
Tries to maximize the Class Semantic Distance between seen classes and unseen classes.
We define the Class Semantic Distance matrix where each element is the euclidean distance between class two class signatures (attribute vectors).
Seen and unseen classes are defined by sorting the classes by the sum of their row (or column) values in descending order.
The first Ns classes are those with the lowest distances overall, meaning that they form a cluster in the semantic space. Those classes will be the seen classes.
The other Nu are far from this cluster in the semantic space, so they will form another cluster
(although it is not a proper cluster since those classes are probably far away from each other as well), and they will be the unseen classes.
'''
def clustered_class_split(features, labels, attributes, inverse):
	distances = []
	for a1 in attributes:
		att_distances = []
		for a2 in attributes:
			d = np.linalg.norm(a1 - a2)
			att_distances.append(d)
		sum_att_distances = np.sum(att_distances)
		distances.append(sum_att_distances)
	sorted_distances = np.argsort(distances)			# from smaller to largest sum
	new_seen = sorted_distances[:n_seen_classes] if inverse else sorted_distances[n_unseen_classes:]
	new_unseen = sorted_distances[n_seen_classes:] if inverse else sorted_distances[:n_unseen_classes]
	return new_seen, new_unseen, attributes


''' Minimal Attribute Split (MAS)
Removes unnecessary (i.e. highly correlated) attributes.
We measure correlation between attributes i and j in a class as the ratio of co-occurrencies of i and j over i or j. Notice that this is not symmetric.
TODO TO FIX! The point was to remove attributes, not to define new splits!
'''
def minimal_attribute_split(features, labels, attributes, inverse):
	correlations = []
	for a1 in attributes:
		att_correlations = []
		for a2 in attributes:
			d = np.correlate(a1, a2)
			# print(d)
			att_correlations.append(d)
		sum_att_correlations = np.sum(att_correlations)
		# print('---')
		print(sum_att_correlations)
		# print('----------------------------------')
		correlations.append(sum_att_correlations)
	sorted_correlations = np.argsort(correlations)			# from smaller to largest sum
	# TODO seen and unseen remain the same, return new attributes
	# this is probably wrong anyway
	new_seen = sorted_correlations[:n_seen_classes] if inverse else sorted_correlations[n_unseen_classes:]
	new_unseen = sorted_correlations[n_seen_classes:] if inverse else sorted_correlations[:n_unseen_classes]
	return new_seen, new_unseen, attributes


# TODO minimal correlation split: generate a series of random splits until you get one with correlation < K
def minimal_correlation_split(features, labels, attributes, inverse):
	att_correlations = 1000
	while att_correlations > 910:
		old_seen = torch.from_numpy(np.unique(seen_labels))
		old_unseen = torch.from_numpy(np.unique(unseen_labels))
		old_classes = np.concatenate((old_seen, old_unseen))
		np.random.shuffle(old_classes)
		new_seen = old_classes[:int(n_seen_classes)]
		new_unseen = old_classes[int(n_seen_classes):]
		# TODO correlations
		seen_attributes = attributes[new_seen]
		att_correlations = 0
		for a1 in seen_attributes:
			corr = []
			for a2 in seen_attributes:
				d = np.correlate(a1, a2)
				corr.append(d)
			sum_corr = np.sum(corr)
			att_correlations += sum_corr
	return new_seen, new_unseen, attributes


def pca_attribute_split(features, labels, attributes, inverse):
	new_attributes = attributes
	pca = PCA(n_components=args.pca_components)
	new_attributes = pca.fit_transform(new_attributes)
	old_seen = torch.from_numpy(np.unique(seen_labels))
	old_unseen = torch.from_numpy(np.unique(unseen_labels))
	return old_seen, old_unseen, new_attributes


args = get_args()

split_types = {
	'rnd': random_split,
	'gcs': greedy_class_split,
	'ccs': clustered_class_split,
	'mas': minimal_attribute_split,
	'mcs': minimal_correlation_split,
	'pca': pca_attribute_split,
}


# load dataset
matcontent_res101 = sio.loadmat(args.path + '/res101.mat')
matcontent_att_splits = sio.loadmat(args.path + '/att_splits.mat')

# get data: features, labels, and attributes
features = matcontent_res101['features'].T
labels = matcontent_res101['labels'].astype(int).squeeze() - 1
attributes = matcontent_att_splits['att'].T

# get loc data and splits
test_seen_loc = matcontent_att_splits['test_seen_loc'].squeeze() - 1		# tot 4958 - seen classes (GZSL testing) - tot (test_seen + test_unseen + trainval) 30475
test_unseen_loc = matcontent_att_splits['test_unseen_loc'].squeeze() - 1	# tot 5685 - unseen classes (ZSL/GZSL testing) - tot test (seen + unseen) 10643
trainval_loc = matcontent_att_splits['trainval_loc'].squeeze() - 1			# tot 19832 - (train + val - test_seen)
train_loc = matcontent_att_splits['train_loc'].squeeze() - 1				# ONLY VALIDATION MODE - tot 16864
val_loc = matcontent_att_splits['val_loc'].squeeze() - 1					# ONLY VALIDATION MODE - tot 7926 - ...val_unseen_loc (but includes test_seen, why?)

test_seen_ratio = test_seen_loc.size / (test_seen_loc.size + trainval_loc.size)

trainval_loc = matcontent_att_splits['trainval_loc'].squeeze() - 1
test_unseen_loc = matcontent_att_splits['test_unseen_loc'].squeeze() - 1
seen_labels = torch.from_numpy(labels[trainval_loc]).long().numpy()
unseen_labels = torch.from_numpy(labels[test_unseen_loc]).long().numpy()
n_seen_classes = torch.from_numpy(np.unique(seen_labels)).size(0)
n_unseen_classes = torch.from_numpy(np.unique(unseen_labels)).size(0)

# print data info
print('Features: ' + str(features.shape))
print('Labels: ' + str(labels.shape))
print('Attributes: ' + str(attributes.shape))
print('Seen classes: ' + str(n_seen_classes))
print('Unseen classes: ' + str(n_unseen_classes))

new_seen, new_unseen, new_attributes = split_types[args.split](features, labels, attributes, args.inverse)
matcontent_att_splits_new = matcontent_att_splits.copy()

# get new seen_loc and unseen_loc from new splits
seen_loc = np.where(np.in1d(labels, new_seen))[0]
test_unseen_loc = np.where(np.in1d(labels, new_unseen))[0]

# TODO found the problem: you have to randomize here because labels are ordered
np.random.shuffle(seen_loc)
test_seen_loc = seen_loc[:int(test_seen_ratio * seen_loc.size)]
trainval_loc = seen_loc[int(test_seen_ratio * seen_loc.size):]
matcontent_att_splits_new['test_seen_loc'] = test_seen_loc + 1
matcontent_att_splits_new['test_unseen_loc'] = test_unseen_loc + 1
matcontent_att_splits_new['trainval_loc'] = trainval_loc + 1

print(matcontent_att_splits_new['att'])
print(matcontent_att_splits_new['att'].shape)
matcontent_att_splits_new['att'] = new_attributes.T
print('------------------------------------')
print(matcontent_att_splits_new['att'])
print(matcontent_att_splits_new['att'].shape)

if args.save:
	if args.split == 'pca':
		args.split += str(args.pca_components)
	sio.savemat(args.path + '/att_splits_' + args.split + '.mat', matcontent_att_splits_new)
	print('Saved')
