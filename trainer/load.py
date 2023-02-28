import numpy as np
import scipy.io
from sklearn import preprocessing
from config import *

TRAINVAL_COL = 'trainval_loc'
TRAIN_COL = 'train_loc'
VAL_COL = 'val_loc'
TEST_COL = 'test_unseen_loc'
TEST_SEEN_COL = 'test_seen_loc'

def get_mat():
	'''
	From the .mat files extract all the features from resnet and the attribute splits. 
	- The res101 contains features and the corresponding labels.
	- att_splits contains the different splits for trainval, train, val and test set.
	'''
	res101 = scipy.io.loadmat(FEATURES_PATH)
	att_splits = scipy.io.loadmat(ATTRIBUTES_PATH)
	return res101, att_splits

def map_labels(source, dest):
	for label in source:
		dest[dest == label] = label
	return dest

def get_labels(res101, att_splits):
	'''
	We need the corresponding ground-truth labels/classes for each training example
	for all our train, val, trainval and test set according to the split locations provided.
	'''
	labels = dict()
	labels['all'] = res101['labels'] - 1
	labels['train'] = labels['all'][np.squeeze(att_splits[TRAIN_COL] - 1)]
	labels['val'] = labels['all'][np.squeeze(att_splits[VAL_COL] - 1)]
	labels['trainval'] = labels['all'][np.squeeze(att_splits[TRAINVAL_COL] - 1)]
	labels['test'] = labels['all'][np.squeeze(att_splits[TEST_COL] - 1)]
	labels['testval'] = labels['all'][np.squeeze(att_splits[TEST_SEEN_COL] - 1)]
	labels['train_seen'] = np.unique(labels['train'])
	labels['val_unseen'] = np.unique(labels['val'])
	labels['trainval_seen'] = np.unique(labels['trainval'])
	labels['test_unseen'] = np.unique(labels['test'])
	labels['test_seen'] = np.unique(labels['testval'])
	labels['train'] = map_labels(labels['train_seen'], labels['train'])
	labels['val'] = map_labels(labels['val_unseen'], labels['val'])
	labels['trainval'] = map_labels(labels['trainval_seen'], labels['trainval'])
	labels['test'] = map_labels(labels['test_unseen'], labels['test'])
	labels['testval'] = map_labels(labels['test_seen'], labels['testval'])
	return labels

def get_features(res101, att_splits):
	'''
	Let us denote the features X ∈ [d×m] available at training stage,
	where d is the dimensionality of the data, and m is the number of instances.
	'''
	features = dict()
	scaler = preprocessing.MinMaxScaler()
	features['all'] = res101['features'].transpose()
	features['trainval'] = features['all'][np.squeeze(att_splits[TRAINVAL_COL] - 1), :]
	features['trainval'] = scaler.fit_transform(features['trainval'])
	features['train'] = features['all'][np.squeeze(att_splits[TRAIN_COL] - 1), :]
	features['train'] = scaler.transform(features['train'])
	features['val'] = features['all'][np.squeeze(att_splits[VAL_COL] - 1), :]
	features['val'] = scaler.transform(features['val'])
	features['test'] = features['all'][np.squeeze(att_splits[TEST_COL] - 1), :]
	features['test'] = scaler.transform(features['test'])
	features['testval'] = features['all'][np.squeeze(att_splits[TEST_SEEN_COL] - 1), :]
	features['testval'] = scaler.transform(features['testval'])
	features['all'] = scaler.transform(features['all'])
	return features

def get_signatures(att_splits):
	'''Each of the classes in the dataset have an attribute (a) description.
	This vector is known as the `Signature matrix` of dimension S ∈ [0, 1]a×z.
	For training stage there are z classes and z' classes for test S ∈ [0, 1]a×z'.
	The occurance of an attribute corresponding to the class is given.
	For instance, if the classes are `horse` and `zebra` and the corresponding attributes are
	[wild_animal, 4_legged, carnivore]
	```
	Horse	  	Zebra
	[0.00354613 0.00000000] Domestic_animal
	[0.13829921 0.20209503] 4_legged
	[0.06560347 0.04155225] carnivore
	```
	'''
	attrs = att_splits['att'].transpose()
	signatures = list()
	for i, attr in enumerate(attrs):
		signatures.append((i, attr))
	classnames, signatures = zip(*signatures)
	classnames = list(classnames)
	signatures = np.asarray(signatures, dtype=np.float)
	return signatures, classnames


def get_attributes(labels, signatures):
	attributes = dict()
	attributes['all'] = np.array([signatures[y] for y in labels['all']]).squeeze()
	attributes['train'] = np.array([signatures[y] for y in labels['train']]).squeeze()
	attributes['val'] = np.array([signatures[y] for y in labels['val']]).squeeze()
	attributes['trainval'] = np.array([signatures[y] for y in labels['trainval']]).squeeze()
	attributes['test'] = np.array([signatures[y] for y in labels['test']]).squeeze()
	attributes['testval'] = np.array([signatures[y] for y in labels['testval']]).squeeze()
	return attributes

def load_data():
	'''
	att_splits keys: 'allclasses_names', 'att', 'original_att',
	'test_seen_loc', 'test_unseen_loc', 'train_loc', 'trainval_loc', 'val_loc'
	'''
	res101, att_splits = get_mat()
	labels = get_labels(res101, att_splits)
	features = get_features(res101, att_splits)
	signatures, classnames = get_signatures(att_splits)
	attributes = get_attributes(labels, signatures)
	return features, attributes, labels, signatures, classnames
