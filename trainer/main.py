from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
from classifier_emb import EmbeddedClassifier
from classifier import Classifier
from gan import Gan
from load import load_data
from config import *

disable_eager_execution()
features, attributes, labels, signatures, classnames = load_data()
train_seenX, train_seenA, train_seenY = features['trainval'], attributes['trainval'], labels['trainval']
test_unseenX, test_unseenA, test_unseenY = features['test'], attributes['test'], labels['test']
test_seenX, test_seenA, test_seenY = features['testval'], attributes['testval'], labels['testval']

if GAN:
	# create and train classifier for GAN
	precls = Classifier()
	precls.train(train_seenX, train_seenY)
	precls.eval(test_unseenX, test_unseenY)
	precls.eval(test_seenX, test_seenY)
	# create and train GAN, create synthetic dataset
	gan = Gan(precls.get_model())
	gan.train(train_seenX, train_seenA, train_seenY)
	fake_trainX, fake_trainA, fake_trainY = gan.generate_synth_dataset(signatures, labels['test_unseen'])
	if GZSL:
		fake_trainX = np.concatenate((fake_trainX, train_seenX))
		fake_trainA = np.concatenate((fake_trainA, train_seenA))
		fake_trainY = np.concatenate((fake_trainY, train_seenY.squeeze()))
	# create, train, and test final classifier
	postcls = Classifier()
	best_acc_seen, best_acc_unseen, best_H, best_epoch = 0, 0, 0, 0
	for epoch in range(N_EPOCHS_CLS):
		postcls.train_epoch(fake_trainX, fake_trainY)
		acc_seen = postcls.get_per_class_accuracy(test_seenX, test_seenY, labels['test_seen'])
		acc_unseen = postcls.get_per_class_accuracy(test_unseenX, test_unseenY, labels['test_unseen'])
		H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
		if H > best_H:
			best_acc_seen, best_acc_unseen, best_H, best_epoch = acc_seen, acc_unseen, H, epoch
	print('acc_seen=%.4f, acc_unseen=%.4f, h=%.4f, epoch=%d' % (best_acc_seen, best_acc_unseen, best_H, best_epoch))
else:
	cls = EmbeddedClassifier(signatures)
	cls.train(train_seenX, train_seenY)
	signatures_eval = signatures if GZSL else signatures[labels['test_unseen'], :]
	cls.eval(test_unseenX, test_unseenY, classnames, signatures_eval)
	cls.eval(test_seenX, test_seenY, classnames, signatures_eval)
