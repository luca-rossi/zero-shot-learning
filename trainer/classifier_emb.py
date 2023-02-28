from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from sklearn.neighbors import KDTree
from config import *

CLASS_VECTORS = list()

def custom_kernel_init(shape, dtype=None):
	return CLASS_VECTORS.T

class EmbeddedClassifier:
	def __init__(self, signatures, input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE_CLS, beta=BETA):
		self.model = self.__build_model(signatures, input_shape, learning_rate, beta)

	def __build_model(self, signatures, input_shape, learning_rate, beta):
		global CLASS_VECTORS
		CLASS_VECTORS = signatures
		model = Sequential()
		model.add(Dense(signatures.shape[1], input_shape=(input_shape,), activation='relu', kernel_initializer=RandomNormal(stddev=0.02)))
		model.add(Dense(signatures.shape[0], activation='softmax', trainable=False, kernel_initializer=custom_kernel_init))
		print('-----------------------')
		print('Classifier')
		model.summary()
		adam = Adam(lr=learning_rate, beta_1=beta)
		model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=adam, metrics=['accuracy'])
		return model

	def get_model(self):
		return self.model

	def train(self, trainX, trainY, n_batch=N_BATCH_CLS, n_epochs=N_EPOCHS_CLS):
		self.model.fit(trainX, trainY, verbose=2, epochs=n_epochs, batch_size=n_batch, shuffle=True)

	def eval(self, x, y, classnames, signatures):
		print()
		# score = self.model.evaluate(x, y, verbose=0)
		# print('Test loss:', score[0])
		# print('Test accuracy:', score[1])
		inp = self.model.input
		out = self.model.layers[-2].output
		model = Model(inp, out)
		predY = model.predict(x)
		tree = KDTree(signatures)
		top5, top3, top1 = 0, 0, 0
		for i, pred in enumerate(predY):
			pred = np.expand_dims(pred, axis=0)
			dist_5, index_5 = tree.query(pred, k=5)
			# TODO fix index_5 !!!
			pred_labels = [classnames[index] for index in index_5[0]]
			true_label = y[i]
			if true_label in pred_labels:
				top5 += 1
			if true_label in pred_labels[:3]:
				top3 += 1
			if true_label == pred_labels[0]:
				top1 += 1
		print("ZERO SHOT LEARNING SCORE")
		print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(x))))
		print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(x))))
		print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(x))))
