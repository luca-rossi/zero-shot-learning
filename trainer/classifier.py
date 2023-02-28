from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from config import *

class Classifier:
	def __init__(self, input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE_CLS, beta=BETA):
		self.model = self.__build_model(input_shape, learning_rate, beta)

	def __build_model(self, input_shape, learning_rate, beta):
		model = Sequential()
		model.add(Dense(N_CLASSES, input_shape=(input_shape,), activation='softmax', kernel_initializer=RandomNormal(stddev=0.02)))
		print('-----------------------')
		print('Classifier')
		model.summary()
		feature = Input(shape=(input_shape,))
		classes = model(feature)
		model = Model(feature, classes)
		opt = Adam(learning_rate=learning_rate, beta_1=beta, beta_2=0.999)
		loss = SparseCategoricalCrossentropy(from_logits=True)
		model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
		return model

	def get_model(self):
		return self.model

	def train(self, x, y, n_batch=N_BATCH_CLS, n_epochs=N_EPOCHS_CLS):
		self.model.fit(x, y, batch_size=n_batch, epochs=n_epochs, shuffle=True, verbose=1)

	def train_epoch(self, x, y, n_batch=N_BATCH_CLS):
		idx = np.random.permutation(len(x))
		x, y = x[idx], y[idx]
		n_batches = int(len(x) / n_batch)
		for batch in range(n_batches):
			batchX = x[batch * n_batch : (batch + 1) * n_batch]
			batchY = y[batch * n_batch : (batch + 1) * n_batch]
			self.model.train_on_batch(batchX, batchY)

	def eval(self, x, y):
		score = self.model.evaluate(x, y, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

	def get_per_class_accuracy(self, test_X, test_Y, target_classes):
		test_Y = test_Y.squeeze()
		predicted_label = self.model.predict(test_X).argmax(axis=-1)
		acc_per_class = 0
		for i in target_classes:
			idx = (test_Y == i)
			acc_per_class += np.sum(test_Y[idx] == predicted_label[idx]) / np.sum(idx)
		acc_per_class /= target_classes.shape[0]
		return acc_per_class
