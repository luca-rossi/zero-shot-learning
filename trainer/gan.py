import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, ReLU, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from functools import partial
import numpy as np
from losses import wasserstein_loss, gradient_penalty_loss
from config import *

class RandomWeightedAverage(tf.keras.layers.Layer):
	def call(self, inputs, **kwargs):
		alpha = backend.random_uniform((inputs[2], 1))
		return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class Gan:
	def __init__(self, classifier, n_attributes=N_ATTRIBUTES, input_shape=INPUT_SHAPE, n_batch=N_BATCH_GAN, latent_dim=LATENT_DIM):
		self.critic = self.__build_critic(n_attributes, input_shape)
		self.generator = self.__build_generator(n_attributes, latent_dim)
		self.classifier = classifier
		self.compiled_critic = self.__compile_critic(n_batch)
		self.compiled_generator = self.__compile_generator()

	def __build_critic(self, n_attributes, input_shape):
		model = Sequential()
		model.add(Dense(4096, input_dim=(n_attributes + input_shape), kernel_initializer=RandomNormal(stddev=0.02)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, kernel_initializer=RandomNormal(stddev=0.02)))
		print('-----------------------')
		print('Critic')
		model.summary()
		feature = Input(shape=(input_shape,))
		label = Input(shape=(n_attributes,))
		model_input = Concatenate()([feature, label])
		validity = model(model_input)
		return Model([feature, label], validity)

	def __build_generator(self, n_attributes, latent_dim):
		model = Sequential()
		model.add(Dense(4096, input_dim=(n_attributes + latent_dim), kernel_initializer=RandomNormal(stddev=0.02)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(INPUT_SHAPE, kernel_initializer=RandomNormal(stddev=0.02)))
		model.add(ReLU())
		print('-----------------------')
		print('Generator')
		model.summary()
		noise = Input(shape=(latent_dim,))
		label = Input(shape=(n_attributes,))
		model_input = Concatenate()([noise, label])
		img = model(model_input)
		return Model([noise, label], img)

	def __compile_critic(self, n_batch):
		# freeze generator's layers while training critic
		self.critic.trainable = True
		self.generator.trainable = False
		# features input (real sample)
		real_features = Input(shape=INPUT_SHAPE)
		# noise input
		z_disc = Input(shape=(LATENT_DIM,))
		# generate features based of noise (fake sample) and add label to the input
		label = Input(shape=(N_ATTRIBUTES,))
		fake_features = self.generator([z_disc, label])
		# discriminator determines validity of the real and fake images
		fake = self.critic([fake_features, label])
		valid = self.critic([real_features, label])
		# construct weighted average between real and fake images
		interpolated_features = RandomWeightedAverage()(inputs=[real_features, fake_features, n_batch])
		# determine validity of weighted sample
		validity_interpolated = self.critic([interpolated_features, label])
		# use Python partial to provide loss function with additional 'averaged_samples' argument
		partial_gp_loss = partial(gradient_penalty_loss,
								  averaged_samples=interpolated_features)
		partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names
		c_model = Model(inputs=[real_features, label, z_disc], outputs=[valid, fake, validity_interpolated])
		opt = Adam(lr=LEARNING_RATE, beta_1=BETA)
		c_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
						optimizer=opt, loss_weights=[1, 1, GRADIENT_PENALTY_WEIGHT],
						experimental_run_tf_function=False)
		return c_model

	def __compile_generator(self):
		# for the generator we freeze the critic's layers + classification Layers
		self.critic.trainable = False
		self.classifier.trainable = False
		self.generator.trainable = True
		# sampled noise for input to generator
		z_gen = Input(shape=(LATENT_DIM,))
		# add label to the input
		label = Input(shape=(N_ATTRIBUTES,))
		# generate images based of noise
		features = self.generator([z_gen, label])
		# discriminator determines validity
		valid = self.critic([features, label])
		# discriminator determines class
		classx = self.classifier(features)
		g_model = Model([z_gen, label], [valid, classx])
		opt = Adam(lr=LEARNING_RATE, beta_1=BETA)
		g_model.compile(loss=[wasserstein_loss, SparseCategoricalCrossentropy(from_logits=True)],
						optimizer=opt, loss_weights=[1, CLS_LOSS_WEIGHT],
						experimental_run_tf_function=False)
		return g_model

	def train(self, trainX, trainA, trainY, n_batch=N_BATCH_GAN, n_epochs=N_EPOCHS_GAN, load=LOAD,
			  n_critic=N_CRITIC, latent_dim=LATENT_DIM, save_every=SAVE_MODEL_EVERY, save_path=MODEL_SAVE_PATH):
		if load:
			self.generator.load_weights(save_path)
			return
		# adversarial ground truths
		valid = -np.ones((n_batch, 1))
		fake = np.ones((n_batch, 1))
		# dummy gt for gradient penalty
		dummy = np.zeros((n_batch, 1))
		for epoch in range(n_epochs):
			for i in range(0, trainX.shape[0], n_batch):
				for _ in range(n_critic):
					# select a random batch of images
					idx = np.random.permutation(trainX.shape[0])[0:n_batch]
					features, labels, attr = trainX[idx], trainY[idx], trainA[idx]
					# sample generator input
					noise = np.random.normal(0, 1, (n_batch, latent_dim))
					# train the critic
					d_loss = self.compiled_critic.train_on_batch([features, attr, noise], [valid, fake, dummy])
				noise = np.random.normal(0, 1, (n_batch, latent_dim))
				g_loss = self.compiled_generator.train_on_batch([noise, attr], [valid, labels])
			wass = -np.mean(d_loss[1] + d_loss[2])
			print("%d [D loss: %f] [G loss: %f] [wass: %f] [real: %f] [fake: %f]" % (
			epoch, np.mean(d_loss[0]), np.mean(g_loss[0]), wass, np.mean(d_loss[1]), np.mean(d_loss[2])))
			if epoch % save_every == 0:
				self.generator.save_weights(save_path)

	def generate_synth_dataset(self, signatures, labels, n_samples=N_SAMPLES_GAN, latent_dim=LATENT_DIM):
		z_input = np.random.normal(0, 1, (n_samples, latent_dim))
		fakeY = np.random.randint(0, len(labels), n_samples)
		fakeY = labels[fakeY]
		fakeA = signatures[fakeY]
		fakeX = self.generator.predict([z_input, fakeA])
		return fakeX, fakeA, fakeY
