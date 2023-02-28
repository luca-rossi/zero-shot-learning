import tensorflow as tf
from tensorflow.keras import backend
import numpy as np

def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

def _compute_gradients(tensor, var_list):
	grads = tf.gradients(tensor, var_list)
	return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
	# first get the gradients, assuming:
	# - that y_pred has dimensions (batch_size, 1)
	# - averaged_samples has dimensions (batch_size, nbr_features)
	# gradients afterwards has dimension (batch_size, nbr_features), basically
	# a list of nbr_features-dimensional gradient vectors
	gradients = _compute_gradients(y_pred, [averaged_samples])[0]
	# compute the euclidean norm by squaring ...
	gradients_sqr = backend.square(gradients)
	# ... summing over the rows ...
	gradients_sqr_sum = backend.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
	# ... and sqrt
	gradient_l2_norm = backend.sqrt(gradients_sqr_sum)
	# compute lambda * (1 - ||grad||)^2 still for each single sample
	gradient_penalty = backend.square(1 - gradient_l2_norm)
	# return the mean as loss over all the batch samples
	return backend.mean(gradient_penalty)
