import numpy as np
import tensorflow as tf

class SGLD_LR(tf.keras.Model):
	def __init__(self, hparams):
		super(SGLD_LR, self).__init__()
		self.hparams = hparams
		self.grads_log = []

		self.dense = tf.keras.layers.Dense(
						1, input_shape=(2,),
						use_bias=False,
						kernel_initializer=tf.random_normal_initializer(0.1)
					)

	def call(self, input):
		logit = tf.reshape(self.dense(input), [-1])
		return logit

	def loss(self, input, label):
		logit = self.call(input)
		return tf.reduce_sum(tf.log(tf.sigmoid(label * logit)))

	def gradient(self, input, label):
		with tf.GradientTape() as tape:
			loss = self.loss(input, label)
		grads = tape.gradient(loss, self.variables)
		return grads

	def update(self, input, label, epsilon, dataset_len):
		grads = self.gradient(input, label)
		for i, g in enumerate(grads):
			grads[i] *= dataset_len / self.hparams.batch_size
			prior_grads = tf.sign(self.variables[i])
			noize = tf.random_normal(
						grads[i].shape,
						stddev=epsilon
					)
			grads[i] = (grads[i] + prior_grads) * epsilon / 2 + noize

		for i in range(len(self.variables)):
			self.variables[i].assign_add(grads[i])

		# for variance calculation
		self.grads_log.append(grads)
		if len(self.grads_log) > 30:
			self.grads_log.pop(0)

	def accuracy(self, input, label):
		logit = self.call(input)
		correct = tf.cast((label * logit) > 0, tf.float32)
		return tf.reduce_mean(correct)

	def grads_variances(self):
		vars = []
		for i in range(self.grads_log[0][0].numpy().shape[0]):
			grads = []
			for j in range(len(self.grads_log)):
				grads.append(self.grads_log[j][0].numpy()[i])
			vars.append(np.var(grads))
		return vars
