import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

# class myTFRNNModel(keras.Model):
#     def __init__(self, max_length):
#         super().__init__()
#         self.embed_layer = layers.Embedding(10, 32, input_length=max_length)
#         self.rnn_cell = layers.SimpleRNNCell(64)
#         self.rnn_layer = layers.RNN(self.rnn_cell, return_sequences=True)
#         self.dense = layers.Dense(10, activation=tf.nn.softmax)
#
#     @tf.function
#     def call(self, num1, num2):
#         logits1, logits2 = self.embed_layer(num1), self.embed_layer(num2)
#         logits = tf.concat([logits1, logits2], 2)
#         logits = self.rnn_layer(logits)
#         logits = self.dense(logits)
#         return logits
#
# @tf.function
# def compute_loss(logits, labels):
#     losses = tf.keras.losses.MSE(logits, labels)
#     return tf.reduce_mean(losses)
#
# with tf.GradientTape() as tape:
#     x = np.random.randint(0, 10, (1000, 10)).astype(float)
#     model = myTFRNNModel(10)
#     y = model(x, x)
#     # loss = compute_loss(y, x)
#     # grads = tape.gradient(loss, model.trainable_variables)
#     # optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     print(x)
#     print(y)

y = tf.constant([1, 2, 3])
y1 = tf.keras.backend.get_value(y)
print(y1)
