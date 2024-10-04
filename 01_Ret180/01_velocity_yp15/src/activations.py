import tensorflow as tf
def thres_relu(x,app):
   return tf.keras.activations.relu(x, threshold=app.RELU_THRESHOLD)
