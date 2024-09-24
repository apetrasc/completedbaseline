import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, \
                                       ModelCheckpoint, LearningRateScheduler
class SubTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(SubTensorBoard, self).__init__(*args, **kwargs)

    def lr_getter(self):
        # Get vals
        #decay = self.model.optimizer.decay
        decay = self.model.optimizer.learning_rate
        lr = self.model.optimizer.lr
        iters = self.model.optimizer.iterations # only this should not be const
        beta_1 = self.model.optimizer.beta_1
        beta_2 = self.model.optimizer.beta_2
        # calculate
        lr = lr * (1. / (1. + decay * K.cast(iters, K.dtype(decay))))
        t = K.cast(iters, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t)))
        return np.float32(K.eval(lr_t))

    def on_epoch_end(self, episode, logs = {}):
        logs.update({'learning_rate': self.lr_getter()})
        super(SubTensorBoard, self).on_epoch_end(episode, logs)