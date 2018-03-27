__author__ = 'yihanjiang'

from keras.engine.topology import Layer
import keras
from keras import backend as K
import tensorflow as tf

class MultiInputLayer(Layer):
    def __init__(self, output_code_len, use_bias = True, activation=None, **kwargs):
        super(MultiInputLayer, self).__init__(**kwargs)
        self.output_code_len = output_code_len
        self.activation      = keras.activations.get(activation)
        self.use_bias        = use_bias

    def build(self, input_shape):
        self.num_user = input_shape[1]
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (input_shape[1], input_shape[2], self.output_code_len),
                                      initializer='glorot_uniform',
                                      trainable=True)

        # Create a trainable weight variable for this layer.
        if self.use_bias:
            self.bias   = self.add_weight(name='bias',
                                          shape=(input_shape[1], self.output_code_len),
                                          initializer='glorot_uniform',
                                          trainable=True)

        super(MultiInputLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        res = []
        for idx in range(self.num_user):
            tmp = tf.matmul(x[:,idx,:], self.kernel[idx, :, :])
            res.append(tmp)

        output = tf.stack(res)
        output = tf.transpose(output, perm = [1, 0, 2])

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.output_code_len )


def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

def stack(args):
    s1, s2 = args
    return tf.stack([s1, s2], axis=2)
