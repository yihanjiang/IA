__author__ = 'yihanjiang'
import numpy as np
import tensorflow as tf
import keras
import math
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, Conv1D
from keras.layers import BatchNormalization, Lambda
from keras.models import Sequential, Model
from keras.constraints import Constraint
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_user', type=int, default=3)

    parser.add_argument('-random_H', choices = ['random', 'random_int', 'random_diag', 'not', 'zero_diag'], default='random')
    parser.add_argument('-random_U', choices = ['random', 'random_int', 'random_diag', 'not', 'zero_diag'], default='random')
    parser.add_argument('-random_W', choices = ['random', 'random_int', 'random_diag', 'not', 'zero_diag'], default='random')
    parser.add_argument('-G_mode', choices = ['random', 'H.T'], default='H.T')
    parser.add_argument('-random_code', choices = ['random', 'random_int'], default='random_int')
    parser.add_argument('-code_symbol', type=int, default=2)   # only valid when random_int is chosen
    parser.add_argument('-num_epoch',type=int, default=400)

    parser.add_argument('-noise_sigma',type=float, default=0.000)

    args = parser.parse_args()
    return args


class MyLayer(Layer):
    def __init__(self, output_dim, use_bias = True, activation=None, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

        self.activation = keras.activations.get(activation)
        self.use_bias   = use_bias

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

        self.bias   = self.add_weight(name='bias',
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        output = tf.multiply(x, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class YourLayer(Layer):
    def __init__(self, output_dim, use_bias = True, activation=None, **kwargs):
        self.output_dim = output_dim
        super(YourLayer, self).__init__(**kwargs)

        self.activation = keras.activations.get(activation)
        self.use_bias   = use_bias

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, 2),
                                      initializer='glorot_uniform',
                                      trainable=True)

        self.bias   = self.add_weight(name='bias',
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(YourLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        output = tf.multiply(x, self.kernel)
        output =tf.reduce_sum(output, axis = 2)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def clip(x):
    return K.clip(x, min_value=-2.0, max_value=2.0)


def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

def stack(args):
    s1, s2 = args
    return tf.stack([s1, s2], axis=2)


def build_model(num_user, H, U, W, noise_sigma):

    def H_channel(x):
        HH = K.variable(H)
        return K.dot(x, HH) + noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)

    def U_channel(x):
        UU = K.variable(U)
        return K.dot(x, UU) + noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)


    def W_channel(x):
        WW = K.variable(W)
        return K.dot(x, WW) + noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)



    act = 'linear'
    output_act = 'linear'

    inputs = Input(shape = (num_user,))
    x = inputs

    # 1st Forward
    s_output = MyLayer(num_user, activation=output_act, name ='S_enc_1')(x)

    # 1st Backward
    d_received_1 = Lambda(H_channel, name = '1st_Forward_channel')(s_output)
    s_received   = Lambda(U_channel, name = '1st_Source_channel')(s_output)

    s_pre        = Lambda(stack,     name = 'U_and_inputs')([inputs, s_received])
    s_output     = YourLayer(num_user, activation=output_act, name ='S_enc_2')(s_pre)

    d_received_2 = Lambda(H_channel, name = '2nd_Forward_channel')(s_output)
    d_enc_1      = MyLayer(num_user, activation=output_act, name ='D_enc_1')(d_received_1)
    d_enc_rec_1  = Lambda(W_channel, name = '2nd_Dest_channel')(d_enc_1)
    D_final      = keras.layers.Add()([d_enc_rec_1, d_received_2])
    t_output     = MyLayer(num_user, activation=output_act, name ='final')(D_final)

    return Model(inputs, t_output)

def main():
    args = get_args()

    print args
    num_user = args.num_user

    if args.random_H == 'random':
        H_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
    elif args.random_H == 'random_int':
        H_matrix = np.random.randint(0, 2, size = (num_user, num_user))
    elif args.random_H == 'random_diag':
        v = np.random.normal(0, 1.0, size = (num_user,1))
        u = np.random.normal(0, 1.0, size = (num_user,1))
        H_matrix = np.diag(np.random.normal(0, 1.0, size = (num_user))) + np.dot(v, u.T)

    elif args.random_H == 'zero_diag':
        H_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
        for idx in range(num_user):
            H_matrix[idx][idx] = 0.0
    else:
        H_matrix = np.ones((num_user,num_user)) - np.eye(num_user)


    if args.random_U == 'random':
        U_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
    elif args.random_U == 'random_int':
        U_matrix = np.random.randint(0, 2, size = (num_user, num_user))
    elif args.random_U == 'random_diag':
        v = np.random.normal(0, 1.0, size = (num_user,1))
        u = np.random.normal(0, 1.0, size = (num_user,1))
        U_matrix = np.diag(np.random.normal(0, 1.0, size = (num_user))) + np.dot(v, u.T)

    elif args.random_U == 'zero_diag':
        U_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
        for idx in range(num_user):
            U_matrix[idx][idx] = 0.0
    else:
        U_matrix = np.ones((num_user,num_user)) - np.eye(num_user)

    if args.random_W == 'random':
        W_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
    elif args.random_W == 'random_int':
        W_matrix = np.random.randint(0, 2, size = (num_user, num_user))
    elif args.random_W == 'random_diag':
        v = np.random.normal(0, 1.0, size = (num_user,1))
        u = np.random.normal(0, 1.0, size = (num_user,1))
        W_matrix = np.diag(np.random.normal(0, 1.0, size = (num_user))) + np.dot(v, u.T)

    elif args.random_W == 'zero_diag':
        W_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
        for idx in range(num_user):
            W_matrix[idx][idx] = 0.0
    else:
        W_matrix = np.ones((num_user,num_user)) - np.eye(num_user)

    learning_rate = 0.001
    optimizer = Adam(learning_rate)

    model = build_model(num_user, H_matrix, U_matrix, W_matrix, noise_sigma = args.noise_sigma)

    loss = 'mse'  #'binary_crossentropy'
    if args.random_code == 'random_int':
        model.compile(loss=loss, optimizer=optimizer, metrics=[errors])
    else:
        model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

    model.summary()

    print 'H',
    print H_matrix
    print 'U'
    print U_matrix
    print 'W'
    print W_matrix

    num_block_train = 5000
    num_block_test  = 1000

    if args.random_code == 'random_int':
        message_train  = np.random.randint(0,args.code_symbol, size = (num_block_train, num_user))
        message_test   = np.random.randint(0,args.code_symbol, size = (num_block_test, num_user))
    elif args.random_code == 'random':
        message_train = np.random.normal(0, 1, size = (num_block_train, num_user))
        message_test = np.random.normal(0, 1, size = (num_block_train, num_user))

    def scheduler(epoch):

        if epoch > 300 and epoch <=400:
            print 'changing by /10 lr'
            lr = learning_rate/10.0
        elif epoch >400 and epoch <=550:
            print 'changing by /100 lr'
            lr = learning_rate/100.0
        elif epoch >550 and epoch <=700:
            print 'changing by /1000 lr'
            lr = learning_rate/1000.0
        elif epoch > 700:
            print 'changing by /10000 lr'
            lr = learning_rate/10000.0
        else:
            lr = learning_rate

        return lr
    change_lr = LearningRateScheduler(scheduler)


    model.fit(message_train, message_train,
              callbacks=[change_lr] ,
              validation_data=(message_test, message_test), batch_size=10, epochs=args.num_epoch)

    model.save('./tmp/test_noisy.h5')

if __name__ == '__main__':
    main()
