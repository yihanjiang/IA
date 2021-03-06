__author__ = 'yihanjiang'
import numpy as np
import argparse
import math

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

from utils import MultiInputLayer, errors,stack


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_user', type=int, default=3)
    parser.add_argument('-input_block_len', type=int, default=1)
    parser.add_argument('-output_block_len', type=int, default=1)
    parser.add_argument('-num_block', type=int, default=10000)

    parser.add_argument('-random_H', choices = ['random', 'random_same', 'random_diag', 'not', 'zero_diag'], default='random_same')
    parser.add_argument('-random_code', choices = ['random', 'random_int', 'random_uniform'], default='random')
    parser.add_argument('-code_symbol', type=int, default=2)   # only valid when random_int is chosen
    parser.add_argument('-num_epoch',type=int, default=400)

    parser.add_argument('-noise_sigma',type=float, default=0.000)
    parser.add_argument('-learning_rate',type=float, default=0.001)
    parser.add_argument('-batch_size',type=int, default=100)

    parser.add_argument('-num_layer', type=int, default=1)
    parser.add_argument('-pre_code', type=int, default=2)  # 1 for enable precode, 2 for diff source/inward precode
    parser.add_argument('-post_code', type=int, default=1) # 1 can distinguish dest/forward, 0 is not

    parser.add_argument('-is_bias',  type=int, default=1)

    args = parser.parse_args()
    return args


def build_model(args, H_list, U_list, W_list):

    def H_channel(x):
        res_list = []
        for idx in range(len(H_list)):
            HH = K.variable(H_list[idx])
            xx = x[:, :, idx]
            tmp = tf.matmul(xx,HH)  + args.noise_sigma*tf.random_normal(tf.shape(xx),dtype=tf.float32, mean=0., stddev=1.0)
            res_list.append(tmp)

        res = tf.stack(res_list)
        res = tf.transpose(res, perm=[1, 2,0])
        return res

    def U_channel(x):
        res_list = []
        for idx in range(len(U_list)):
            UU = K.variable(U_list[idx])
            xx = x[:, :, idx]
            tmp = tf.matmul(xx,UU)  + args.noise_sigma*tf.random_normal(tf.shape(xx),dtype=tf.float32, mean=0., stddev=1.0)
            res_list.append(tmp)

        res = tf.stack(res_list)
        res = tf.transpose(res, perm=[1, 2,0])
        return res

    def W_channel(x):
        res_list = []
        for idx in range(len(W_list)):
            WW = K.variable(W_list[idx])
            xx = x[:, :, idx]
            tmp = tf.matmul(xx,WW)  + args.noise_sigma*tf.random_normal(tf.shape(xx),dtype=tf.float32, mean=0., stddev=1.0)
            res_list.append(tmp)

        res = tf.stack(res_list)
        res = tf.transpose(res, perm=[1, 2,0])
        return res

    output_act = 'linear'

    inputs = Input(shape = (args.num_user, args.input_block_len))

    # 1st Forward
    if args.pre_code:
        if args.pre_code == 2:
            s_output_fwd = MultiInputLayer(args.output_block_len, use_bias=args.is_bias,activation=output_act, name ='S_fenc_1')(inputs)
            s_output_iwd = MultiInputLayer(args.output_block_len, use_bias=args.is_bias,activation=output_act, name ='S_ienc_1')(inputs)
        else:
            s_output_fwd = MultiInputLayer(args.output_block_len, use_bias=args.is_bias,activation=output_act, name ='S_enc_1')(inputs)
            s_output_iwd = s_output_fwd
    else:
        s_output_fwd = inputs
        s_output_iwd = inputs

    d_received_1 = Lambda(H_channel, name = '1st_Forward_channel')(s_output_fwd)
    s_received   = Lambda(U_channel, name = '1st_Source_channel')(s_output_iwd)
    s_pre        = keras.layers.Concatenate(axis=2)([inputs,s_received])

    # 2nd Forward
    s_output     = MultiInputLayer(args.output_block_len, use_bias=args.is_bias, activation=output_act, name ='S_enc_2')(s_pre)

    d_received_2 = Lambda(H_channel, name = '2nd_Forward_channel')(s_output)
    d_enc_1      = MultiInputLayer(args.output_block_len, use_bias=args.is_bias,activation=output_act, name ='D_enc_1')(d_received_1)
    d_enc_rec_1  = Lambda(W_channel, name = '2nd_Dest_channel')(d_enc_1)

    if args.post_code == 1:
        D_final      = keras.layers.Concatenate(axis=2)([d_enc_rec_1,d_received_2, d_received_1])
    else:
        x = keras.layers.Add(axis = 2)([d_enc_rec_1,d_received_2])
        D_final  = keras.layers.Concatenate(axis=2)([x, d_received_1])

    t_output     = MultiInputLayer(args.input_block_len, use_bias=args.is_bias,activation=output_act, name ='final')(D_final)

    return Model(inputs, t_output)

def main():
    args = get_args()
    print args

    H_list, U_list, W_list = [], [], []

    if args.random_H == 'random_same':
        H_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
        U_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
        W_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))

        for idx in range(args.output_block_len):
            H_list.append(H_matrix)
            U_list.append(U_matrix)
            W_list.append(W_matrix)
    else:
        for idx in range(args.output_block_len):
            H_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
            U_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
            W_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
            H_list.append(H_matrix)
            U_list.append(U_matrix)
            W_list.append(W_matrix)

    learning_rate = args.learning_rate
    optimizer = Adam(learning_rate)

    model = build_model(args, H_list, U_list, W_list)

    if args.random_code == 'random_int':
        loss = 'binary_crossentropy'
        model.compile(loss=loss, optimizer=optimizer, metrics=[errors])
    else:
        loss = 'mse'
        model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

    model.summary()

    print 'H',
    print H_matrix
    print 'U'
    print U_matrix
    print 'W'
    print W_matrix

    num_block_train = args.num_block
    num_block_test  = args.num_block/10

    if args.random_code == 'random_int':
        message_train  = np.random.randint(0,args.code_symbol, size = (num_block_train, args.num_user, args.input_block_len))
        message_test   = np.random.randint(0,args.code_symbol, size = (num_block_test, args.num_user, args.input_block_len))
    elif args.random_code == 'random':
        message_train = np.random.normal(0, 1, size = (num_block_train, args.num_user, args.input_block_len))
        message_test = np.random.normal(0, 1, size = (num_block_train, args.num_user, args.input_block_len))
    elif args.random_code == 'random_uniform':
        message_train = np.random.uniform(low=-args.signal_sigma, high=args.signal_sigma,
                                          size = (num_block_train, args.num_user, args.input_block_len))
        message_test  = np.random.uniform(low=-args.signal_sigma, high=args.signal_sigma,
                                          size = (num_block_train, args.num_user, args.input_block_len))
    model.fit(message_train, message_train,
              validation_data=(message_test, message_test), batch_size=args.batch_size, epochs=args.num_epoch)

    model.save('./tmp/test_inband.h5')

if __name__ == '__main__':
    main()
