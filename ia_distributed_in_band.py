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

    parser.add_argument('-num_user', type=int, default=6)
    parser.add_argument('-input_block_len', type=int, default=1)
    parser.add_argument('-output_block_len', type=int, default=1)
    parser.add_argument('-num_block', type=int, default=10000)

    parser.add_argument('-random_H', choices = ['random', 'random_same', 'H_only'], default='random')
    parser.add_argument('-random_code', choices = ['random', 'random_int', 'random_uniform'], default='random')
    parser.add_argument('-code_symbol', type=int, default=2)   # only valid when random_int is chosen
    parser.add_argument('-num_epoch',type=int, default=400)

    parser.add_argument('-noise_sigma',type=float, default=0.000)
    parser.add_argument('-learning_rate',type=float, default=0.001)
    parser.add_argument('-batch_size',type=int, default=100)

    parser.add_argument('-num_layer', type=int, default=1)
    parser.add_argument('-num_hidden_unit', type=int, default=100)
    parser.add_argument('-act_hidden', choices = ['linear', 'relu', 'selu','tanh', 'sigmoid'], default='linear')


    parser.add_argument('-pre_code', type=int, default=1)   # 2 is seperate source/forward pre_code, 1 is not seperated
    parser.add_argument('-is_bias',  type=int, default=1)

    args = parser.parse_args()
    return args

def tf_exp(x, idx):
    final        = tf.expand_dims(x[:, idx,:], 1)
    return final

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
    args.act_hidden
    # inputs
    input_list = [None for _ in range(args.num_user)]
    for idx in range(args.num_user):
        input_list[idx] = Input(shape = (1, args.input_block_len))

    # 1st round forward coding
    forward_code_list = [None for _ in range(args.num_user)]
    for idx in range(args.num_user):
        x = input_list[idx]
        for ldx in range(args.num_layer -1):
            x = MultiInputLayer(args.num_hidden_unit, use_bias=args.is_bias,activation=args.act_hidden,
                            name ='S_fwd1_'+ str(idx) +'_'+  str(ldx))(x)
        res_x = MultiInputLayer(args.output_block_len, use_bias=args.is_bias,activation=output_act,
                            name ='S_fwd1_'+ str(idx) +'_'+  str(args.num_layer))(x)
        forward_code_list[idx] = res_x

    forward_code  = keras.layers.Concatenate(axis = 1)(forward_code_list)

    if args.pre_code ==1:
        inward_code = forward_code
    elif args.pre_code == 2:
        # 1st round inward coding
        inward_code_list = [None for _ in range(args.num_user)]
        for idx in range(args.num_user):
            x = input_list[idx]
            for ldx in range(args.num_layer -1):
                x = MultiInputLayer(args.num_hidden_unit, use_bias=args.is_bias,activation=args.act_hidden,
                                name ='S_iwd1_'+ str(idx) +'_'+ str(ldx))(x)
            res_x = MultiInputLayer(args.output_block_len, use_bias=args.is_bias,activation=output_act,
                                name ='S_iwd1_'+ str(idx) +'_'+ str(args.num_layer))(x)
            inward_code_list[idx] = res_x

        inward_code  = keras.layers.Concatenate(axis = 1)(inward_code_list)

    # 1st round forward and inward channel
    d_forward_1st = Lambda(H_channel, name = '1st_Forward_channel')(forward_code)
    s_inward_1st  = Lambda(U_channel, name = '1st_Source_channel')(inward_code)

    d_forward_1st_list = [None for _ in range(args.num_user)]
    s_inward_1st_list = [None for _ in range(args.num_user)]
    for idx in range(args.num_user):
        d_forward_1st_list[idx]   = Lambda(tf_exp, arguments = {'idx':idx})(d_forward_1st)
        s_inward_1st_list[idx]   = Lambda(tf_exp, arguments = {'idx':idx})(s_inward_1st)

    # 2rd round forward coding
    input_2nd_round_list = [None for _ in range(args.num_user)]
    for idx in range(args.num_user):
        input_2nd_round_list[idx] = keras.layers.Concatenate(axis=2)([input_list[idx],s_inward_1st_list[idx]])

    forward_2nd_list = [None for _ in range(args.num_user)]
    for idx in range(args.num_user):
        x = input_2nd_round_list[idx]
        for ldx in range(args.num_layer -1):
            x = MultiInputLayer(args.num_hidden_unit, use_bias=args.is_bias,activation=args.act_hidden,
                            name ='S_fwd2_'+ str(idx) +'_'+  str(ldx))(x)
        res_x = MultiInputLayer(args.output_block_len, use_bias=args.is_bias,activation=output_act,
                            name ='S_fwd2_'+ str(idx) +'_'+  str(args.num_layer))(x)
        forward_2nd_list[idx] = res_x

    forward_code  = keras.layers.Concatenate(axis = 1)(forward_2nd_list)
    d_forward_2rd = Lambda(H_channel, name = '2nd_Forward_channel')(forward_code)

    # 2rd round outward coding (W channel)
    outward_2nd_list = [None for _ in range(args.num_user)]
    for idx in range(args.num_user):
        x = d_forward_1st_list[idx]
        for ldx in range(args.num_layer -1):
            x = MultiInputLayer(args.num_hidden_unit, use_bias=args.is_bias,activation=args.act_hidden,
                            name ='S_owd2_'+  str(idx) +'_'+ str(ldx))(x)
        res_x = MultiInputLayer(args.output_block_len, use_bias=args.is_bias,activation=output_act,
                            name ='S_owd2_'+  str(idx) +'_'+ str(args.num_layer))(x)
        outward_2nd_list[idx] = res_x

    outward_code  = keras.layers.Concatenate(axis = 1)(outward_2nd_list)

    d_outward_2rd = Lambda(W_channel, name = '2nd_Outward_channel')(outward_code)

    # Final Decode
    D_final      = keras.layers.Concatenate(axis=2)([d_forward_1st, d_forward_2rd,d_outward_2rd])

    output_list = [None for _ in range(args.num_user)]
    for idx in range(args.num_user):
        final   = Lambda(tf_exp, arguments = {'idx':idx})(D_final)
        for ldx in range(args.num_layer -1):
            final = MultiInputLayer(args.num_hidden_unit, use_bias=args.is_bias,activation=args.act_hidden,
                            name ='final_'+  str(idx) +'_'+ str(ldx))(final)

        t_output= MultiInputLayer(args.input_block_len, use_bias=args.is_bias,
                                  activation=output_act, name ='final_'+str(idx))(final)
        output_list[idx] = t_output

    return Model(inputs=input_list, outputs=output_list)

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
    elif args.random_H == 'H_only':
        H_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
        U_matrix = np.zeros(shape = (args.num_user, args.num_user))
        W_matrix = np.zeros(shape = (args.num_user, args.num_user))

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
        losses = [loss for _ in range(args.num_user)]

        model.compile(loss=losses, optimizer=optimizer, metrics=[errors])
    else:
        loss = 'mse'
        losses = [loss for _ in range(args.num_user)]
        model.compile(loss=losses, optimizer=optimizer, metrics=['mae'])

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
        message_train  = [np.random.randint(0,args.code_symbol, size = (num_block_train, 1, args.input_block_len)) for _ in range(args.num_user)]
        message_test   = [np.random.randint(0,args.code_symbol, size = (num_block_test, 1, args.input_block_len)) for _ in range(args.num_user)]
    elif args.random_code == 'random':
        message_train = [np.random.normal(0, 1, size = (num_block_train, 1, args.input_block_len)) for _ in range(args.num_user)]
        message_test = [np.random.normal(0, 1, size = (num_block_test, 1, args.input_block_len)) for _ in range(args.num_user)]
    elif args.random_code == 'random_uniform':
        message_train = [np.random.uniform(low= -args.signal_sigma, high=args.signal_sigma, size = (num_block_train, 1 , args.input_block_len)) for _ in range(args.num_user)]
        message_test  = [np.random.uniform(low=-args.signal_sigma,  high=args.signal_sigma, size = (num_block_test, 1 , args.input_block_len)) for _ in range(args.num_user)]
    model.fit(message_train, message_train,
              validation_data=(message_test, message_test), batch_size=args.batch_size, epochs=args.num_epoch)

    model.save('./tmp/test_inband.h5')

if __name__ == '__main__':
    main()
