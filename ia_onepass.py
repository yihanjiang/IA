__author__ = 'yihanjiang'
import numpy as np
import tensorflow as tf
import keras
import math
from keras import backend as K
from keras.layers import Input
from keras.layers import BatchNormalization, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

from utils import MultiInputLayer, errors, stack

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_user', type=int, default=4)

    parser.add_argument('-num_block', type=int, default=100000)

    parser.add_argument('-num_layer',  type=int, default=2)
    parser.add_argument('-num_hidden_unit', type=int, default=100)

    parser.add_argument('-input_block_len', type=int, default=1)
    parser.add_argument('-output_block_len', type=int, default=1)

    parser.add_argument('-num_antenna', type=int, default=2)

    parser.add_argument('-random_H', choices = ['same','random', 'random_int',
                                                'random_diag', 'not', 'zero_diag', 'default'], default='random')
    parser.add_argument('-random_code', choices = ['random', 'random_int', 'random_uniform'], default='random')
    parser.add_argument('-code_symbol', type=int, default=2)   # only valid when random_int is chosen
    parser.add_argument('-num_epoch',type=int, default=400)
    parser.add_argument('-batch_size',type=int, default=100)
    parser.add_argument('-learning_rate',type=float, default=0.001)

    parser.add_argument('-num_dec_layer',  type=int, default=1)
    parser.add_argument('-num_enc_layer',  type=int, default=1)

    parser.add_argument('-act_hidden', choices = ['linear', 'relu', 'selu','tanh', 'sigmoid'], default='linear')
    parser.add_argument('-act_output', choices = ['linear', 'relu', 'selu','tanh', 'sigmoid'], default='linear')
    parser.add_argument('-use_bn', type=int, default=0)

    parser.add_argument('-noise_sigma',type=float, default=0.000)

    parser.add_argument('-is_bias',  type=int, default=0)


    args = parser.parse_args()
    return args



def build_model(args, H_list):
    if args.is_bias == 1:
        is_bias = True
    else:
        is_bias = False

    def H_channel(x, ai, aj):
        res_list = []
        for idx in range(args.output_block_len):
            HH = K.variable(H_list[ai*args.num_antenna+aj])
            xx = x[:, :, idx]
            tmp = tf.matmul(xx,HH) + args.noise_sigma*tf.random_normal(tf.shape(xx),dtype=tf.float32, mean=0., stddev=1.0)
            res_list.append(tmp)

        res = tf.stack(res_list)
        res = tf.transpose(res, perm=[1, 2, 0])
        return res

    inputs = Input(shape = (args.num_user, args.input_block_len))
    x = inputs

    # current not support multiple layer, try to mimic Brusler&Tse First

    pre_encs = [None for idx in range(args.num_antenna)]
    for a_i in range(args.num_antenna):
        for idx in range(args.num_layer-1):
            x = MultiInputLayer(args.num_hidden_unit, use_bias=is_bias, activation=args.act_output,
                                     name ='S_enc_'+str(a_i)+ '_'+str(idx))(x)

        pre_encs[a_i] = MultiInputLayer(args.output_block_len, use_bias=is_bias, activation=args.act_hidden,
                                 name ='S_enc_'+str(a_i))(x)

    received = [[None for idx in range(args.num_antenna)] for jdx in range(args.num_antenna)]

    # the channel doesn't matter (about the ordering a_j/ a_i)since all channel are different
    for a_i in range(args.num_antenna):
        for a_j in range(args.num_antenna):
            received[a_i][a_j] = Lambda(H_channel,arguments={'ai':a_i, 'aj':a_j},
                name = 'H_Channel_'+str(a_i)+'_'+str(a_j))(pre_encs[a_i])

    dec_data = [None for idx in range(args.num_antenna)]
    # reorder the received
    for a_i in range(args.num_antenna):
        dec_data[a_i] = keras.layers.Add()([received[j][a_i] for j in range(args.num_antenna)])

    dec    = keras.layers.Concatenate(axis = 2)(dec_data)

    for idx in range(args.num_layer-1):
        dec = MultiInputLayer(args.num_hidden_unit, use_bias=is_bias, activation=args.act_hidden,
                                     name ='D_enc_'+str(a_i)+ '_'+str(idx))(dec)

    final  = MultiInputLayer(args.input_block_len, use_bias=is_bias, activation=args.act_output,
                                 name ='D_enc_' + str(args.num_dec_layer))(dec)

    return Model(inputs, final)


def main():

    args = get_args()
    print args

    matrix_number  = args.num_antenna**2  # number of matrixs

    H_list= []
    if args.random_H !='same':
        for idx in range(matrix_number):
            if args.random_H == 'random':
                H_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
            elif args.random_H == 'random_int':
                H_matrix = np.random.randint(0, 2, size = (args.num_user, args.num_user))
            elif args.random_H == 'random_diag':
                v = np.random.normal(0, 1.0, size = (args.num_user,1))
                u = np.random.normal(0, 1.0, size = (args.num_user,1))
                H_matrix = np.diag(np.random.normal(0, 1.0, size = (args.num_user))) + np.dot(v, u.T)

            elif args.random_H == 'zero_diag':
                H_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
                for idx in range(args.num_user):
                    H_matrix[idx][idx] = 0.0

            else:
                H_matrix = np.ones((args.num_user,args.num_user)) - np.eye(args.num_user)

            H_list.append(H_matrix)
    else:
        H_matrix = np.random.normal(0, 1.0, size = (args.num_user, args.num_user))
        for idx in range(matrix_number):
            H_list.append(H_matrix)


    learning_rate = 0.001
    optimizer = Adam(learning_rate)

    model = build_model(args, H_list)

    if args.random_code == 'random_int':
        loss = 'binary_crossentropy'
        model.compile(loss=loss, optimizer=optimizer, metrics=[errors])
    else:
        loss = 'mse'
        model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

    model.summary()

    print 'H',
    print H_list

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

    model.save('./tmp/test_noisy.h5')

if __name__ == '__main__':
    main()
