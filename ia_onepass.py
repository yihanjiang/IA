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

    parser.add_argument('-num_user', type=int, default=3)

    parser.add_argument('-num_block', type=int, default=100000)

    parser.add_argument('-input_block_len', type=int, default=3)
    parser.add_argument('-output_block_len', type=int, default=9)

    parser.add_argument('-random_H', choices = ['same','random', 'random_int', 'random_diag', 'not', 'zero_diag', 'default'], default='same')
    parser.add_argument('-random_code', choices = ['random', 'random_int'], default='random')
    parser.add_argument('-code_symbol', type=int, default=2)   # only valid when random_int is chosen
    parser.add_argument('-num_epoch',type=int, default=400)
    parser.add_argument('-batch_size',type=int, default=100)
    parser.add_argument('-learning_rate',type=float, default=0.001)

    parser.add_argument('-num_dec_layer',  type=int, default=1)
    parser.add_argument('-num_enc_layer',  type=int, default=1)
    parser.add_argument('-num_hidden_unit', type=int, default=500)
    parser.add_argument('-act_hidden', choices = ['linear', 'relu', 'selu','tanh', 'sigmoid'], default='linear')
    parser.add_argument('-act_output', choices = ['linear', 'relu', 'selu','tanh', 'sigmoid'], default='linear')
    parser.add_argument('-use_bn', type=int, default=0)

    parser.add_argument('-noise_sigma',type=float, default=0.000)

    parser.add_argument('-is_bias',  type=int, default=1)


    args = parser.parse_args()
    return args



def build_model(args, H_list):

    def H_channel(x):
        res_list = []
        for idx in range(len(H_list)):
            HH = K.variable(H_list[idx])
            xx = x[:, :, idx]
            tmp = tf.matmul(xx,HH) + args.noise_sigma*tf.random_normal(tf.shape(xx),dtype=tf.float32, mean=0., stddev=1.0)
            res_list.append(tmp)

        res = tf.stack(res_list)
        res = tf.transpose(res, perm=[1, 2,0])
        return res

    inputs = Input(shape = (args.num_user, args.input_block_len))
    x = inputs

    for idx in range(args.num_enc_layer-1):
        x          = MultiInputLayer(args.num_hidden_unit, use_bias=args.is_bias, activation=args.act_hidden,
                                     name ='S_enc_'+str(idx+1))(x)
        if args.use_bn == 1:
            x = BatchNormalization()(x)
    s_sent     = MultiInputLayer(args.output_block_len, use_bias=args.is_bias, activation=args.act_output,
                                 name ='S_enc_'+str(args.num_enc_layer))(x)

    d_received      = Lambda(H_channel, name = '1st_H_Channel')(s_sent)
    x          = d_received

    for jdx in range(args.num_dec_layer-1):
        x          = MultiInputLayer(args.num_hidden_unit, use_bias=args.is_bias, activation=args.act_hidden,
                                     name ='D_enc_'+ str(jdx+1))(x)
        if args.use_bn == 1:
            x = BatchNormalization()(x)
    final      = MultiInputLayer(args.input_block_len, use_bias=args.is_bias, activation=args.act_output,
                                 name ='D_enc_' + str(args.num_dec_layer))(x)

    return Model(inputs, final)


def main():

    args = get_args()
    print args

    num_user = args.num_user
    block_len = args.input_block_len  # input
    code_len  = args.output_block_len  # code

    H_list= []
    if args.random_H !='same':
        for idx in range(code_len):
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

            H_list.append(H_matrix)
    else:
        H_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
        for idx in range(code_len):
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
        message_train  = np.random.randint(0,args.code_symbol, size = (num_block_train, num_user, block_len))
        message_test   = np.random.randint(0,args.code_symbol, size = (num_block_test, num_user, block_len))
    elif args.random_code == 'random':
        message_train = np.random.normal(0, 1, size = (num_block_train, num_user, block_len))
        message_test = np.random.normal(0, 1, size = (num_block_train, num_user, block_len))


    model.fit(message_train, message_train,
              validation_data=(message_test, message_test), batch_size=args.batch_size, epochs=args.num_epoch)

    model.save('./tmp/test_noisy.h5')

if __name__ == '__main__':
    main()
