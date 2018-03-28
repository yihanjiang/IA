__author__ = 'yihanjiang'
import numpy as np
import argparse
import tensorflow as tf
import keras
import math
from keras import backend as K

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, Conv1D
from keras.layers import BatchNormalization, Lambda
from keras.models import Sequential, Model
from keras.constraints import Constraint
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler

from utils import MultiInputLayer, errors, stack

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_user', type=int, default=3)
    parser.add_argument('-num_block', type=int, default=10000)

    parser.add_argument('-input_block_len', type=int, default=4)
    parser.add_argument('-output_block_len', type=int, default=4)

    parser.add_argument('-random_H', choices = ['random', 'random_int', 'random_diag', 'random_same',
                                                'not', 'zero_diag', 'default'], default='random')
    parser.add_argument('-G_mode', choices = ['random', 'H.T'], default='H.T')
    parser.add_argument('-random_code', choices = ['random', 'random_int'], default='random')
    parser.add_argument('-code_symbol', type=int, default=2)   # only valid when random_int is chosen
    parser.add_argument('-num_epoch',type=int, default=400)

    parser.add_argument('-batch_size',type=int, default=100)

    parser.add_argument('-act_hidden', choices = ['linear', 'relu','elu', 'tanh', 'sigmoid'], default='linear')
    parser.add_argument('-act_output', choices = ['linear', 'relu','elu', 'tanh', 'sigmoid'], default='linear')

    parser.add_argument('-learning_rate',type=float, default=0.001)
    parser.add_argument('-noise_sigma',type=float, default=0.000)
    parser.add_argument('-signal_sigma',type=float, default=1.0)

    parser.add_argument('-num_layer', type=int, default=1)
    parser.add_argument('-pre_code', type=int, default=1)
    parser.add_argument('-is_bias',  type=int, default=1)

    args = parser.parse_args()
    return args

def build_model(args, H_list, G_list):

    noise_sigma = args.noise_sigma
    act         = args.act_hidden
    output_act  = args.act_output
    use_bias      =  args.is_bias
    is_pre_coding = args.pre_code

    def H_channel(x):
        res_list = []
        for idx in range(len(H_list)):
            HH = K.variable(H_list[idx])
            xx = x[:, :, idx]
            tmp = tf.matmul(xx,HH)  +  noise_sigma*tf.random_normal(tf.shape(xx),dtype=tf.float32, mean=0., stddev=1.0)
            res_list.append(tmp)

        res = tf.stack(res_list)
        res = tf.transpose(res, perm=[1, 2,0])
        return res

    def G_channel(x):
        res_list = []
        for idx in range(len(G_list)):
            GG = K.variable(G_list[idx])
            xx = x[:, :, idx]
            tmp = tf.matmul(xx,GG)  +  noise_sigma*tf.random_normal(tf.shape(xx),dtype=tf.float32, mean=0., stddev=1.0)
            res_list.append(tmp)

        res = tf.stack(res_list)
        res = tf.transpose(res, perm=[1, 2,0])
        return res


    inputs = Input(shape = (args.num_user, args.input_block_len))

    # 1st Forward
    if is_pre_coding:
        s_sent          = MultiInputLayer(args.output_block_len, use_bias=use_bias, activation=output_act,  name ='S_enc_1')(inputs)
    else:
        s_sent = inputs

    d_received      = Lambda(H_channel, name = '1st_H_Channel')(s_sent)

    d_sent          = MultiInputLayer(args.output_block_len, use_bias=use_bias, activation=act, name ='D_enc_1')(d_received)

    s_received      = Lambda(G_channel, name = '1st_G_Channel')(d_sent)

    inputs_with_received = keras.layers.Concatenate(axis=2)([inputs,s_received ])
    s_sent_2             = inputs_with_received
    for layer in range(1, args.num_layer):
        s_sent_2             = MultiInputLayer(args.output_block_len, use_bias=use_bias, activation=act, name ='S_enc_2_'+str(layer))(s_sent_2)
    s_sent_2    = MultiInputLayer(args.output_block_len, use_bias=use_bias, activation=output_act, name ='S_enc_2_'+str(args.num_layer))(s_sent_2)

    d_received_2         = Lambda(H_channel, name = '2nd_H_Channel')(s_sent_2)

    d_combine            = keras.layers.Concatenate(axis=2)([d_received,d_received_2 ])
    final                = d_combine
    for layer in range(1, args.num_layer):
        final = MultiInputLayer(args.output_block_len, use_bias=use_bias, activation=act, name ='D_enc_2_'+str(layer))(final)
    final = MultiInputLayer(args.input_block_len, use_bias=use_bias, activation=output_act, name ='D_enc_2_'+str(args.num_layer))(final)

    return Model(inputs, final)


def main():
    args = get_args()
    print args

    num_user = args.num_user
    block_len = args.input_block_len  # input
    code_len  = args.output_block_len  # code

    H_list, G_list = [], []

    if args.random_H == 'random_same':
        H_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
        if args.G_mode == 'random':
            G_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
        else:
            G_matrix = np.transpose(H_matrix)
        for idx in range(code_len):
            H_list.append(H_matrix)
            G_list.append(G_matrix)
    else:
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

            if args.G_mode == 'random':
                G_matrix = np.random.normal(0, 1.0, size = (num_user, num_user))
            else:
                G_matrix = np.transpose(H_matrix)

            H_list.append(H_matrix)
            G_list.append(G_matrix)

    learning_rate = args.learning_rate
    optimizer = Adam(learning_rate)

    model = build_model(args,  H_list = H_list, G_list=G_list)

    if args.random_code == 'random_int':
        loss = 'binary_crossentropy'
        model.compile(loss=loss, optimizer=optimizer, metrics=[errors])
    else:
        loss = 'mse'
        model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

    model.summary()

    print 'H',
    print H_list
    print 'G',
    print G_list


    num_block_train = args.num_block
    num_block_test  = args.num_block/10

    if args.random_code == 'random_int':
        message_train  = np.random.randint(0,args.code_symbol, size = (num_block_train, num_user, block_len))
        message_test   = np.random.randint(0,args.code_symbol, size = (num_block_test, num_user, block_len))
    elif args.random_code == 'random':
        message_train = np.random.normal(0, args.signal_sigma, size = (num_block_train, num_user, block_len))
        message_test = np.random.normal(0, args.signal_sigma, size = (num_block_train, num_user, block_len))

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
              validation_data=(message_test, message_test), batch_size=args.batch_size, epochs=args.num_epoch)

    model.save('./tmp/test_noisy.h5')

if __name__ == '__main__':
    main()
