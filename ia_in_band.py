__author__ = 'yihanjiang'
import numpy as np
import tensorflow as tf
import keras
import math
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization, Lambda
from keras.models import Sequential, Model
from keras.constraints import Constraint
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler

from utils import MultiInputLayer, errors,stack


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_user', type=int, default=3)
    parser.add_argument('-input_block_len', type=int, default=1)
    parser.add_argument('-output_block_len', type=int, default=1)

    parser.add_argument('-random_H', choices = ['random', 'random_int', 'random_diag', 'not', 'zero_diag'], default='random')
    parser.add_argument('-random_U', choices = ['random', 'random_int', 'random_diag', 'not', 'zero_diag'], default='random')
    parser.add_argument('-random_W', choices = ['random', 'random_int', 'random_diag', 'not', 'zero_diag'], default='random')
    parser.add_argument('-G_mode', choices = ['random', 'H.T'], default='H.T')
    parser.add_argument('-random_code', choices = ['random', 'random_int'], default='random_int')
    parser.add_argument('-code_symbol', type=int, default=2)   # only valid when random_int is chosen
    parser.add_argument('-num_epoch',type=int, default=400)

    parser.add_argument('-noise_sigma',type=float, default=0.000)

    parser.add_argument('-num_layer', type=int, default=1)
    parser.add_argument('-pre_code', type=int, default=1)
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

    # def H_channel(x):
    #     HH = K.variable(H)
    #     return K.dot(x, HH) + args.noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)

    # def U_channel(x):
    #     UU = K.variable(U)
    #     return K.dot(x, UU) + args.noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)
    #
    # def W_channel(x):
    #     WW = K.variable(W)
    #     return K.dot(x, WW) + args.noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)


    act = 'linear'
    output_act = 'linear'

    inputs = Input(shape = (args.num_user, args.input_block_len))
    x = inputs

    # 1st Forward

    s_output = MultiInputLayer(args.output_block_len, activation=output_act, name ='S_enc_1')(x)

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

    learning_rate = args.learning_rate
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
