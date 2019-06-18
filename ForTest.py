# desktop=r'c:/users/yelei/desktop'
# import os
# background_image_filename =os.path.join(desktop,'back.jpg')
# mouse_image_filename =os.path.join(desktop,'fulu.png')
# # 指定图像文件名称
#
# import pygame
# # 导入pygame库
# from pygame.locals import *
# # 导入一些常用的函数和常量
# from sys import exit
#
# # 向sys模块借一个exit函数用来退出程序
#
# pygame.init()
# # 初始化pygame,为使用硬件做准备
#
# screen = pygame.display.set_mode((640, 480), 0, 0)
# # 创建了一个窗口
# pygame.display.set_caption("Hello, World!")
# # 设置窗口标题
#
# background = pygame.image.load(background_image_filename)  ##.convert()
# mouse_cursor = pygame.image.load(mouse_image_filename).convert_alpha()
# # 加载并转换图像
#
# while True:
#     # 游戏主循环
#
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             # 接收到退出事件后退出程序
#             exit()
#
#     screen.blit(background, (0, 0))
#     # 将背景图画上去
#
#     x, y = pygame.mouse.get_pos()
#     # 获得鼠标位置
#     a=mouse_cursor.get_width()
#     b=mouse_cursor.get_height()
#
#     x -= mouse_cursor.get_width() / 2
#     y -= mouse_cursor.get_height() / 2
#     # 计算光标的左上角位置
#     screen.blit(mouse_cursor, (x, y))
#     # 把光标画上去
#
#     pygame.display.update()
#     # 刷新一下画面
#
#
#
# # import pygame
# # if pygame.font is None:
# #     print("the font module is not available")
# #     exit()
# for i in range(1000):
#     print(str(i).zfill(5))

# import os
# import shutil
# baseDir=r'C:\Users\yelei\Desktop\baseDir'
# newbaseDir=r'C:\Users\yelei\Desktop\newbaseDir'
# fileIndex=0
# fileDirs=os.listdir(baseDir)
# for fileDir in fileDirs:
#     fileFullDir=os.path.join(baseDir,fileDir)
#     filenames=os.listdir(fileFullDir)
#     for filename in filenames:
#         filefullname=os.path.join(fileFullDir,filename)
#         exName=os.path.splitext(filefullname)[-1]
#         exNameUpper=str(exName).upper()
#         if exNameUpper in ['.JPG','.PNG']:
#             if not os.path.exists(newbaseDir):
#                 os.makedirs(newbaseDir)
#             newfilefullname=os.path.join(newbaseDir,'origin_{0}{1}'.format(str(fileIndex).zfill(7),exName))
#             os.rename(filefullname,newfilefullname)
#             fileIndex+=1


#!/usr/bin/env python


import tensorflow as tf
import cv2
import sys
import FlappyBird.flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0050 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LR=0.01

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)  ## 全连接层 1600*512

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2   ##全连接层 512*2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    ##tf.matmul 是两个矩阵的矩阵乘
    ##tf.multiply是两个矩阵对应元素的相乘
    ##这里的tf.reduce_sum(,reduction_indices=?)
    ##[[1,1,1] , [1,1,1]]
    ##默认不写：总体求和6
    ##0 按照列求和 [2,2,2]
    ##1 按照行求和[ 3,3]
    ## keep_dims=True 是按照行的维度求和[[3],[3]]
    ##[0,1]  行列求和 和第一个一样 6
    cost = tf.reduce_mean(tf.square(y - readout_action))
    ##tf.square  举证每个元素平方

    ##2019/04/26  Insert 指数衰减法 学习率lr
    ##Failed Delete
    # global_step=tf.Variable(0,trainable=False)
    # decay_lr=tf.train.exponential_decay(LR,global_step=global_step,decay_steps=10000,decay_rate=0.9,staircase=True)

    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)  ##2019/04/26 Update  1e-6 ->decay_lr

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    # a_file = open("logs_" + GAME + "/readout.txt", 'w')
    # h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()





