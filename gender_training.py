#!/usr/bin/env python3
#coding=utf-8

#===========================================#
#Program:数据训练
#Data:2019.4.30
#Author:liheng
#Version:V1.0
#===========================================#

import tensorflow as tf
import gender_train_data as train_data
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

train_epochs=3000#迭代(训练)次数
batch_size = 9#批大小
drop_prob = 0.4#丢弃比例。用于正则化
learning_rate=0.00001#学习率


#======================辅助函数  start================================================#
#权重初始化(卷积核初始化)
#参数shpae为一个列表对象,例如[5, 5, 1, 32]对应
#5,5 表示卷积核的大小, 1代表通道channel,对彩色图片做卷积是3,单色灰度为1
#最后一个数字32,卷积核的个数,(也就是卷基层提取的特征数量)
def weight_init(shape):
    weight = tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.Variable(weight)

#偏置初始化
def bias_init(shape):
    bias = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(bias)

#全连接矩阵初始化
def fch_init(layer1,layer2,const=1):
    min = -const * (6.0 / (layer1 + layer2))
    max = -min
    weight = tf.random_uniform([layer1, layer2], minval=min, maxval=max, dtype=tf.float32)
    return tf.Variable(weight)

# tf.nn.conv2d()是一个二维卷积函数,
# stirdes 是卷积核移动的步长,4个1表示,在x张量维度的四个参数上移动步长
# padding 参数'SAME',表示对原始输入像素进行填充,保证卷积后映射的2D图像与原图大小相等
# 填充,是指在原图像素值矩阵周围填充0像素点
# 如果不进行填充,假设 原图为 32x32 的图像,卷积和大小为 5x5 ,卷积后映射图像大小 为 28x28
def conv2d(images,weight):
    return tf.nn.conv2d(images,weight,strides=[1,1,1,1],padding='SAME')

#最大值池化
#卷积核在提取特征时的动作成为padding，
#它有两种方式：SAME和VALID。
#卷积核的移动步长不一定能够整除图片像素的宽度，
# 所以在有些图片的边框位置有些像素不能被卷积。
# 这种不越过边缘的取样就叫做 valid padding，卷积后的图像面积小于原图像。
# 为了让卷积核覆盖到所有的像素，可以对边缘位置进行0像素填充，然后在进行卷积。
# 这种越过边缘的取样是 same padding。
# 如过移动步长为1，那么得到和原图一样大小的图像。
# 如果步长很大，超过了卷积核长度，那么same padding，得到的特征图也会小于原来的图像。
# 池化跟卷积的情况有点类似
# images 是卷积后,有经过非线性激活后的图像,
# ksize 是池化滑动张量
# ksize 的维度[batch, height, width, channels],跟 images 张量相同
# strides [1, 2, 2, 1],与上面对应维度的移动步长
# padding与卷积函数相同,padding='VALID',对原图像不进行0填充
def max_pool2x2(images,tname):
    return tf.nn.max_pool(images,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=tname)

#======================辅助函数  end================================================#


#images_input 为输入的图片，labels_input为输入的标签
#占位符
images_input = tf.placeholder(tf.float32,[None,112*92*3],name='input_images')
labels_input = tf.placeholder(tf.float32,[None,2],name='input_labels')
#把图像转换为112*92*3的形状
x_input = tf.reshape(images_input,[-1,112,92,3])


#===================================训练过程，神经网络结构定义==================#

#输入图像尺寸:112*92*3

# 卷积核3*3*3 16个     第一层卷积+池化
w1 = weight_init([3,3,3,16])
b1 = bias_init([16])
# 结果 NHWC  N H W C
conv_1 = conv2d(x_input,w1)+b1
relu_1 = tf.nn.relu(conv_1,name='relu_1')
max_pool_1 = max_pool2x2(relu_1,'max_pool_1')
#第一层卷积+池化后输出:(56*46)*16


# 卷积核3*3*16  32个  第二层卷积
w2 = weight_init([3,3,16,32])
b2 = bias_init([32])
conv_2 = conv2d(max_pool_1,w2) + b2
relu_2 = tf.nn.relu(conv_2,name='relu_2')
max_pool_2 = max_pool2x2(relu_2,'max_pool_2')
#第二层卷积+池化后输出:(28*23)*32

#卷积核3*3*32 64个 第三层卷积
w3 = weight_init([3,3,32,64])
b3 = bias_init([64])
conv_3 = conv2d(max_pool_2,w3)+b3
relu_3 = tf.nn.relu(conv_3,name='relu_3')
max_pool_3 = max_pool2x2(relu_3,'max_pool_3')
#第三层层卷积+池化后输出:(14*12)*64
#其到全连接层的大小为:14*12*64=10752




#把第三层的卷积结果平铺成一维向量
f_input = tf.reshape(max_pool_3,[-1,14*12*64])

#全连接第一层 14*12*64,512
f_w1= fch_init(14*12*64,512)
f_b1 = bias_init([512])
f_r1 = tf.matmul(f_input,f_w1) + f_b1
f_relu_r1 = tf.nn.relu(f_r1)#激活函数，relu随机丢掉一些权重提供泛华能力
# 为了防止网络出现过拟合的情况,对全连接隐藏层进行 Dropout(正则化)处理,在训练过程中随机的丢弃部分
# 节点的数据来防止过拟合.Dropout同把节点数据设置为0来丢弃一些特征值,仅在训练过程中,
# 预测的时候,仍使用全数据特征
# 传入丢弃节点数据的比例
f_dropout_r1 = tf.nn.dropout(f_relu_r1,drop_prob)


#全连接第二层512*128
f_w2 = fch_init(512,128)
f_b2 = bias_init([128])
f_r2 = tf.matmul(f_dropout_r1,f_w2) + f_b2
f_relu_r2 = tf.nn.relu(f_r2)
f_dropout_r2 = tf.nn.dropout(f_relu_r2,drop_prob)


#全连接第三层 128*2
f_w3 = fch_init(128,2)
f_b3 = bias_init([2])
f_r3 = tf.matmul(f_dropout_r2,f_w3) + f_b3#未激活的输出
#最终的输出结果[0,1]之间
f_softmax = tf.nn.softmax(f_r3,name='f_softmax')

#损失函数
#定义交叉熵
cross_entry =  tf.reduce_mean(tf.reduce_sum(-labels_input*tf.log(f_softmax)))
#优化器，自动执行梯度下降算法
optimizer  = tf.train.AdamOptimizer(learning_rate).minimize(cross_entry)

#计算准确率&损失
arg1 = tf.argmax(labels_input,1)
arg2 = tf.argmax(f_softmax,1)
cos = tf.equal(arg1,arg2)
acc = tf.reduce_mean(tf.cast(cos,dtype=tf.float32))

#变量初始化
init = tf.global_variables_initializer()
#启动会话，然后开始训练


with tf.Session() as sess:
    sess.run(init)

    Cost = []
    Accuracy = []
    for i in range(train_epochs):
        idx = random.randint(0, len(train_data.images) - 20)
        batch = random.randint(6, 18)
        train_input = train_data.images[idx:(idx + batch)]
        train_labels = train_data.labels[idx:(idx + batch)]
        result, acc1, cross_entry_r, cos1, f_softmax1, relu_1_r = sess.run(
            [optimizer, acc, cross_entry, cos, f_softmax, relu_1],
            feed_dict={images_input: train_input, labels_input: train_labels})

        #Cost.append(cross_entry_r)
        #Accuracy.append(acc1)
        if i % 100 == 0:
            Cost.append(cross_entry_r)
            Accuracy.append(acc1)
            print('Epoch : %d ,  Cost : %.7f , Accuracy: %.7f' % (i + 1, cross_entry_r, acc1))

    # 保存模型
    if not os.path.exists('model'):
        os.mkdir('model')
    saver = tf.train.Saver()#创建Saver对象。如果
    saver.save(sess, './model/my-gender-v1.0')
    #其中tf.train.Saver()创建Saver对象。在创建这个Saver对象的时候，
    # 有一个参数我们经常会用到，就是 max_to_keep 参数，
    # 这个是用来设置保存模型的个数，默认为5，即 max_to_keep=5，保存最近的5个模型。
    # 如果你想每训练一代（epoch)就想保存一次模型，则可以将 max_to_keep设置为None或者0，如：
    #saver = tf.train.Saver(max_to_keep=0)
    #但是这样做除了多占用硬盘，并没有实际多大的用处，因此不推荐。
    #当然，如果你只想保存最后一代的模型，则只需要将max_to_keep设置为1即可
    #save()函数：第一个参数sess, 这个就不用说了。第二个参数设定保存的路径和名字，第三个参数将训练的次数作为后缀加入到模型名字中。
    #saver.save(sess, 'my-model', global_step=0) == > filename: 'my-model-0'
    #...
    #saver.save(sess, 'my-model', global_step=1000) == > filename: 'my-model-1000'
    #在实验中，最后一代可能并不是验证精度最高的一代，因此我们并不想默认保存最后一代，而是想保存验证精度最高的一代，
    # 则加个中间变量和判断语句就可以了。


    # 测试集验证
    arg2_r = sess.run(arg2, feed_dict={images_input: train_data.test_images, labels_input: train_data.test_labels})
    arg1_r = sess.run(arg1, feed_dict={images_input: train_data.test_images, labels_input: train_data.test_labels})
    print(classification_report(arg1_r, arg2_r))
    print('training finished')

    # 代价函数曲线
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    plt.plot(Cost)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cost')
    plt.title('Cross Loss')
    plt.grid()
    plt.show()

    # 准确率曲线
    fig7, ax7 = plt.subplots(figsize=(10, 7))
    plt.plot(Accuracy)
    ax7.set_xlabel('Epochs')
    ax7.set_ylabel('Accuracy Rate')
    plt.title('Train Accuracy Rate')
    plt.grid()
    plt.show()









