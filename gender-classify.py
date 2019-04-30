#!/usr/bin/env python3
#coding=utf-8

#===========================================#
#Program:使用保存好的模型进行识别
#Data:2019.4.30
#Author:liheng
#Version:V1.0
#===========================================#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import gender_train_data as train_data

np.set_printoptions(suppress=True)


if __name__ == '__main__':
    # 取一张图片及其对应标签
    input_image = train_data.images[0:1]
    input_label = train_data.labels[0:1]
    # 显示图片
    fig, axis = plt.subplots(figsize=(2, 2))
    axis.imshow(np.reshape(input_image, (112, 92, 3)))
    plt.axis('off')  # 不显示坐标轴
    plt.title("input image")
    plt.show()

    #加载模型
    with tf.Session() as sess:
        graph_path = os.path.abspath('./model/my-gender-v1.0.meta')
        model = os.path.abspath('./model/')

        server = tf.train.import_meta_graph(graph_path)
        server.restore(sess, tf.train.latest_checkpoint(model))  # 恢复模型，latest_checkpoint函数自动获取最后一次保存的模型

        graph = tf.get_default_graph()

        # 填充feed_dict---输入
        x = graph.get_tensor_by_name('input_images:0')
        feed_dict = {x: input_image}

        #全连接最后一层---输出
        f_softmax = graph.get_tensor_by_name('f_softmax:0')

        res = sess.run(f_softmax, feed_dict)
        print(res)
