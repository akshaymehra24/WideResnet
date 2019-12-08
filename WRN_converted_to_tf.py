# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:16:02 2019

@author: Akshay Mehra
"""
# Reaches 95.89 in 130 epochs, might be a little slow initially but picks up eventually
import tensorflow as tf
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import config as cf
import keras
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Training parameters
batch_size = 100  
num_classes = 10
adam = False

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean["cifar10"], cf.std["cifar10"]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean["cifar10"], cf.std["cifar10"]),
])

print("| Preparing CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
num_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session
sess = tf.Session(config=config)
print("Created TensorFlow session.")

# Define input TF placeholder
x_tf = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y_tf = tf.placeholder(tf.float32, shape=(None, 10))
lr_tf = tf.placeholder(tf.float32,[],'lr_tf') 


istraining_ph = tf.placeholder_with_default(True,shape=())
def wide_basic(inputs, in_planes, out_planes, dropout_rate, stride, name, index):
    with tf.variable_scope('wide_basic' + name):
        if stride != 1 or in_planes != out_planes:
            skip_c = tf.layers.conv2d(inputs, out_planes, kernel_size=1, strides=stride, use_bias = True, padding='SAME')
        else:
            skip_c = inputs
        
        x = tf.contrib.layers.batch_norm(inputs, updates_collections=None, decay=0.9, scale=True, center=True, is_training = istraining_ph)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, out_planes, kernel_size=3, strides=1, use_bias = True, padding='SAME')
        x = tf.layers.dropout(x, rate=0.1, training=istraining_ph)
        x = tf.contrib.layers.batch_norm(x, updates_collections=None, decay=0.9, scale=True, center=True, is_training = istraining_ph)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, out_planes, kernel_size=3, strides=stride, use_bias = True, padding='SAME')
        
        #print("skip:", skip_c.shape)
        #print("x:", x.shape)
        x = tf.add(skip_c, x)
        
    return x

def wide_layer(out, in_planes, out_planes, num_blocks, dropout_rate, stride, name):
    with tf.variable_scope('wide_layer' + name):
        strides = [stride] + [1]*int(num_blocks-1)
        #print("strides:", strides)
        i = 0
        for strid in strides:
            #print("i:", i)
            out = wide_basic(out, in_planes, out_planes, dropout_rate, strid, name = 'layer1_'+str(i)+'_', index = i)
            in_planes = out_planes
            i += 1
        
    return out
        
def make_resnet_filter(ins, depth=28, widen_factor=10, dropout_rate = 0.3, reuse=False):
    n = (depth-4)/6
    k = widen_factor
    print('| Wide-Resnet %dx%d' %(depth, k))
    nStages = [16, 16*k, 32*k, 64*k]
    with tf.variable_scope('WRN', reuse=reuse):
        x = tf.layers.conv2d(ins, nStages[0], kernel_size=3, strides=1, use_bias = True, padding='SAME')
        x = wide_layer(x, nStages[0], nStages[1], n, dropout_rate, stride=1, name = 'layer1_')
        x = wide_layer(x, nStages[1], nStages[2], n, dropout_rate, stride=2, name = 'layer2_')
        x = wide_layer(x, nStages[2], nStages[3], n, dropout_rate, stride=2, name = 'layer3_')
        x = tf.contrib.layers.batch_norm(x, updates_collections=None, decay=0.9, scale=True, center=True, is_training = istraining_ph)
        x = tf.nn.relu(x)
        x = tf.keras.layers.AvgPool2D([8,8])(x)
        x = tf.reshape(x, (-1, 640))
        x = tf.layers.dense(x, num_classes)
        return x
    
with tf.variable_scope('WRN', reuse=False):
    logits_train = make_resnet_filter(x_tf, depth=28, widen_factor=10, dropout_rate = 0.3, reuse=False)
    
with tf.variable_scope('WRN', reuse=False):
    logits_test = make_resnet_filter(x_tf, depth=28, widen_factor=10, dropout_rate = 0.3, reuse=True)
    
var_filt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='WRN')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits_train, labels=y_tf)) + 5E-4 * tf.add_n([tf.nn.l2_loss(v) for v in var_filt])

optimizer_min = tf.train.MomentumOptimizer(learning_rate = lr_tf, momentum = 0.9)
optim_min = optimizer_min.minimize(loss, var_list=var_filt)

prediction = tf.nn.softmax(logits_test)

sess.run(tf.global_variables_initializer())

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)
    
epochs = 200
for ep in range(1, epochs): 
    
    lr_mi = learning_rate(0.1, ep)
        
    batch = 0
    for xs, ys in trainloader:
        feed_dict_min = {x_tf:xs.permute(0, 2, 3, 1), y_tf:keras.utils.to_categorical(ys, 10), lr_tf: lr_mi}
        for min_i in range(1):
            sess.run(optim_min, feed_dict = feed_dict_min)
        batch += 1
        if batch%100 == 0:
            print(batch)
    
    if ep%1 == 0:          
        print("Testing", ep, adam)
        acc = 0
        num = 0
        for xs, ys in testloader:
            feed_dict_test = {x_tf:xs.permute(0, 2, 3, 1), istraining_ph:False}
            pred = sess.run(prediction, feed_dict_test)
            acc += np.sum(np.argmax(pred, 1)==np.argmax(keras.utils.to_categorical(ys, 10), 1))
            num += 1

        acc /= float(num*batch_size)
        print('mean acc = %f\n\n'%(acc))
