# BIGD
# Yue Teng
# Version 3.0
# 11/28/2020

import numpy as np
import dippykit as dip
import random as rand
import scipy
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

def sing_bigd(img, scale_size, pair_num, patch_size, samp_factor):
    samp_ver = int((len(img)-patch_size)/2)
    samp_hor = int((len(img[0])-patch_size)/2)
    patch_num = int(samp_ver*samp_hor*samp_factor)
    patches = rand.sample(range(0, samp_ver*samp_hor-1), patch_num)
    BIGDdescriptors = np.empty([patch_num, pair_num*5])
    block_pairs = np.empty((pair_num,4),dtype=int)
    i = 0
    while i < pair_num:
        block_pairs[i] = gen_block_pairs(scale_size, patch_size)
        i += 1
    # print(block_pairs)
    item_count = 0
    for item in patches:
        patch_x = int(item/samp_hor)*2
        patch_y = (item%samp_hor)*2
        patch = img[patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
        pair_count = 0
        while(pair_count < pair_num):
            block1 = patch[block_pairs[pair_count,0]:block_pairs[pair_count,0]+scale_size,block_pairs[pair_count,1]:block_pairs[pair_count,1]+scale_size]
            block2 = patch[block_pairs[pair_count,2]:block_pairs[pair_count,2]+scale_size,block_pairs[pair_count,3]:block_pairs[pair_count,3]+scale_size]
            BIGDdescriptors[item_count,5*pair_count:5*pair_count+5] = bigd_vec(block1)-bigd_vec(block2)
            pair_count += 1
        item_count += 1
    # print(BIGDdescriptors)
    return BIGDdescriptors

def mult_bigd(img, pair, patch_size, samp_factor):
    samp_ver = int((len(img)-patch_size)/2)
    samp_hor = int((len(img[0])-patch_size)/2)
    patch_num = int(samp_ver*samp_hor*samp_factor)
    patches = rand.sample(range(0, samp_ver*samp_hor-1), patch_num)
    pair_num = pair.size
    BIGDdescriptors = np.empty([patch_num, pair_num*5])
    block_pairs = np.empty((pair_num,4),dtype=int)
    i = 0
    while i < pair_num:
        block_pairs[i] = gen_block_pairs(pair[i], patch_size)
        i += 1
    # print(block_pairs)
    item_count = 0
    for item in patches:
        patch_x = int(item/samp_hor)*2
        patch_y = (item%samp_hor)*2
        patch = img[patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
        pair_count = 0
        while(pair_count < pair_num):
            block1 = patch[block_pairs[pair_count,0]:block_pairs[pair_count,0]+pair[pair_count],block_pairs[pair_count,1]:block_pairs[pair_count,1]+pair[pair_count]]
            block2 = patch[block_pairs[pair_count,2]:block_pairs[pair_count,2]+pair[pair_count],block_pairs[pair_count,3]:block_pairs[pair_count,3]+pair[pair_count]]
            BIGDdescriptors[item_count,5*pair_count:5*pair_count+5] = bigd_vec(block1)-bigd_vec(block2)
            pair_count += 1
        item_count += 1
    # print(BIGDdescriptors)
    return BIGDdescriptors

def gen_block_pairs(scale_size, patch_size):
    samp = patch_size - scale_size;
    blocks = rand.sample(range(0, samp**2-1), 2)
    block_pairs = np.zeros((1,4),dtype=int)
    block_pairs[0,0] = int(blocks[0]/samp) # x coord of block 1
    block_pairs[0,1] = blocks[0]%samp # y coord of block 1
    block_pairs[0,2] = int(blocks[1]/samp) # x coord of block 2
    block_pairs[0,3] = blocks[1]%samp # y coord of block 2
    return block_pairs

def bigd_vec(block):
    vec = np.zeros(5)
    vec[0] = np.average(block)
    dx = scipy.ndimage.sobel(block, 0)
    dy = scipy.ndimage.sobel(block, 1)
    vec[1] = np.average(dx)
    vec[2] = np.average(dy)
    vec[3] = np.average(np.abs(dx))
    vec[4] = np.average(np.abs(dy))
    return vec

def vlad_encoder(BIGDdescriptors,k):
    kmeans = KMeans(n_clusters=k).fit(BIGDdescriptors)
    encoded_descriptor = np.zeros(len(BIGDdescriptors[0])*k)
    i = 0
    for descriptor in BIGDdescriptors:
        label = kmeans.labels_[i]
        encoded_descriptor[label*len(BIGDdescriptors[0]):(label+1)*len(BIGDdescriptors[0])] += (kmeans.cluster_centers_[label]-descriptor)
        i += 1
    # print(encoded_descriptor)
    return encoded_descriptor

def calc_correct(result, correct_label):
    i = 0
    correct = 0
    while i < correct_label.size:
        if result[i] == correct_label[i]:
            correct += 1
        i += 1
    return correct/correct_label.size
    

k = 64
patch_size = 9
block_pairs = 8
i = 0
training_vlad = np.empty((80,5*block_pairs*k))
pair = np.array([2,2,3,3,4,4])
correct_label = np.ones(16)*2
correct_label = np.concatenate((correct_label, np.ones(16)*3))
correct_label = np.concatenate((correct_label, np.ones(16)*1))
correct_label = np.concatenate((correct_label, np.zeros(16)))

while i < 3:
    patch_size = 9
    while patch_size < 16:
        block_size = 2
        while block_size < 6:
            directory = r'D:/GT/FALL2020/IMG/Project/Train/Class0'
            count0 = 0
            for filename in os.listdir(directory):
                img = dip.im_read('D:/GT/FALL2020/IMG/Project/Train/Class0/'+filename)
                bigd = sing_bigd(img, block_size, block_pairs, patch_size, 0.05)
                training_vlad[count0] = vlad_encoder(bigd,k)
                # print(training_vlad[count0])
                count0 += 1
            label = np.zeros(count0)

            directory = r'D:/GT/FALL2020/IMG/Project/Train/Class1'
            count1 = 0
            for filename in os.listdir(directory):
                img = dip.im_read('D:/GT/FALL2020/IMG/Project/Train/Class1/'+filename)
                bigd = sing_bigd(img, block_size, block_pairs, patch_size, 0.05)
                training_vlad[count0+count1] = vlad_encoder(bigd,k)
                # print(training_vlad[count0+count1])
                count1 += 1
            label = np.concatenate((label, np.ones(count1)))

            directory = r'D:/GT/FALL2020/IMG/Project/Train/Class2'
            count2 = 0
            for filename in os.listdir(directory):
                img = dip.im_read('D:/GT/FALL2020/IMG/Project/Train/Class2/'+filename)
                bigd = sing_bigd(img, block_size, block_pairs, patch_size, 0.05)
                training_vlad[count0+count1+count2] = vlad_encoder(bigd,k)
                # print(training_vlad[count0+count1+count2])
                count2 += 1
            label = np.concatenate((label, np.ones(count2)*2))

            directory = r'D:/GT/FALL2020/IMG/Project/Train/Class3'
            count3 = 0
            for filename in os.listdir(directory):
                img = dip.im_read('D:/GT/FALL2020/IMG/Project/Train/Class3/'+filename)
                bigd = sing_bigd(img, block_size, block_pairs, patch_size, 0.05)
                training_vlad[count0+count1+count2+count3] = vlad_encoder(bigd,k)
                # print(training_vlad[count0+count1+count2+count3])
                count3 += 1
            label = np.concatenate((label, np.ones(count3)*3))

            # print(label)
            clf = make_pipeline(StandardScaler(), svm.SVC())
            clf.fit(training_vlad, label)

            directory = r'D:/GT/FALL2020/IMG/Project/Test'
            test_im = 0
            result = np.empty(64)
            for filename in os.listdir(directory):
                img = dip.im_read('D:/GT/FALL2020/IMG/Project/Test/'+filename)
                bigd = sing_bigd(img, block_size, block_pairs, patch_size, 0.05)
                result[test_im] = clf.predict([vlad_encoder(bigd,k)])
                test_im += 1
            
            print('block size = ', block_size)
            print('patch size = ', patch_size)
            print(calc_correct(result, correct_label))

            block_size += 1
        patch_size += 2
    i += 1
