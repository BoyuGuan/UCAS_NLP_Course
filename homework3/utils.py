'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-11-09 21:39:41
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-11-17 11:34:28
FilePath: /guan/ucas/nlp/homework3/utils.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import os
import pickle
import torch

# config for training
class Config():

    def __init__(self):

        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.epochNum = 500
        self.data_dir = './data/'
        self.savePath = './toUse/' # 存放词向量等中间件
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.save_model = 'NERmodel.pth'
        self.batch_size = 1

def build_dict():
    tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7, "START_TAG": 8, "STOP_TAG": 9}
    with open(os.path.join('./word2vec', 'vocab_word2idx.pkl'), 'rb') as f:
        word2id = pickle.load(f)
    return word2id, tag2id

def cal_max_length(data_dir):
    file = data_dir + 'train' + '_corpus.txt'
    lines = open(file).readlines()
    max_len = 0
    for line in lines:
        if len(line.split()) > max_len :
            max_len = len(line.split())
    return max_len

def log_sum_exp(vec):
    max_score = vec[0, torch.argmax(vec, 1)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))