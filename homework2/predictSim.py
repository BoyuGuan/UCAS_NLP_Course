'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-27 16:33:07
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-11-02 19:17:12
FilePath: /guan/ucas/nlp/homework2/predictSim.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
from cmath import log
import pickle
import os

import torch

import train
import makeDataset

def get_similar_tokens(query_token, k, embed, token_to_idx, idx_to_token):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {idx_to_token[i]}')

if __name__ == '__main__':
    en, zh = 0, 1
    logDir = ['./log/cnn', './log/renmin'][zh]
    vocabLen = [43638, 49324][zh]
    with open(os.path.join(logDir, 'vocab_idx2token.pkl') , 'rb') as f:
        idx_to_token = pickle.load(f)
    with open(os.path.join(logDir, 'vocab_token2idx.pkl'), 'rb') as f:
        token_to_idx = pickle.load(f)
    embed = train.getNet(vocabLen)
    embed.load_state_dict(torch.load(os.path.join(logDir ,'net.pt')))
    embed = embed[0]

    toFind = input("Please input the word you want to find similar words in word2vec\n")
    print("10 most similar words in word2vec is:")
    get_similar_tokens(toFind, 10, embed, token_to_idx, idx_to_token)
