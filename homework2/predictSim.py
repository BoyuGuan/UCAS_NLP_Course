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
    logDir = './log/2022-10-27-16-58-47'
    with open(os.path.join(logDir, 'vocab_idx2token.pkl') , 'rb') as f:
        idx_to_token = pickle.load(f)
    with open(os.path.join(logDir, 'vocab_token2idx.pkl'), 'rb') as f:
        token_to_idx = pickle.load(f)
    batch_size, max_window_size, num_noise_words = 2048, 5, 5
    # data_iter, vocab = makeDataset.load_data_loader(batch_size,\
    #     max_window_size,num_noise_words, './data/renmin_tiny.txt')
    # print(len(vocab))
    embed = train.getNet(20875)
    embed.load_state_dict(torch.load(os.path.join(logDir ,'net.pt')))
    embed = embed[0]


    get_similar_tokens('欧洲', 3, embed, token_to_idx, idx_to_token)
