'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-24 22:32:11
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-10-26 23:45:15
FilePath: /guan/ucas/nlp/homework2/train.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import math
import torch
from torch import nn
from d2l import torch as d2l

import makeDataset



def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


def getNet(embed_size = 100):
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                    embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                    embedding_dim=embed_size))
    return net

def train(net, data_iter, lr, num_epochs, device='cuda'):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    loss = SigmoidBCELoss()
    net = net.to(device)
    minLoss = 1666666
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 规范化的损失之和，规范化的损失数
    for epoch in range(num_epochs):
        print(f'The {epoch + 1}th epoch')
        num_batches = len(data_iter)
        epochLoss = 0
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l = l.sum()
            epochLoss += l / num_batches
            if i % 10 == 0:
                print(f'The loss is {l}')
            l.backward()
            optimizer.step()
        if epochLoss < minLoss:
            minLoss = epochLoss
            print(f'min loss is {minLoss}')
            torch.save(net.state_dict(), './log/net.pt')


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

if __name__ == '__main__':
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = makeDataset.load_data_loader(batch_size, max_window_size,num_noise_words)
    lr, num_epochs = 0.002, 500
    net = getNet()
    train(net, data_iter, lr, num_epochs)
    get_similar_tokens('中国', 3, net[0])
