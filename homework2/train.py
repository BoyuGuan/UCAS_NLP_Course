'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-24 22:32:11
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-11-10 22:56:58
FilePath: /guan/ucas/nlp/homework2/train.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import logging
import os
from datetime import datetime
import pickle

import torch
from torch import nn

import makeDataset

logger = logging.getLogger('word2vector')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 
logDir = './log/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

minLossTest = 6666666

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


def getNet(num_embeddings, embed_size = 100):
    net = nn.Sequential(nn.Embedding(num_embeddings=num_embeddings,
                                    embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=num_embeddings,
                                    embedding_dim=embed_size))
    return net

def train(net, lossFunction, trainDataIter, testDataIter, lr, num_epochs, device='cuda'):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    loss = lossFunction
    net = net.to(device)
    minLossTrain = 1666666
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 规范化的损失之和，规范化的损失数
    for epoch in range(num_epochs):
        logger.info(f'The {epoch + 1}th epoch')
        num_batches = len(trainDataIter)
        epochLoss = 0
        net.train()
        logger.info(f'epoch {epoch + 1} start train')
        for i, batch in enumerate(trainDataIter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l = l.sum()
            epochLoss += l / num_batches
            if i % 20 == 0:
                print(f'epoch {epoch + 1}, batch {i}/{num_batches}. The loss is {l}')
            l.backward()
            optimizer.step()
        if epochLoss < minLossTrain:
            minLossTrain = epochLoss
            logger.info(f'***epoch {epoch + 1} get MIN train loss, is {minLossTrain}')
        else:
            logger.info(f'epoch {epoch + 1} loos is {epochLoss}')
        test(net, loss, testDataIter, epoch, device)
    
def test(net, criterion, data_iter, epoch, device = 'cuda' ):
    global minLossTest
    net.eval()
    with torch.no_grad():
        loss = criterion
        epochLoss = 0
        num_batches = len(data_iter)
        for _, batch in enumerate(data_iter):
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l = l.sum()
            epochLoss += l / num_batches
        if epochLoss < minLossTest:
            minLossTest = epochLoss
            logger.info(f'**** *** epoch {epoch + 1} get MIN test loss, is {epochLoss}\n')
            torch.save(net.state_dict(), os.path.join(logDir, 'net.pt'))
        else:
            logger.info(f'epoch {epoch + 1} get test loss, is {epochLoss} \n')

if __name__ == '__main__':
    os.makedirs(logDir,exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(logDir, 'train.log') )
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    zh, en = 0, 1
    trainFile, testFile = [  ('./data/renmin_tiny2.txt', './data/renmin_tiny2_test.txt'), ('./data/cnn.txt', './data/cnn_test.txt')][en]
    logger.info('Preparing data...')
    batch_size, max_window_size, num_noise_words = 2048, 5, 5
    trainDataIter, testDataIter, vocab = makeDataset.load_data_loader(batch_size,\
        max_window_size, num_noise_words, trainFile, testFile)
    with open( os.path.join(logDir,'vocab_idx2token.pkl') , 'wb') as f:
        pickle.dump(vocab.idx_to_token, f)
    with open( os.path.join(logDir,'vocab_token2idx.pkl') , 'wb') as f:
        pickle.dump(vocab.token_to_idx, f)

    lr, num_epochs = 0.002, 5
    net = getNet(len(vocab))
    logger.info('Start training')
    loss = SigmoidBCELoss()
    train(net, loss, trainDataIter, testDataIter, lr, num_epochs)
    logger.info('Training finished')
    
    # get_similar_tokens('中国', 3, net[0], vocab)