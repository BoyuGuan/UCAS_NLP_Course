'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-13 20:17:46
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-11-10 23:01:20
FilePath: /guan/ucas/nlp/homework3/word2vec.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import os
import random
import collections
import logging
from datetime import datetime

import pickle
import torch
from torch import nn

logger = logging.getLogger('word2vector')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 
logDir = './word2vec/'
minLossTest = 6666666

def count_corpus(tokens):
    """Count token frequencies.

    Defined in :numref:`sec_text_preprocessing`"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['unk'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs



def read_txt(txtPath:str = './data/renmin.txt'):
    """将PTB数据集加载到文本行的列表中"""
    # data_dir = d2l.download_extract('ptb')
    # print(data_dir)
    # Readthetrainingset.
    with open(txtPath) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


def count_corpus(tokens):
    """Count token frequencies.

    Defined in :numref:`sec_text_preprocessing`"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)



def subsample(sentences, vocab):
    """下采样高频词"""
    # 排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = count_corpus(sentences)

    return ([[token for token in line] for line in sentences], counter)

def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词和上下文词
        中心词是一D的list，每个值一个单词。上下文是2D的list其中
        每一个子list都对应同样位置的中心词的上下文
    """
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下文窗口中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存10000个随机采样结果，避免来回调用
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    # 根据word2vec论文中的建议，将噪声词的采样概率设置为其在字典中的相对频率，其幂为0.75
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            # 每一个上下文词配K个噪声词
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

def batchify(data):
    """返回带有负采样的跳元模型的小批量样本"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        # 将其改成等长的返回数组
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)] # 有意义的值
        labels += [[1] * len(context) + [0] * (max_len - len(context))] # 是正确上下文的值
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))

def load_data_loader(batch_size, max_window_size, num_noise_words, \
    trainDatasetPath, testDatasetPath ):
    """下载PTB数据集，然后将其加载到内存中"""
    num_workers = 0
    trainSentences = read_txt(trainDatasetPath)
    testSentences = read_txt(testDatasetPath)
    allVocab = Vocab(trainSentences + testSentences, min_freq=0, reserved_tokens=['pad'])
    trainSubsampled, trainCounter = subsample(trainSentences, allVocab)
    testSubsampled, testCounter = subsample(testSentences, allVocab)
    trainCorpus = [allVocab[line] for line in trainSubsampled]
    testCorpus = [allVocab[line] for line in testSubsampled]
    trainAllCenters, trainAllContexts = get_centers_and_contexts(
        trainCorpus, max_window_size)
    trainAllNegatives = get_negatives(
        trainAllContexts, allVocab, trainCounter, num_noise_words)
    testAllCenters, testAllContexts = get_centers_and_contexts(
        testCorpus, max_window_size)
    testAllNegatives = get_negatives(
        testAllContexts, allVocab, testCounter, num_noise_words)

    class corpusDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            # 长度必然相等，centers每个元素是文本中每一个被选出来的词的中心词（自己），
            # context每个元素是上下文词（一个装有上下文的1D list）
            # negatives每个元素是负采样词（装有负采样元素的K倍长于上下文词的1D list）
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            # 取一次取一个tuple，分别是一个中心词，一个1D的上下文list，
            # 一个是上下文list K倍长的1D list
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)
    trainDataset = corpusDataset(trainAllCenters, trainAllContexts, trainAllNegatives)
    testDataset = corpusDataset(testAllCenters, testAllContexts, testAllNegatives)
    trainDataIter = torch.utils.data.DataLoader(
        trainDataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
    testDataIter = torch.utils.data.DataLoader(
        testDataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
# 在创建DataLoader类的对象时，collate_fn函数会将batch_size个样本整理成一个batch样本，便于批量训练。
    return trainDataIter, testDataIter, allVocab


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


def getNet(num_embeddings, embed_size = 300):
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
    minLossTrain = 666666
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


    logger.info('Preparing data...')
    trainFile, testFile = './data/train_corpus.txt', './data/test_corpus.txt'
    batch_size, max_window_size, num_noise_words = 2048, 5, 5
    trainDataIter, testDataIter, vocab = load_data_loader(batch_size,\
        max_window_size, num_noise_words, trainFile, testFile)
    with open( os.path.join(logDir,'vocab_word2idx.pkl') , 'wb') as f:
        pickle.dump(vocab.token_to_idx, f)
    lr, num_epochs = 0.002, 5
    net = getNet(len(vocab))
    logger.info(f'Data prepared, len of vocab is {len(vocab)}')
    logger.info('Start training')
    loss = SigmoidBCELoss()
    logger.info("start training")
    train(net, loss, trainDataIter, testDataIter, lr, num_epochs)