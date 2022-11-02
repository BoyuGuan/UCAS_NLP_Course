'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-13 20:17:46
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-11-01 16:11:41
FilePath: /guan/ucas/nlp/homework2/makeDataset.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
from io import TextIOWrapper
import math
import os
import random
import collections

import torch
from d2l import torch as d2l

# import myutils

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
        self.idx_to_token = ['<unk>'] + reserved_tokens
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
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-3 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

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
    allVocab = Vocab(trainSentences + testSentences, min_freq=10)
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

    class renMinDataset(torch.utils.data.Dataset):
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
    trainDataset = renMinDataset(trainAllCenters, trainAllContexts, trainAllNegatives)
    testDataset = renMinDataset(testAllCenters, testAllContexts, testAllNegatives)
    trainDataIter = torch.utils.data.DataLoader(
        trainDataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
    testDataIter = torch.utils.data.DataLoader(
        testDataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
# 在创建DataLoader类的对象时，collate_fn函数会将batch_size个样本整理成一个batch样本，便于批量训练。
    return trainDataIter, testDataIter, allVocab

if __name__ == '__main__':
    data_iter, vocab = load_data_loader(2, 5, 2)
    for batch in data_iter:
        for name, data in zip(['centers', 'contexts_negatives', 'masks',
                               'labels'], batch):
            print(name, 'shape:', data.shape)
        break
