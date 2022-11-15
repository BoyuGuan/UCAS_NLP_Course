# pytorch code for sequence tagging

# 此版本为简单的NER代码，没有使用CRF和训练好的词向量，仅做参考使用。
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.optim import Adam, SGD

from utils import  build_dict, cal_max_length, Config
from word2vec import getNet
from model import BiLSTM_CRF


logger = logging.getLogger('BiLSTM_CRF')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 
logDir = './log/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")



class NERdataset(Dataset):

    def __init__(self, data_dir, split, word2id, tag2id, max_length):
        file_dir = data_dir + split
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file).readlines()
        label = open(label_file).readlines()
        self.corpus = []
        self.label = []
        self.length = []
        self.word2id = word2id
        self.tag2id = tag2id
        for corpusLine, labelLine in zip(corpus, label):
            assert len(corpusLine.split()) == len(labelLine.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['unk']
                                for temp_word in corpusLine.split()])
            self.label.append([tag2id[temp_label] for temp_label in labelLine.split()])
            self.length.append(len(corpusLine.split()))
            if(len(self.corpus[-1]) > max_length):
                self.corpus[-1] = self.corpus[-1][:max_length]
                self.label[-1] = self.label[-1][:max_length]
                self.length[-1] = max_length
            else:
                padCount =  max_length - len(self.corpus[-1])
                self.corpus[-1] += [word2id['pad']] * padCount
                self.label[-1] += [tag2id['PAD']] * padCount

        self.corpus = torch.Tensor(self.corpus).long()
        self.label = torch.Tensor(self.label).long()
        self.length = torch.Tensor(self.length).long()

    def __getitem__(self, item):
        return self.corpus[item], self.label[item], self.length[item]

    def __len__(self):
        return len(self.label)

def val(model, optimizer, dataloader):
    model.eval()
    logger.info(f'epoch start valding')
    preds, labels = [], []
    for index, data in enumerate(dataloader):
        optimizer.zero_grad()
        corpus, label, length = data
        corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()
        best_path_all = model(corpus, length)
        for index, best_path in enumerate(best_path_all):
            preds.extend(best_path[:length[index]])
        for index, label_this_sentence in enumerate(label.tolist()):
            labels.extend(label_this_sentence[:length[index]])
    preds = [pred.to('cpu').item() for pred in preds]
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro', zero_division=1)
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds, zero_division=1)
    logger.info('\n {report}')
    return precision, recall, f1


def train(config, model, trainDataloader, optimizer, testDataloader):
    best_f1 = 0.0
    for epoch in range(config.epochNum):
        model.train()
        logger.info(f'epoch: {epoch + 1} start training')
        for index, data in enumerate(trainDataloader):
            optimizer.zero_grad()
            sentence, tag, length = data
            sentence, tag, length = sentence.cuda(), tag.cuda(), length.cuda()
            loss = model.neg_log_likelihood(sentence, tag, length)
            loss.backward()
            optimizer.step()
            if (index % 10 == 0):
                logger.info(f'epoch: {epoch+1} step:{index}------------loss:{loss.item()}')
        prec, rec, f1 = val(model, optimizer, testDataloader)
        if(f1 > best_f1):
            torch.save(model, os.path.join(logDir, 'model.pkl'))
            # break

if __name__ == '__main__':
    config = Config()

    os.makedirs(logDir,exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(logDir, 'train.log') )
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)



    logger.info('loading data...')
    word2id, tag2id = build_dict()
    max_length = cal_max_length(config.data_dir)
    trainset = NERdataset(config.data_dir, 'train', word2id, tag2id, max_length)
    trainDataloader = DataLoader(trainset, batch_size=config.batch_size)
    testset = NERdataset(config.data_dir, 'test', word2id, tag2id, max_length)
    testDataloader = DataLoader(testset, batch_size=config.batch_size)

    logger.info('loading embedding...')
    word2vecEmbedding = getNet(len(word2id))
    word2vecEmbedding.load_state_dict(torch.load(os.path.join('word2vec' ,'net.pt')))
    word2vecEmbedding = word2vecEmbedding[0].cuda()
    # nerlstm = NERLSTM(config.embedding_dim, config.hidden_dim, word2vecEmbedding, config.dropout, word2id, tag2id).cuda()
    nerlstm = BiLSTM_CRF(config.embedding_dim, config.hidden_dim, word2vecEmbedding, tag2id).cuda()
    optimizer = SGD(nerlstm.parameters(), config.learning_rate, weight_decay=config.weight_decay)

    # 词向量嵌入层不再需要调整
    for para in nerlstm.word_embeds.parameters():
        para.requires_grad = False
    logger.info('start training...')
    train(config, nerlstm, trainDataloader, optimizer, testDataloader)

