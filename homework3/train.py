# pytorch code for sequence tagging

# 此版本为简单的NER代码，没有使用CRF和训练好的词向量，仅做参考使用。
import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.optim import Adam, SGD

from utils import  build_dict, cal_max_length, Config
from word2vec import getNet

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# TODO 改用训练好的词向量
# TODO 加上使用CRF


class NERLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, word2vecEmbedding, dropout, word2id, tag2id):
        super(NERLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id)
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = word2vecEmbedding
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def forward(self, x):
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return outputs



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
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)  # ignore the pad label
    preds, labels = [], []
    for index, data in enumerate(dataloader):
        optimizer.zero_grad()
        corpus, label, length = data
        corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()
        output = model(corpus)
        predict = torch.argmax(output, dim=-1)
        loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))
        leng = []
        for i in label.cpu():
            tmp = []
            for j in i:
                if j.item() < 7:
                    tmp.append(j.item())
            leng.append(tmp)

        for index, i in enumerate(predict.tolist()):
            preds.extend(i[:len(leng[index])])

        for index, i in enumerate(label.tolist()):
            labels.extend(i[:len(leng[index])])

    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro', zero_division=1)
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds, zero_division=1)
    print(report)
    model.train()
    return precision, recall, f1




def train(config, model, trainDataloader, optimizer, testDataloader):

    # ignore the pad label
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    best_f1 = 0.0
    for epoch in range(config.epochNum):
        model.train()
        for index, data in enumerate(trainDataloader):
            optimizer.zero_grad()
            corpus, label, length = data
            corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()
            output = model(corpus)
            loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))
            loss.backward()
            optimizer.step()
            if (index % 200 == 0):
                print('epoch: ', epoch, ' step:%04d,------------loss:%f' % (index, loss.item()))

        prec, rec, f1 = val(model, optimizer, testDataloader)
        if(f1 > best_f1):
            torch.save(model, config.save_model)


if __name__ == '__main__':
    config = Config()
    os.makedirs(config.savePath, exist_ok=True)

    word2id, tag2id = build_dict()
    max_length = cal_max_length(config.data_dir)
    trainset = NERdataset(config.data_dir, 'train', word2id, tag2id, max_length)
    trainDataloader = DataLoader(trainset, batch_size=config.batch_size)
    testset = NERdataset(config.data_dir, 'test', word2id, tag2id, max_length)
    testDataloader = DataLoader(testset, batch_size=config.batch_size)

    word2vecEmbedding = getNet(len(word2id))
    word2vecEmbedding.load_state_dict(torch.load(os.path.join('word2vec' ,'net.pt')))
    word2vecEmbedding = word2vecEmbedding[0]
    nerlstm = NERLSTM(config.embedding_dim, config.hidden_dim, word2vecEmbedding, config.dropout, word2id, tag2id).cuda()

    # 词向量嵌入层不再需要调整
    for para in nerlstm.word_embeds.parameters():
        para.requires_grad = False
    
    optimizer = Adam(nerlstm.parameters(), config.learning_rate)
    train(config, nerlstm, trainDataloader, optimizer, testDataloader)

