'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-11-09 21:39:41
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-11-10 17:23:42
FilePath: /guan/ucas/nlp/homework3/model.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import torch
import torch.nn as nn

class NERLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id)
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def forward(self, x):
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return outputs