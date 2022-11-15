'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-11-09 21:39:41
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-11-15 16:39:47
FilePath: /guan/ucas/nlp/homework3/model.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence

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

class BiLSTM_CRF(nn.Module):

    def __init__(self,  embedding_dim, hidden_dim, word2vecEmbedding, tag2id):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag2id)
        self.tag2id = tag2id
        self.word_embeds = word2vecEmbedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # 每个rnn节点输出是各个词性的概率
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 各个词性的转移矩阵，transitions[i][j]是从j转移到i的概率
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 确保不会从其他词性转移到START_TAG，也不会从STOP_TAG转移到其他词性
        self.transitions.data[self.tag2id["START_TAG"], :] = -10000
        self.transitions.data[:, self.tag2id["STOP_TAG"]] = -10000
        # self.transitions.data[:, self.tag2id["PAD"]] = 


    def _get_lstm_features(self, sentences):
        # self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentences)
        lstm_out, _hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out.cuda())
        return lstm_feats

    def _viterbi_decode(self, feats, length):

        # 初始化 viterbi 变量
        init_vvars = torch.full((len(feats), self.tagset_size), -10000.).cuda()
        init_vvars[:, self.tag2id["START_TAG"]] = 0
        # path_score = []
        best_path_all = []
        for i, sentence in enumerate(feats):
            # 在第j步中 forward_var 保存着第j-1步的 viterbi 变量
            backpointers = []
            forward_var = init_vvars[i].view(1,-1)
            for j, feat in enumerate(sentence):
                if j >= length[i]:
                    break
                # 对于每个词bilstm输出的各个词性的概率
                bptrs_t = []  # 保存这一步的回溯指针 holds the backpointers for this step
                viterbivars_t = []  # 保存这一步的viterbi 变量  holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] 中存放着tag i之前的viterbi变量并加上从tag i到 next_tag的转移概率（因为需要计算argmax，所以放在循环后做）
                    # next_tag_var[i] holds the viterbi variable for tag i at the previous step, plus the score of transitioningfrom tag i to next_tag.
                    next_tag_var = forward_var + self.transitions[next_tag]     # 从各个词性转移过来该词性需要花费的转移向量
                    best_tag_id = torch.argmax(next_tag_var, 1) # best_tag_id: 从i-1步转移到i步词性是next_tag的最大概率词性
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # 加上emmision值，然后将其存入forward_var中供下一步使用 Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # 计算向句末转移的概率
            terminal_var = forward_var + self.transitions[self.tag2id["STOP_TAG"]]

            best_tag_id =  torch.argmax(terminal_var, 1)
            # path_score.append(terminal_var[0][best_tag_id])

            # 反向找最近的路径
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # 去除开始的"START"tag Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag2id["START_TAG"]  # Sanity check
            best_path.reverse()
            best_path_all.append(best_path)
        # return path_score, best_path_all
        return best_path_all


    def forward(self, sentence, length):  # 数据在模型中的正向传播
        # 获得正向传播的emmison分数
        lstm_feats = self._get_lstm_features(sentence)
        # 找到最优路径
        return self._viterbi_decode(lstm_feats, length)
        # 找到最优路径，给出分数
        # score, tag_seq = self._viterbi_decode(lstm_feats, length)
        # return score, tag_seq

    def _log_sum_exp(self, vec):
        # 计算前向传播时的log_sum_exp
        max_score = vec[0, torch.argmax(vec, 1)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _forward_alg(self, feats, length):
        # 前向算法并计算score

        # START_TAG拥有全部的分数, len(feats)是为了获得其batch_size
        init_alphas = torch.full((len(feats), self.tagset_size), -10000.).cuda()
        init_alphas[:, self.tag2id["START_TAG"]] = 0.

        # 在第i步中 forward_var 保存着第i-1步的 viterbi 变量
        alpha = []

        # 对每个句子进行计算
        for i, sentence in enumerate(feats):
            forward_var = init_alphas[i].view(1,-1)
            # 对每个词的各个词性的概率进行迭代
            for j, feat in enumerate(sentence):
                if j >= length[i]:
                    break
                alphas_t = []  # 在此步的alpha变量
                for next_tag in range(self.tagset_size):
                    # transitions矩阵的第next_tag行，是从其他标签转移到next_tag的概率
                    trans_score = self.transitions[next_tag].view(1, -1)
                    # 广播发射分数，从一个数广播成(1, self.tagset_size)大小。 无论前一个标签是什么，它都是一样的
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                    # next_tag_var[i]中存放着edge(i -> next_tag)在我们做log-sum-exp的分数
                    # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                    next_tag_var = forward_var + trans_score + emit_score
                    # 这个tag的forward var是所有分数的log-sum-exp 
                    # The forward variable for this tag is log-sum-exp of all the scores.
                    alphas_t.append(self._log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[self.tag2id["STOP_TAG"]]
            alpha.append(self._log_sum_exp(terminal_var).view(1))
        return torch.cat(alpha).view(-1)


    def _score_sentence(self, feats, tags, length):
        # 计算给定特征和标签的分数
        score = torch.zeros((len(feats), 1)).cuda()
        for sentenceIndex, sentence in enumerate(feats):    
            sentence_tag = torch.cat([torch.tensor([self.tag2id["START_TAG"]], dtype=torch.long).cuda(), tags[sentenceIndex]])
            for i in range(length[sentenceIndex]):
                score[sentenceIndex] += self.transitions[sentence_tag[i + 1], sentence_tag[i]] + sentence[i, sentence_tag[i + 1]]
            score[sentenceIndex] += self.transitions[self.tag2id["STOP_TAG"], sentence_tag[-1]]
        return score.view(-1)

    def neg_log_likelihood(self, sentence, tags, length):
        # 此函数用来计算正向传播时的loss
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats, length)
        gold_score = self._score_sentence(feats, tags, length)
        return torch.sum(forward_score - gold_score)


