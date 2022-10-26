'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-26 20:45:21
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-10-26 22:49:51
FilePath: /guan/ucas/nlp/homework2/handleRawData.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import os

import jieba

def get_stopword_list(file):
    stopWords = set()
    with open(file, 'r', encoding='utf-8') as f:    # 
        for line in f:
            stopWords.add(line.strip('\n'))
    return stopWords



def makeFilesInToOne(stopWords: set, dataSetPath: str = './data/RenMin_Daily', savePath: str = './data/renmin.txt' ):
    """将所有文件合并到一个文件中"""
    with open(savePath, 'w') as renminAll:
        for fileName in os.listdir(dataSetPath):
            with open(os.path.join(dataSetPath, fileName), 'r') as f1:
                for line in f1:
                    line = line.strip()
                    # 空行
                    if line == '':
                        continue
                    newLine = []
                    for word in jieba.cut(line, cut_all= False):
                        # 去除停顿词
                        if word not in stopWords:
                            newLine.append(word)
                    # 太短的行不要
                    if len(newLine)  <  4:
                        continue
                    newLine = " ".join(newLine)
                    newLine = newLine.strip()
                    newLine += '\n'
                    renminAll.write(newLine)

if __name__  == '__main__':
    stopWords = get_stopword_list('./data/hit_stopwords.txt')
    # makeFilesInToOne(stopWords)










