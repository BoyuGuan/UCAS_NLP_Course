'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-26 20:45:21
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-10-28 14:43:56
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



def makeDataFiles(stopWords: set, dataSetPath: str = './data/RenMin_Daily', trainSavePath: str = './data/renmin.txt',\
    tesSavePath: str = './data/renmin_test.txt', percent = 0.1 ):
    """将所有文件合并到一个文件中"""
    fileNames = os.listdir(dataSetPath)
    numOfFiles = int(len(fileNames) * percent)
    with open(trainSavePath, 'w') as trainFile:
        for i in range(numOfFiles):
            fileName = fileNames[i]
            with open(os.path.join(dataSetPath, fileName), 'r') as f1:
                for line in f1:
                    line = line.strip()
                    # 空行
                    if line == '':
                        continue
                    newLine = []
                    for word in jieba.cut(line, cut_all= False):
                        # 去除停顿词
                        word = word.strip()
                        if word != '' and word not in stopWords:
                            newLine.append(word)
                    # 太短的行不要
                    if len(newLine)  <  4:
                        continue
                    newLine = " ".join(newLine)
                    newLine = newLine.strip()
                    newLine += '\n'
                    trainFile.write(newLine)
                    
    with open(tesSavePath, 'w') as testFile:
        for i in range(numOfFiles+2, numOfFiles * 2 + 2):
            fileName = fileNames[i]
            with open(os.path.join(dataSetPath, fileName), 'r') as f1:
                for line in f1:
                    line = line.strip()
                    # 空行
                    if line == '':
                        continue
                    newLine = []
                    for word in jieba.cut(line, cut_all= False):
                        # 去除停顿词
                        word = word.strip()
                        if word != '' and word not in stopWords:
                            newLine.append(word)
                    # 太短的行不要
                    if len(newLine)  <  4:
                        continue
                    newLine = " ".join(newLine)
                    newLine = newLine.strip()
                    newLine += '\n'
                    testFile.write(newLine)

if __name__  == '__main__':
    stopWords = get_stopword_list('./data/hit_stopwords.txt')
    makeDataFiles(stopWords, trainSavePath='./data/renmin_tiny2.txt', tesSavePath='./data/renmin_tiny2_test.txt', percent = 0.2)


