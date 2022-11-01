'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-26 20:45:21
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-11-01 11:22:33
FilePath: /guan/ucas/nlp/homework2/handleRawData.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import os
from tkinter import E

import jieba
import pandas as pd

def get_stopword_list(file):
    stopWords = set()
    with open(file, 'r', encoding='utf-8') as f:    # 
        for line in f:
            stopWords.add(line.strip('\n'))
    return stopWords



def makeChineseDataFile(stopWords: set, dataSetPath: str = './data/RenMin_Daily', trainSavePath: str = './data/renmin.txt',\
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

def makeEnglishDataFile(stopWords: set, dataSetPath: str = './data/cnn.csv', trainSavePath: str = './data/cnn.txt',\
    tesSavePath: str = './data/cnn_test.txt', percent = 0.2 ):
    cnnNews = pd.read_csv(dataSetPath)["Article text"]
    cnnNews = cnnNews.dropna()
    cnnNews = cnnNews.tolist()
    numOfFiles = int(len(cnnNews) * percent)
    with open(trainSavePath, 'w') as trainFile:
        for i in range(numOfFiles):
            paragraph = cnnNews[i]
            paragraph = paragraph.split('.')
            for line in paragraph:
                line = line.strip()
                newLine = []
                for word in line.split():
                    word = word.replace('(CNN)', '')
                    word = word.replace(',', '')
                    word = word.replace('!', '')
                    word = word.replace('?', '')
                    word = word.replace(';', '')
                    word = word.replace(':', '')
                    word = word.replace('(', '')
                    word = word.replace(')', '')
                    word = word.replace('[', '')
                    word = word.replace(']', '')
                    word = word.replace('{', '')
                    word = word.replace('}', '')
                    word = word.replace('--', '')
                    word = word.replace('-', '')
                    word = word.replace('\'', '')
                    word = word.replace('\"', '')
                    word = word.replace('`', '')
                    word = word.replace('’', '')
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

    with open(tesSavePath, 'w') as trainFile:
        for i in range(numOfFiles + 2, 2 * numOfFiles + 2):
            paragraph = cnnNews[i]
            paragraph = paragraph.split('.')
            for line in paragraph:
                line = line.strip()
                newLine = []
                for word in line.split():
                    word = word.replace('(CNN)', '')
                    word = word.replace(',', '')
                    word = word.replace('!', '')
                    word = word.replace('?', '')
                    word = word.replace(';', '')
                    word = word.replace(':', '')
                    word = word.replace('(', '')
                    word = word.replace(')', '')
                    word = word.replace('[', '')
                    word = word.replace(']', '')
                    word = word.replace('{', '')
                    word = word.replace('}', '')
                    word = word.replace('--', '')
                    word = word.replace('-', '')
                    word = word.replace('\'', '')
                    word = word.replace('\"', '')
                    word = word.replace('`', '')
                    word = word.replace('’', '')
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



if __name__  == '__main__':
    # chineseStopWords = get_stopword_list('./data/hit_stopwords.txt')
    # makeChineseDataFile(chineseStopWords, trainSavePath='./data/renmin_tiny2.txt', tesSavePath='./data/renmin_tiny2_test.txt', percent = 0.2)
    englishStopWords = get_stopword_list('./data/en_stopwords.txt')

    makeEnglishDataFile(englishStopWords)