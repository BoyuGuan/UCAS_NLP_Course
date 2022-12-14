<!--
 * @Author: Jack Guan cnboyuguan@gmail.com
 * @Date: 2022-11-09 21:39:41
 * @LastEditors: Jack Guan cnboyuguan@gmail.com
 * @LastEditTime: 2022-11-15 15:04:39
 * @FilePath: /guan/ucas/nlp/homework3/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
-->
## 说明
此代码为序列标注（命名实体识别）的参考代码，实现的模型为双向LSTM，使用的数据为实验作业所给定的数据。没有使用训练好的词向量以及CRF模型，请同学们自行实现。

## 代码结构:
    .
    ├── data                         
    │   ├── train_corpus.txt              
    │   ├── train_label.txt          
    │   ├── test_corpus.txt         
    │   ├── test_label.txt  
    ├── train.py      # 模型训练、测试以及保存
    ├── model.py      # BiLSTM模型实现
    └── utils.py      # 工具类，包含词表构建以及训练的配置文件

## 使用:
训练模型
```
python train.py
```
测试模型请调用train.py中的val函数

## 踩坑
想将torch的二维矩阵的第3列统一赋值为0，结果写成了`mat[:][2]=0`，这是不对的，`mat[:]`等同于mat，`mat[:][2]=0`最终的效果等同于`mat[2]=0`变成了将第3行统一赋值为0。正确的写法应该是`mat[:,2]=0`。
