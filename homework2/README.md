<!--
 * @Author: Jack Guan cnboyuguan@gmail.com
 * @Date: 2022-10-12 21:53:46
 * @LastEditors: Jack Guan cnboyuguan@gmail.com
 * @LastEditTime: 2022-11-02 19:49:49
 * @FilePath: /guan/ucas/nlp/homework2/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
-->
# 第二次作业，处理词向量

第二次的作业是训练word2vec。  
`handleRawData.py`将原始中英文数据集整合、清洗成一个txt文件  
`makeDataset.py`将数据集整理成跳元法格式的dataloader  
`train.py`用来训练word2vec并保存最优值  
`predictSim.py`用来预测相似的词向量  
`monitor.py`用来监测内存、显存使用并输出log保存  