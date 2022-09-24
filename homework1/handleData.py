'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-09-20 21:18:55
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-09-24 11:18:56
FilePath: /guan/ucas/nlp/homework1/handleRawData.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEtor
'''
import os

from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms


def splitTrainData(trainDataPath:str ):
    """将原始数据集按类别分割到不同文件夹

    Args:
        trainDataPath (str): 原始数据集路径
    """
    dataPath = './data'
    os.makedirs(os.path.join(dataPath, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(dataPath, 'dog'), exist_ok=True)

    catCount, dogCount = 0, 0
    for imageName in os.listdir(trainDataPath):
        if imageName.startswith('cat'):
            catCount += 1
            os.rename(os.path.join(trainDataPath, imageName), os.path.join(dataPath, 'cat', f'cat_{catCount}.jpg'))
        elif imageName.startswith('dog'):
            dogCount += 1
            os.rename(os.path.join(trainDataPath, imageName), os.path.join(dataPath, 'dog', f'dog_{dogCount}.jpg'))

def makeDatasetToSquqre():
    """将原始数据集转换为方形
    """
    
    os.makedirs(os.path.join('./data1', 'cat'), exist_ok=True)
    os.makedirs(os.path.join('./data1', 'dog'), exist_ok=True)
    classNames = ['cat', 'dog']
    for className in classNames:
        for imageName in os.listdir(os.path.join('./data', className)):
            image = Image.open(os.path.join('./data', className, imageName))
            width, height = image.size
            newLen = min(width, height)
            image = image.crop((  int((width-newLen)/2), int((height-newLen)/2), int((width-newLen)/2) + newLen,  int((height-newLen)/2) + newLen))
            image.save(os.path.join('./data1', className, imageName))
    
    os.makedirs('./test', exist_ok=True)
    for imageName in os.listdir('./test1'):
        image = Image.open(os.path.join('./test1', imageName))
        width, height = image.size
        newLen = min(width, height)
        image = image.crop((  int((width-newLen)/2), int((height-newLen)/2), int((width-newLen)/2) + newLen,  int((height-newLen)/2) + newLen))
        image.save(os.path.join('./data1', className, imageName))
    
    


def prepaingData(rootpath='./data', batchSize=32):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    allDataset = torchvision.datasets.ImageFolder(root=rootpath)
    trainsetLen = int(len(allDataset) * 0.8 )
    trainset, testset = torch.utils.data.random_split(allDataset, [trainsetLen, len(allDataset) - trainsetLen])
    trainset.dataset.transform = transform_train
    testset.dataset.transform = transform_test
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batchSize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2)

    classes = ('cat', 'dog')
    return trainloader, testloader, classes

if __name__ == "__main__":
    # splitTrainData('./train')
    makeDatasetToSquqre()