'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-09-21 21:23:04
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-09-24 01:46:24
FilePath: /guan/ucas/nlp/homework1/DNNtrain.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models

from utils import progress_bar
from handleData import prepaingData

logger = logging.getLogger('myDNNTrain')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 
logDir = './log/' + 'DNN_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
best_acc = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*224*224, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def test(net, criterion, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        torch.save(net, os.path.join(logDir,'DNNTrained.pth'))
        best_acc = acc
        logger.info(f'get best acc {best_acc}')



def train(net, epochs, criterion, trainloader, testloader):
    net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr= 0.0001,
                        momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # scheduler.step()
        test(net, criterion, testloader)


if __name__ == "__main__":
    os.makedirs(logDir,exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(logDir, 'DNNtrain.log') )
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    logger.info('Preparing data...')
    trainloader, testloader, classes = prepaingData('./data1')
    logger.info('Data prepared')

    net = DNN()
    logger.info("Use DNN model")
    train(net, 200, nn.CrossEntropyLoss(), trainloader, testloader)