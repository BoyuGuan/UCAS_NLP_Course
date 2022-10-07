import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from utils import progress_bar
from handleData import prepaingData

logger = logging.getLogger('myRNNTrain')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 
logDir = './log/' + 'RNN_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
best_acc = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNN(nn.Module):
    def __init__(self, in_feature=3*224, hidden_feature=1000, num_class=2, num_layers=2):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(in_feature, hidden_feature, num_layers) # 使用两层 GRU
        self.classifier = nn.Linear(hidden_feature, num_class) 

    def forward(self, x):
        x = x.permute(0,2,3,1) # 此步与下一步是为了将每一列3个通道的像素值放在一起
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(1, 0, 2) # 将最后一维放到第一维，变成每次输入的维度为3*224，输入224次
        out, _ = self.rnn(x) 
        out = out[-1] # 取序列中的最后一个输出
        out = self.classifier(out) # 将其导入全连接层分类
        return out


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
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
        scheduler.step()
        test(net, criterion, testloader)


if __name__ == "__main__":
    os.makedirs(logDir,exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(logDir, 'RNNtrain.log') )
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

    net = RNN()
    logger.info("Use RNN model")
    train(net, 300, nn.CrossEntropyLoss(), trainloader, testloader)