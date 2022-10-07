'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-09-24 13:40:20
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-09-24 15:26:20
FilePath: /guan/ucas/nlp/homework1/makePrediction.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from PIL import Image
from torch import nn

import torch
from torchvision import transforms

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def predictDataset(modelPath:str):
    model = torch.load(modelPath)
    model.eval()
    model.to("cuda")
    print('id', 'label', sep=',')
    for i in range(1, 12501):
        imageName = f'{i}.jpg'
        image = Image.open(os.path.join("./test", imageName))
        image = transform_test(image)
        image = image.to("cuda")
        image = image.unsqueeze(0)
        out = model(image)
        out = float(nn.functional.softmax(out, dim=1)[0, 1])
        print(i, out, sep=',')


def predictFault(modelPath:str):
    model = torch.load(modelPath)
    model.eval()
    model.to("cuda")
    classes = ['dog', 'cat']
    for className in classes:
        for imageName in os.listdir(os.path.join("./data1", className)):
            image = Image.open(os.path.join("./data1", className, imageName))
            image = transform_test(image)
            image = image.to("cuda")
            image = image.unsqueeze(0)
            out = int(model(image).argmax())
            if out == 0 and className == 'dog':
                print(imageName)
            elif out == 1 and className == 'cat':
                print(imageName)
            
if __name__ == "__main__":
    # predictFault("./log/CNN_2022-09-24-01-47-39/CNNTrained.pth")

    predictDataset("./log/CNN_2022-09-24-01-47-39/CNNTrained.pth")
