'''
Author: Jack Guan cnboyuguan@gmail.com
Date: 2022-10-27 11:13:08
LastEditors: Jack Guan cnboyuguan@gmail.com
LastEditTime: 2022-10-27 16:57:27
FilePath: /guan/ucas/nlp/homework2/monitor.py
Description: 

Copyright (c) 2022 by Jack Guan cnboyuguan@gmail.com, All Rights Reserved. 
'''
import logging
import time
import os

import pynvml
import psutil




logger = logging.getLogger('monitor')
logger.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s: %(message)s")
os.makedirs('./monitorLog', exist_ok=True)
fileHandler = logging.FileHandler(os.path.join('./monitorLog', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "_monitor.log"))
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
commandHandler = logging.StreamHandler()
commandHandler.setLevel(logging.INFO)
commandHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.addHandler(commandHandler)

while True:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    logger.info( f'{str(psutil.virtual_memory().used / (2**30))}   {str( meminfo.used/1024**2)}' )
    time.sleep(4)