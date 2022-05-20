import shutil
import os
import random
import pickle
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import glob

path1 = "/home/ec2-user/SageMaker/hockey/"
out_path = "/home/ec2-user/SageMaker/hockey_all/"

classes = sorted(os.listdir(path1))

for category in classes:
    if not os.path.exists(out_path+category):
        os.makedirs(out_path+category)
    images = glob.glob(path1+category+'/**/*.jpg', recursive=True)
    
    
    for img in images:
        shutil.copy(img,out_path+category)
    
    
#     if len(images)>=10:
#         image_list = random.sample(images, 10)
#         for img in image_list:
#             shutil.copy(img,out_path+category)
#     elif len(images)<10:
#         nn = 0
#         while(len(os.listdir(out_path+category))<10):
#             img = random.sample(images, 1)[0]
#             print(img)
#             shutil.copy(img , out_path+category+"/"+img.split("/")[-1].split(".")[0]+"_"+str(nn)+"."+img.split(".")[-1])
#             nn = nn+1