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

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
# from backbone import Backbone


from model_irse import Backbone
import knn_infer
import threading

root_path = "/home/ec2-user/SageMaker/soccer_year_all_merge/"
out_path = "/home/ec2-user/SageMaker/result/"
emb_model_path = "/home/ec2-user/SageMaker/models/Backbone_IR_SE_152_Epoch_65_Batch_126700_Time_2022-05-12-05-53_checkpoint.pth"
# knn_path = "/home/ec2-user/SageMaker/all_knn/"
# labels_path = "/home/ec2-user/SageMaker/all_knn/"

knn_path  = "knn-bv/"
labels_path = "knn-bv/"

seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Embedding models
EMBEDDING_MODEL_PATH = emb_model_path
EMBEDDING_SIZE = 512
INPUT_SIZE =[224, 224]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = Backbone(INPUT_SIZE, 152, 'ir_se') # NEW
backbone.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
backbone.to(device)
backbone.eval()

# knn_out = knn_infer.Knn_infer(backbone,knn_path,labels_path) 
sports_category = sorted(os.listdir(root_path))



process = []
for i in range(len(sports_category)):
    years = sorted(os.listdir(root_path+sports_category[i]))
    for year in years:
        print(root_path+sports_category[i]+"/"+year)
        print("##########################33",knn_path+sports_category[i]+"/"+year+"/"+"KNN_"+year+".pickle")
        knn_out = knn_infer.Knn_infer(backbone,knn_path+sports_category[i]+"/"+year+"/"+"KNN_"+year+".pickle",labels_path+sports_category[i]+"/"+year+"/"+"all_labels_"+year+".json") 
        # knn_out.inference(root_path+sports_category[i]+"/"+year,out_path)
        t1 = threading.Thread(target= knn_out.inference, args=(root_path+sports_category[i]+"/"+year,out_path))
        process.append(t1)
        
        
for pro in process[:10]:
    pro.start()
for pro in process[:10]:
    pro.join()
    
for pro in process[10:20]:
    pro.start()
for pro in process[10:20]:
    pro.join()
    
for pro in process[20:30]:
    pro.start()
for pro in process[20:30]:
    pro.join()
    
for pro in process[30:40]:
    pro.start()
for pro in process[30:40]:
    pro.join()   
    
for pro in process[40:50]:
    pro.start()
for pro in process[40:50]:
    pro.join()    
    
for pro in process[50:60]:
    pro.start()
for pro in process[50:60]:
    pro.join()    
    
for pro in process[60:]:
    pro.start()
for pro in process[60:]:
    pro.join()    







