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
import threading
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
# from backbone import Backbone
import boto3

from model_irse import Backbone
import knn_model



# bucket_name = "ludex-ml-testing"

# s3 = boto3.resource('s3')

# s3.Bucket(bucket_name).download_file(orig_file, dest_file)

# from cloudpathlib import CloudPath
# cp = CloudPath("s3://ludex-ml-testing/knn_data/")
# cp.download_to("/home/ec2-user/SageMaker")



root_path = "/home/ec2-user/SageMaker/hockey_year_before_1999_parallel_10/"
out_path = "/home/ec2-user/SageMaker/all_knn/"
emb_model_path = "/home/ec2-user/SageMaker/models/Backbone_IR_SE_152_Epoch_65_Batch_126700_Time_2022-05-12-05-53_checkpoint.pth"

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

process = []
knn = knn_model.KNN(backbone,out_path) 
sports_category = sorted(os.listdir(root_path))
for i in range(len(sports_category)):
    years = sorted(os.listdir(root_path+sports_category[i]))
    for year in years:
        print(root_path+sports_category[i]+"/"+year)
        t1 = threading.Thread(target=knn.generate_knn, args=(root_path+sports_category[i]+"/"+year,))
        process.append(t1)
#         knn.generate_knn(root_path+sports_category[i]+"/"+year)



# for pro in process:
#     pro.start()
# for pro in process:
#     pro.join()



for pro in process[:5]:
    pro.start()
for pro in process[:5]:
    pro.join()
    
for pro in process[5:10]:
    pro.start()
for pro in process[5:10]:
    pro.join()
    
for pro in process[10:15]:
    pro.start()
for pro in process[10:15]:
    pro.join()
    
for pro in process[15:20]:
    pro.start()
for pro in process[15:20]:
    pro.join()   
    
for pro in process[20:25]:
    pro.start()
for pro in process[20:25]:
    pro.join()    
    
for pro in process[25:30]:
    pro.start()
for pro in process[25:30]:
    pro.join()    
    
for pro in process[30:35]:
    pro.start()
for pro in process[30:35]:
    pro.join()    

for pro in process[35:40]:
    pro.start()
for pro in process[35:40]:
    pro.join()  
    
for pro in process[40:]:
    pro.start()
for pro in process[40:]:
    pro.join()     

# for pro in process[45:50]:
#     pro.start()
# for pro in process[45:50]:
#     pro.join()

# for pro in process[50:55]:
#     pro.start()
# for pro in process[50:55]:
#     pro.join()
    
# for pro in process[55:60]:
#     pro.start()
# for pro in process[55:60]:
#     pro.join()
    
# for pro in process[60:65]:
#     pro.start()
# for pro in process[60:65]:
#     pro.join()

# for pro in process[65:70]:
#     pro.start()
# for pro in process[65:70]:
#     pro.join()
    
# for pro in process[70:]:
#     pro.start()
# for pro in process[70:]:
#     pro.join()   

    
# for pro in process[40:50]:
#     pro.start()
# for pro in process[40:50]:
#     pro.join()  
    
    
# for pro in process[50:60]:
#     pro.start()
# for pro in process[50:60]:
#     pro.join()
    
# for pro in process[60:]:
#     pro.start()
# for pro in process[60:]:
#     pro.join()       
    
    
# for pro in process[60:70]:
#     pro.start()
# for pro in process[60:70]:
#     pro.join()
    
# for pro in process[70:]:
#     pro.start()
# for pro in process[70:]:
#     pro.join()    