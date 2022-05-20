import os
import shutil

path1 = "/home/ec2-user/SageMaker/hockey_all/"
dest_path = "/home/ec2-user/SageMaker/hockey_all_year_merge/hockey/"
categories  = os.listdir(path1)

year = 2022
while year>1900:
    for category in categories:
        print(category)
        # print(category.split("--")[1])
        try:
            if str(year) in category.split("--")[1].split("_")[0]:
                
                if not os.path.exists(dest_path+str(year)+"/"+category):
                    os.makedirs(dest_path+str(year)+"/"+category)
                images = os.listdir(path1+category)
                for img in images:
                    shutil.copy(path1+category+"/"+img,dest_path+str(year)+"/"+category)
        except:
            continue
    year = year-1  