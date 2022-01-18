
import os
import torch
import torch.distributed as dist
from util.misc import launch_job
from train_net import train
import random
import shutil
from options.train_options import TrainOptions
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"    
def main():
    dataset_root = '../../datasets/DPR/'
    # trainfile = dataset_root+'randomdpr_test.txt'
    # last_dir = ""
    # dir_paths = []
    # save_image = []
    # with open(trainfile,'r') as f:
    #     for line in f.readlines():
    #         file_names = line.rstrip()
    #         file_name_arr = file_names.split(" ")
    #         save_image.append(file_name_arr[0]+'/'+file_name_arr[1])
    #         dir_paths.append(file_name_arr[0]+'/'+file_name_arr[2])
    #         # if last_dir != file_name_arr[0]:
    #         #     last_dir = file_name_arr[0]
    #         #     dir_paths.append(file_name_arr[0])
    # tgt_num = torch.arange(0, len(dir_paths)-1)
    # print(tgt_num.size())
    # tgt_shuffle = torch.randperm(len(dir_paths))
    # print(tgt_shuffle.size())
    # i=0
    # all = []
    # for file in save_image:
    #     target_dir = dir_paths[tgt_shuffle[i]]
    #     file_tmp = file+" "+target_dir
    #     all.append(file_tmp)
    #     i=i+1
    # file=open(dataset_root+"randomdprtgt_test.txt",'w') 
    # # file.write(str(num_image))
    # # file.write('\n')
    # # file.write(mean_score)
    # # file.write('\n')
    # # lists=[str(line)+"\n" for line in fmse_score_list]
    # for line in all:
    #     file.write(str(line)+"\n")
    # file.close() 

    # cp test image
    trainfile = dataset_root+'randomdprtgt_test.txt'
    target_root = dataset_root+"DPR_LT_test/"
    with open(trainfile,'r') as f:
        for line in f.readlines():
            file_names = line.rstrip()
            file_name_arr = file_names.split(" ")
            cpfile(file_name_arr[0], dataset_root)
            cpfile(file_name_arr[1], dataset_root)

def cpfile(source, dataroot):
    file_arr = source.split("/")
    to_dir = dataroot+'DPR_LT_test/'+file_arr[0]
    source_path = dataroot+'DPR_dataset/'+source
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    shutil.copy(source_path, to_dir)



if __name__=="__main__":
    main()
