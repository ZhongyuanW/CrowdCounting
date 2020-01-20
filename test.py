# -*- coding: utf-8 -*-
# @Time    : 10/15/19 11:30 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : test.py
# @Software: PyCharm

import torch
from config import *
import os
import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms

def test(image_path = os.path.join(HOME,"part_A_final/test_data/images","IMG_16.jpg"),
         save_path = SAVE_PATH):

    path_list = os.listdir("models/%s"%save_path)
    path_list.remove("log.txt")
    ep_list = [int(i.split("_")[2]) for i in path_list]
    curr_index = ep_list.index(max(ep_list))
    weight_path = os.path.join("models/%s"%save_path,path_list[curr_index])

    net = models.net.Net()
    print("load weight from: %s"%weight_path)

    net = net.cuda()
    weight = torch.load(weight_path)
    net.load_state_dict(weight)
    print("load weight completed!")
    net.eval()

    if not os.path.exists(image_path):
        print("not find image path!")
        exit(-1)
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image,dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_tensor = transform(image)
    image = Variable(img_tensor.unsqueeze(0).cuda())

    gt_dmap_root = os.path.join(HOME, "part_A_final", "test_data/density_maps_constant")
    gt_dmap = np.load(os.path.join(gt_dmap_root, image_path.split("/")[-1].replace('.jpg', '.npy')))

    density_map = net(image).squeeze(0).cpu().data

    crowd_counting = density_map.sum()
    return crowd_counting,gt_dmap.sum(),density_map


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    count, gt, densitymap = test()
    print("the image has %.4f persons, gt: %.4f"%(count,gt))
    densitymap = densitymap.squeeze(0)
    plt.imsave("res/res.png",densitymap/densitymap.max()*255)
    print("the prec density map save at res/res.png")