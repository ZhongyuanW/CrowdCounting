# -*- coding: utf-8 -*-
# @Time    : 10/15/19 10:59 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : train.py
# @Software: PyCharm

from config import *
import models
import dataset_constant as Dataset
import torch.utils.data.dataloader as Dataloader
import torch.nn as nn
import torch.optim as optim
import time
import torch
import visdom
from torch.autograd import Variable
import sys
import numpy as np
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

viz = visdom.Visdom(env=SAVE_PATH.replace("/","_"))

def train():

    if not os.path.exists("models/%s/%s" % (SAVE_PATH.split("/")[0],SAVE_PATH.split("/")[1])):
        os.mkdir("models/%s/%s" % (SAVE_PATH.split("/")[0],SAVE_PATH.split("/")[1]))
    if not os.path.exists("models/%s/%s/%s" % (SAVE_PATH.split("/")[0],SAVE_PATH.split("/")[1],SAVE_PATH.split("/")[2])):
        os.mkdir("models/%s/%s/%s" % (SAVE_PATH.split("/")[0],SAVE_PATH.split("/")[1],SAVE_PATH.split("/")[2]))
    if not os.path.exists("models/%s"%SAVE_PATH):
        os.mkdir("models/%s"%SAVE_PATH)


    config_log = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()) + \
        "\n-------------------------------------------------------------" \
        "\nconfig:\n%s" \
        "-------------------------------------------------------------"
    l_temp = ""
    for i in range(len(VAR_LIST)):
        l_temp += "\t%s\n" % VAR_LIST[i]
    config_log = config_log % l_temp
    with open(os.path.join("models", SAVE_PATH, "log.txt"), "a+") as f:
        f.write(config_log + "\n\n")
    print(config_log)

    dataset = Dataset.Dataset(gt_downsample=8)
    dataloader = Dataloader.DataLoader(dataset, batch_size=BATCH_SIZE,num_workers=8,
                        shuffle=True, drop_last=False,worker_init_fn=worker_init_fn)
    # print("dataset size is: %d"%dataset.__len__())

    test_dataset = Dataset.Dataset(phase="test", gt_downsample=8)
    test_dataloader = Dataloader.DataLoader(test_dataset,batch_size=1,num_workers=8,
                        shuffle=False, drop_last=False,worker_init_fn=worker_init_fn)

    net = models.net.Net()
    net = net.cuda()
    if OPTIMIZER == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    if LOSS_F == "MSE":
        criterion = nn.MSELoss(reduction='sum').cuda()
    elif LOSS_F == "L1":
        criterion = nn.L1Loss(reduction='sum').cuda()
    else:
        criterion = nn.MSELoss(reduction='sum').cuda()

    t0 = time.time()
    start_epoch = 0
    step_index = 0

    min_mae = sys.maxsize
    min_epoch = -1
    epoch_list = []
    train_loss_list = []
    epoch_loss_list = []
    test_mae_list = []

    # if RESUME:
    #     path_list = os.listdir("models/%s"%SAVE_PATH)
    #     path_list.remove("log.txt")
    #     ep_list = [int(i.split("_")[2]) for i in path_list]
    #     curr_index = ep_list.index(max(ep_list))
    #     start_epoch = ep_list[curr_index]
    #     weight_path = os.path.join("models/%s"%SAVE_PATH,path_list[curr_index])
    #
    #     weight = torch.load(weight_path)
    #     net.load_state_dict(weight)
    #     print("resume weight %s, at %d\n" % (weight_path,start_epoch))
    #     min_mae = float(path_list[curr_index].split("_")[-2][3:])/100
    #     min_epoch = start_epoch
    #     start_epoch = start_epoch + 1
    #
    #     for i in start_epoch:
    #         if i in STEPS:
    #             step_index += 1

    for i in range(start_epoch, MAX_EPOCH):

        if LR_DECAY and (i in STEPS):
            step_index += 1
            adjust_learning_rate(optimizer, LR_DECAY, step_index)

        ## train ##
        epoch_loss = 0
        net.train()
        for _,(images,dt_targets) in enumerate(dataloader):

            images,dt_targets = Variable(images.cuda()),Variable(dt_targets.cuda())

            densitymaps = net(images)

            loss = criterion(densitymaps, dt_targets)


            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss_list.append(epoch_loss)
        train_loss_list.append(epoch_loss/len(dataloader))
        epoch_list.append(i)
        localdate = time.strftime("%Y/%m/%d %H:%M:%S",time.localtime())
        with open(os.path.join("models",SAVE_PATH,"log.txt"),"a+") as f:
            f.write(localdate+"\n")
        print(localdate)
        train_log = "train [%d/%d] timer %.4f, loss %.4f" % \
                    (i, MAX_EPOCH, time.time() - t0, epoch_loss / len(dataloader))
        with open(os.path.join("models",SAVE_PATH,"log.txt"),"a+") as f:
            f.write(train_log+"\n")
        print(train_log)

        t0 = time.time()

        ## eval ##
        net.eval()

        with torch.no_grad():
            mae = 0
            mse = 0

            for _,(images,dt_targets) in enumerate(test_dataloader):
                images, dt_targets = Variable(images.cuda()), Variable(dt_targets.cuda())

                densitymaps = net(images)

                # TODO bug?
                mae += abs(densitymaps.data.sum()-dt_targets.data.sum()).item()
                mse += (densitymaps.data.sum()-dt_targets.data.sum()).item()**2

            mae = mae / len(test_dataloader)
            mse = (mse / len(test_dataloader)) **(1/2)

            if(mae<min_mae):
                min_mae = mae
                min_epoch = i
                save_log = "save state, epoch: %d" % i
                with open(os.path.join("models", SAVE_PATH, "log.txt"), "a+") as f:
                    f.write(save_log + "\n")
                print(save_log)
                torch.save(net.state_dict(), "models/%s/%s_epoch_%d_mae%d_mse%d.pth" % (SAVE_PATH,MODEL,i,mae*100,mse*100))
            test_mae_list.append(mae)

            eval_log = "eval [%d/%d] mae %.4f, mse %.4f, min_mae %.4f, min_epoch %d\n"%(i,MAX_EPOCH,mae,mse,min_mae, min_epoch)
            with open(os.path.join("models",SAVE_PATH,"log.txt"),"a+") as f:
                f.write(eval_log+"\n")
            print(eval_log)

            ## vis ##
            viz.line(win="1", X=epoch_list, Y=train_loss_list, opts=dict(title="train_loss"))
            viz.line(win="2", X=epoch_list, Y=test_mae_list, opts=dict(title="test_mae"))

            index = random.randint(0,len(test_dataloader)-1)
            image,gt_map = test_dataset[index]
            viz.image(win="3",img=image,opts=dict(title="test_image"))
            viz.image(win="4",img=gt_map/(gt_map.max())*255,opts=dict(title="gt_map_%.4f"%(gt_map.sum())))

            image = Variable(image.unsqueeze(0).cuda())
            # densitymap,_ = net(image)
            densitymap = net(image)
            densitymap = densitymap.squeeze(0).detach().cpu().numpy()
            viz.image(win="5",img=densitymap/(densitymap.max())*255,opts=dict(title="predictImages_%.4f"%(densitymap.sum())))

def adjust_learning_rate(optimizer, gamma, step):
    lr = LEARNING_RATE * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def setup_seed(seed=19960715):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed)
    torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id): # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    setup_seed()
    train()