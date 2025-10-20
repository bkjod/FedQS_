'''
Experiments on FedAT.

Reference:
Z. Chai, Y. Chen, A. Anwar, L. Zhao, Y. Cheng, and H. Rangwala, “Fedat: A high-performance and communication-efficient federated learning system with asynchronous tiers,” in Proceedings of the international conference for high performance computing, networking, storage and analysis, 2021, pp. 1–16
'''

import zmq 
import sys
import time
import resnet_model
import torch
import argparse
import json
import random
from data_pre_our import get_dataset
import numpy
import numpy as np
from collections import OrderedDict
import torchvision
import torch
from torch import nn
import traceback
import csv
import os
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)   

mkdir("./fedat")

def eval_model(model, data_loader):
    print("Start evaluating the model!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0

    for batch_id, batch in enumerate(data_loader):
        data, target = batch
        dataset_size += data.size()[0]
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        total_loss += torch.nn.functional.cross_entropy(
              output,
              target,
              reduction='sum'
            ).item()

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            
    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size
 
    return acc, total_l

def valid_model(model, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    dataset_size = 0

    for batch_id, batch in enumerate(data_loader):
        data, target = batch
        dataset_size += data.size()[0]
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            
    acc = 100.0 * (float(correct) / float(dataset_size))
 
    return acc, int(correct), int(dataset_size)

def discord_stale(conf, e, recieved_stamp):
    active_stamp = recieved_stamp[:conf["k"]] 

    decay_stamp = []
    for s in active_stamp:
        s_list = s.split(".pt")
        s_list = s_list[0].split("_")
        s_num = int(s_list[-1])
        decay_stamp.append(e-s_num)

    with open('aflpureAvg_'+conf["model_name"]+'_'+conf["type"]+'_staleness_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([e, decay_stamp, sum(decay_stamp)])


def aggregate_model_tier(global_model, recieved_model, conf, e):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    active_recieved = recieved_model[:conf["k"]]

    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    for name, data in global_model.state_dict().items():
        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            update_layer = (gra_state[name] / conf["k"]) 
            global_gradient[name] += update_layer
            
        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass

        data.copy_(global_gradient[name])

    return global_model

def aggregate_model(global_model, recieved_model, conf, e, Tiers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    active_recieved = recieved_model[:conf["k"]]

    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    weight_t = []
    sum_update = 0 
    for tier, infos in Tiers.items():
        sum_update += infos[1]
    for tier, infos in Tiers.items():
        weight_t.append(infos[1]/sum_update)

    print("Weight: ", weight_t)

    for name, data in global_model.state_dict().items():
        for tier, infos in Tiers.items():
            gra.load_state_dict(torch.load(infos[0]))
            gra_state = gra.state_dict()
            update_layer = (gra_state[name] * weight_t[4 - tier]) 
            global_gradient[name] += update_layer
            
        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass

        data.copy_(global_gradient[name])

    return global_model

def train_model(model, optimizer, data_loader, conf, seq):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    gra_dict = {}
    for name, data in model.state_dict().items():
        gra_dict[name] = model.state_dict()[name].clone()
    
    for e in range(conf["local_epochs"]):
        for batch_id, batch in enumerate(data_loader):
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss = loss + l2_regularization(model, 0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf["clip"])
            optimizer.step()

            if batch_id % 10 == 0:
                print("\t \t Finish ", batch_id, "/", len(data_loader), "batches.")
        
        print("\t Client", seq, " finsh ", e, " epoches train! ")

    torch.save(model.state_dict(), "./fedat/model_" + str(seq) + ".pt")
    for k, v in model.state_dict().items():
        gra_dict[k] = v - gra_dict[k]
    torch.save(gra_dict, "./fedat/gradient_" + str(seq) + ".pt")

    return model

def angle_between(v1, v2):
    dot = torch.dot(v1, v2)
    mag1 = torch.norm(v1)
    mag2 = torch.norm(v2)
    
    angle = dot / (mag1 * mag2)
    theta = torch.arccos(angle)/torch.pi*180
    
    return theta, angle
    
def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)
 
def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)

def main():
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    eval_loader, eval_loader_list = get_dataset("../data/", conf["type"], "s", conf, -1)

    workers = conf["no_models"] 
    freq_for_workers = [0]*conf["no_models"]
    speed_for_workers = [0]*conf["no_models"]
    theta_for_workers = [0]*conf["no_models"]
    feedback_for_workers = [0]*conf["no_models"]
    lr_for_workers = [conf["local_lr"]]*conf["no_models"]
    moment_for_workers = [conf["local_momentum"]]*conf["no_models"]
    Tiers = {0:["./fedat/tier_model_0.pt", 0, []], 1:["./fedat/tier_model_1.pt", 0, []], 2:["./fedat/tier_model_2.pt", 0, []]
            , 3:["./fedat/tier_model_3.pt", 0, []], 4:["./fedat/tier_model_4.pt", 0, []]}

    worker_conf = {} 

    csv_reader = csv.reader(open('./resources_No_100_max_50.csv'))
    for row in csv_reader:
        r = row

    for i in range(len(r)):
        r[i] = int(r[i])
    print(r)

    for i in range(workers):
        resource = r[i]
        print("Client ", i, " has ", resource, " resource.")
        time.sleep(0.5)
        loader, val_loader_list, train_size = get_dataset("../data/", conf["type"], "c", conf, i)
        worker_conf[i] = [resource, loader, 0, 0, "./fedat/global_model_0.pt", val_loader_list]

    # workflow
    global_epoch = 0
    have_recieved_model = []
    have_recieved_stamp = []
    time_clock = 0
    uploaded_model = 0
    theta_ths = 0
    speed_ths = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if conf["model_name"] == "resnet18":
        global_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        tier_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        global_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
        tier_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        global_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        tier_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        global_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        tier_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass
    for name, value in tier_model.state_dict().items():
        tier_model.state_dict()[name] = torch.zeros_like(value)
    torch.save(global_model.state_dict(), "./fedat/global_model_0.pt")
    torch.save(tier_model.state_dict(), "./fedat/tier_model_0.pt")
    torch.save(tier_model.state_dict(), "./fedat/tier_model_1.pt")
    torch.save(tier_model.state_dict(), "./fedat/tier_model_2.pt")
    torch.save(tier_model.state_dict(), "./fedat/tier_model_3.pt")
    torch.save(tier_model.state_dict(), "./fedat/tier_model_4.pt")

    # start federated tasks
    start_time = time.time()
    standard_time = 0
    while global_epoch < conf["global_epochs"]:

        print("\nGlobal Epoch ", global_epoch, " Starts! \n")

        for client_seq_number in range(workers):
            print("Waiting for ", client_seq_number, " to train.")
            resour = worker_conf[client_seq_number][0]
            time_stamp = worker_conf[client_seq_number][2]

            if time_stamp == 0:
                
                print("\t Client ", client_seq_number, "start train!")
                train_loader = worker_conf[client_seq_number][1]
                using_train_model =  worker_conf[client_seq_number][4]
                using_train_epoch = int(using_train_model.split("_")[-1].replace(".pt", ""))
                worker_conf[client_seq_number][3] = worker_conf[client_seq_number][4] 
                if conf["model_name"] == "resnet18":
                    local_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
                elif conf["model_name"] == "vgg16":
                    local_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                elif conf["model_name"] == "CNN":
                    local_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                elif conf["model_name"] == "LSTM":
                    local_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                else:
                    pass
                local_model.load_state_dict(torch.load(using_train_model))

                learning_r = lr_for_workers[client_seq_number]
                moment = moment_for_workers[client_seq_number]
                feedback_for_workers[client_seq_number] = 0
                optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_r, momentum=moment)

                start_standard_time = time.time()
                local_model = train_model(local_model, optimizer, train_loader, conf, client_seq_number)
                end_standard_time = time.time()
                standard_time = max(standard_time, end_standard_time-start_standard_time)
                
                lr_for_workers[client_seq_number] = learning_r
                moment_for_workers[client_seq_number] = moment

                worker_conf[client_seq_number][2] += 1


            elif time_stamp == resour:
                print("Client ", client_seq_number, "finish train and upload gradient!")
                gra =  "./fedat/model_" + str(client_seq_number) + ".pt"
                if global_epoch > 30:
                    update_to_tier = (resour-1) // 10
                    Tiers[update_to_tier][1] += 1
                    Tiers[update_to_tier][2].append([client_seq_number, gra])          
                else:
                    have_recieved_model.append([client_seq_number, gra])          
                worker_conf[client_seq_number][2] = 0                      
                uploaded_model += 1
            else:
                print("Client ", client_seq_number, "keep training!")
                time.sleep(standard_time)
                worker_conf[client_seq_number][2] += 1
        
        time_clock += 1
        recieved_amount = len(have_recieved_model)
        print("\nUsing ", time_clock, " time clocks and recieve ", recieved_amount, " models! \n")

        time.sleep(0.5)
        if global_epoch > 30:
            for t, infos in Tiers.items():
                recieved_amount = len(infos[2])

                if recieved_amount < conf["k"]:
                    print("Waiting for enough models! Need ", conf["k"], ", but recieved ", recieved_amount)  
                else:
                    print("Having recieved enough models. Need ", conf["k"], ", and recieved ", recieved_amount)
                    tier_model.load_state_dict(torch.load(infos[0]))
                    tier_model = aggregate_model_tier(tier_model, infos[2], conf, global_epoch) 
                    torch.save(tier_model.state_dict(), infos[0])
                    total_acc, total_loss = eval_model(tier_model, eval_loader) 
                    print("Global Epoch ", global_epoch, "\t Tier loss: ", total_loss, " \t Tier acc: ", total_acc)
                    global_model = aggregate_model(global_model, infos[2], conf, global_epoch, Tiers) 

                    total_acc, total_loss = eval_model(global_model, eval_loader) 
                    print("Global Epoch ", global_epoch, "\t total loss: ", total_loss, " \t total acc: ", total_acc)
                    
                    infos[2] = infos[2][conf["k"]:]
                    torch.save(global_model.state_dict(), "./fedat/global_model_"+str(global_epoch)+".pt")

                    for client_seq_number in range(workers):
                        worker_conf[client_seq_number][4] = './fedat/global_model_'+str(global_epoch)+'.pt'

                    print("Finish aggregate and leave ", len(infos[2]), " models!")

                    theta_ths = 0
                    for t in theta_for_workers:
                        if t > 0:
                            theta_ths +=1
                    if sum(theta_for_workers) > 0:
                        theta_ths = sum(theta_for_workers)/theta_ths
                    else:
                        theta_ths = sum(theta_for_workers)

                    speed_ths = 0
                    for t in speed_for_workers:
                        if t > 0:
                            speed_ths +=1
                    if sum(speed_for_workers) > 0:
                        speed_ths = sum(speed_for_workers)/speed_ths
                    else:
                        speed_ths = sum(speed_for_workers)

                    this_time = time.time()
                    with open('aflfedatAvg_'+conf["model_name"]+'_'+conf["type"]+'_acc_with'+'_alpha_'+str(conf["alpha"])+'_clip_'+str(conf["clip"])+'.csv', mode='a+', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([global_epoch, total_acc, total_loss, this_time - start_time,theta_ths, speed_ths,theta_for_workers, speed_for_workers, feedback_for_workers, freq_for_workers, lr_for_workers, moment_for_workers])

                    with open('aflfedatAvg_'+conf["model_name"]+'_'+conf["type"]+'_size_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([global_epoch, uploaded_model])
                    global_epoch += 1
        else:
            if recieved_amount < conf["k"]:
                print("Waiting for enough models! Need ", conf["k"], ", but recieved ", recieved_amount) 
            else:
                print("Having recieved enough models. Need ", conf["k"], ", and recieved ", recieved_amount)
                global_model = aggregate_model_tier(global_model, have_recieved_model, conf, global_epoch) 

                total_acc, total_loss = eval_model(global_model, eval_loader) 
                print("Global Epoch ", global_epoch, "\t total loss: ", total_loss, " \t total acc: ", total_acc)
                
                have_recieved_model = have_recieved_model[conf["k"]:]
                torch.save(global_model.state_dict(), "./total/global_model_"+str(global_epoch)+".pt")
                for t, infos in Tiers.items():
                    torch.save(global_model.state_dict(), infos[0])
                for client_seq_number in range(workers):
                    worker_conf[client_seq_number][4] = './total/global_model_'+str(global_epoch)+'.pt'

                print("Finish aggregate and leave ", len(have_recieved_model), " models!")
                
                theta_ths = 0
                for t in theta_for_workers:
                    if t > 0:
                        theta_ths +=1
                if sum(theta_for_workers) > 0:
                    theta_ths = sum(theta_for_workers)/theta_ths
                else:
                    theta_ths = sum(theta_for_workers)

                speed_ths = 0
                for t in speed_for_workers:
                    if t > 0:
                        speed_ths +=1
                if sum(speed_for_workers) > 0:
                    speed_ths = sum(speed_for_workers)/speed_ths
                else:
                    speed_ths = sum(speed_for_workers)

                this_time = time.time()
                with open('aflfedatAvg_'+conf["model_name"]+'_'+conf["type"]+'_acc_with'+'_alpha_'+str(conf["alpha"])+'_clip_'+str(conf["clip"])+'.csv', mode='a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([global_epoch, total_acc, total_loss, this_time - start_time,theta_ths, speed_ths,theta_for_workers, speed_for_workers, feedback_for_workers, freq_for_workers, lr_for_workers, moment_for_workers])

                with open('aflfedatAvg_'+conf["model_name"]+'_'+conf["type"]+'_size_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([global_epoch, uploaded_model])
                global_epoch += 1
                

        time.sleep(0.5)

if __name__ == "__main__":
    main()
