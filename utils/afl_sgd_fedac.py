'''
Experiments on FedAC.

Reference:
Yu Zang, Zhe Xue, Shilong Ou, Lingyang Chu, Junping Du, and Yunfei Long. 2024. Efficient Asynchronous Federated Learning with Prospective Momentum Aggregation and Fine-Grained Correction. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 16642–1665
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
BETA1 = 0.1
BETA2 = 0.1
import os
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)   

mkdir("./fedac")

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

    with open('aflpureSGD_'+conf["model_name"]+'_'+conf["type"]+'_staleness_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([e, decay_stamp, sum(decay_stamp)])

def aggregate_model(global_model, recieved_model, conf, e, freq_for_workers,recieved_stamp):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    active_recieved = recieved_model[:conf["k"]]
    active_stamp = recieved_stamp[:conf["k"]]
    staleness = []
    p_weight = [1]*len(active_stamp)

    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        last_t_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        last_tau_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        last_m_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        last_v_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        global_c_state = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
        last_t_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
        last_tau_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
        last_m_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
        last_v_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
        global_c_state = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        last_t_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        last_tau_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        last_m_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        last_v_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        global_c_state = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        last_t_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        last_tau_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        last_m_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        last_v_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        global_c_state = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    for i in range(len(active_stamp)):
        active_stamp[i] = int(active_stamp[i].split("_")[-1].replace(".pt", ""))
    for stamp in active_stamp:
        staleness.append(e - stamp)
    if e > 0:
        last_t_model.load_state_dict(torch.load("./fedac/global_model_"+str(e-1)+".pt"))
    print(e, "Staleness: ", staleness)
    for i in range(len(active_stamp)):
        if staleness[i] > 1:
            thetas = []
            gra_way = active_recieved[i]
            gra.load_state_dict(torch.load(gra_way[1]))
            last_tau_model.load_state_dict(torch.load("./fedac/global_model_"+str(active_stamp[i])+".pt"))

            for name, value in last_t_model.state_dict().items():
                conv1_weight_a = last_t_model.state_dict()[name].view(-1) - last_tau_model.state_dict()[name].view(-1)
                conv1_weight_b = gra.state_dict()[name].view(-1) - last_tau_model.state_dict()[name].view(-1)
                if name.find("weight") != -1:
                    if conv1_weight_b.all() == 0:
                        continue
                    elif conv1_weight_a.all() == 0:
                        continue
                    theta_a, cos_a = angle_between(conv1_weight_a, conv1_weight_b)
                    thetas.append(cos_a.cpu())
                elif name.find("bias") != -1:
                    if conv1_weight_b.all() == 0:
                        continue
                    elif conv1_weight_a.all() == 0:
                        continue
                    else:
                        theta_a, cos_a = angle_between(conv1_weight_a, conv1_weight_b)
                        thetas.append(cos_a.cpu())
            if len(thetas) > 0:
                p_weight[i] = np.mean(thetas)

        else:
            p_weight[i] = 1
    for i in range(len(p_weight)):
        if not p_weight[i] < 1:
            p_weight[i] = 1/conf["k"]
        elif p_weight[i] < 0:
            p_weight[i]  = -p_weight[i]
    for i in range(len(p_weight)):
        p_weight[i] /= sum(p_weight)

    print(e, "Weight: ", p_weight)

    for gra_way in active_recieved:
        freq_for_workers[gra_way[0]] += 1

    for name, data in global_model.state_dict().items():
        for i in range(len(active_recieved)):
            gra_way = active_recieved[i]
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            update_layer = (gra_state[name] * p_weight[i]) 
            global_gradient[name] += update_layer
            
        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.add_(global_gradient[name])

    torch.save(global_gradient, "./fedac/global_gradient_"+str(e)+".pt")

    print("Finish Update global gradient")
    
    if e == 0:
        for name, data in global_gradient.items():
            last_m_model.state_dict()[name].copy_(data**2)
            last_v_model.state_dict()[name].copy_(data**2)
        torch.save(last_m_model.state_dict(), "./fedac/global_moment_"+str(e)+".pt")
        torch.save(last_v_model.state_dict(), "./fedac/global_vector_"+str(e)+".pt")
        gra.load_state_dict(torch.load("./fedac/global_gradient_0.pt"))
    else:
        last_m_model.load_state_dict(torch.load("./fedac/global_moment_"+str(e-1)+".pt"))
        last_v_model.load_state_dict(torch.load("./fedac/global_vector_"+str(e-1)+".pt"))
        gra.load_state_dict(torch.load("./fedac/global_gradient_"+str(e)+".pt"))
    for name, data in gra.state_dict().items():
        last_m_model.state_dict()[name].copy_(BETA1*last_m_model.state_dict()[name] + (1 - BETA1)*data)
        last_v_model.state_dict()[name].copy_(BETA2*last_v_model.state_dict()[name] + (1 - BETA2)*(data**2))

    torch.save(last_m_model.state_dict(), "./fedac/global_moment_"+str(e)+".pt")
    torch.save(last_v_model.state_dict(), "./fedac/global_vector_"+str(e)+".pt")

    

    for name, data in global_model.state_dict().items():
        update_moment = last_m_model.state_dict()[name]/((last_v_model.state_dict()[name])**0.5 + 1e-8)

        if data.type() != update_moment.type():
            update_moment = torch.round(update_moment).to(torch.int64)
        else:
            pass

    if e > 0:
        global_c_state.load_state_dict(torch.load("./fedac/global_cstate_"+str(e-1)+".pt"))
        torch.save(global_c_state.state_dict(), "./fedac/global_cstate_"+str(e)+".pt")
    return global_model, freq_for_workers


def train_model(model, optimizer, data_loader, conf, seq, global_staleness_cstate, local_c_state):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    gra_dict = {}
    c_hat = {}
    for name, data in model.state_dict().items():
        gra_dict[name] = model.state_dict()[name].clone()
        c_hat[name] = local_c_state.state_dict()[name].clone()
    
    for e in range(conf["local_epochs"]):
        for batch_id, batch in enumerate(data_loader):
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf["clip"])
            optimizer.step()

            if batch_id % 10 == 0:
                print("\t \t Finish ", batch_id, "/", len(data_loader), "batches.")
        
        print("\t Client", seq, " finsh ", e, " epoches train! ")

    for k, v in model.state_dict().items():
        model.state_dict()[k] = v - conf["local_lr"]*(global_staleness_cstate.state_dict()[k] - local_c_state.state_dict()[k])

    for k, v in model.state_dict().items():
        gra_dict[k] = v - gra_dict[k]

    for k, v in local_c_state.state_dict().items():
        local_c_state.state_dict()[k] = (1/(conf["k"]*conf["local_lr"]))*gra_dict[k] - (global_staleness_cstate.state_dict()[k] - v)
        c_hat[k] = local_c_state.state_dict()[k] - c_hat[k]
    torch.save(gra_dict, "./fedac/gradient_" + str(seq) + ".pt")
    torch.save(c_hat, "./fedac/c_state_gradient_" + str(seq) + ".pt")
    torch.save(local_c_state.state_dict(), "./fedac/local_cstate_"+str(seq)+".pt")

    return model

def angle_between(v1, v2):
    dot = torch.dot(v1, v2)

    mag1 = torch.norm(v1)
    mag2 = torch.norm(v2)
    
    angle = dot / (mag1 * mag2)
    theta = torch.arccos(angle)/torch.pi*180
    
    return theta, angle
    
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
        worker_conf[i] = [resource, loader, 0, 0, "./fedac/global_model_0.pt", val_loader_list]

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
        global_c_state = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        global_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
        global_c_state = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        global_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        global_c_state = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        global_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        global_c_state = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass
    torch.save(global_model.state_dict(), "./fedac/global_model_0.pt")
    for k, v in global_c_state.state_dict().items():
        global_c_state.state_dict()[k] = torch.zeros_like(v)
    torch.save(global_c_state.state_dict(), "./fedac/global_cstate_0.pt")

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
                    local_c_state = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
                    global_staleness_cstate = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
                elif conf["model_name"] == "vgg16":
                    local_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                    local_c_state = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                    global_staleness_cstate = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                elif conf["model_name"] == "CNN":
                    local_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                    local_c_state = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                    global_staleness_cstate = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                elif conf["model_name"] == "LSTM":
                    local_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                    local_c_state = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                    global_staleness_cstate = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                else:
                    pass
                local_model.load_state_dict(torch.load(using_train_model))

                if global_epoch == 0:
                    local_c_state.load_state_dict(torch.load("./fedac/global_cstate_0.pt"))
                    torch.save(local_c_state.state_dict(), "./fedac/local_cstate_"+str(client_seq_number)+".pt")
                else:
                    local_c_state.load_state_dict(torch.load("./fedac/local_cstate_"+str(client_seq_number)+".pt"))
                global_staleness_cstate.load_state_dict(torch.load("./fedac/global_cstate_"+str(using_train_epoch)+".pt"))

                learning_r = lr_for_workers[client_seq_number]
                moment = moment_for_workers[client_seq_number]
                feedback_for_workers[client_seq_number] = 0
                optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_r, momentum=moment)
                start_standard_time = time.time()
                local_model = train_model(local_model, optimizer, train_loader, conf, client_seq_number)
                end_standard_time = time.time()
                standard_time = max(standard_time, end_standard_time-start_standard_time)

                worker_conf[client_seq_number][2] += 1


            elif time_stamp == resour:
                print("Client ", client_seq_number, "finish train and upload gradient!")
                gra =  "./fedac/gradient_" + str(client_seq_number) + ".pt"
                have_recieved_model.append([client_seq_number, gra])     
                have_recieved_stamp.append(worker_conf[client_seq_number][3])
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

        if recieved_amount < conf["k"]:
            print("Waiting for enough models! Need ", conf["k"], ", but recieved ", recieved_amount) 
        else:
            print("Having recieved enough models. Need ", conf["k"], ", and recieved ", recieved_amount)
            discord_stale(conf, global_epoch, have_recieved_stamp)
            global_model, freq_for_workers = aggregate_model(global_model, have_recieved_model, conf, global_epoch, freq_for_workers,have_recieved_stamp) 

            total_acc, total_loss = eval_model(global_model, eval_loader) 
            print("Global Epoch ", global_epoch, "\t total loss: ", total_loss, " \t total acc: ", total_acc)
            
            have_recieved_model = have_recieved_model[conf["k"]:]
            have_recieved_stamp = have_recieved_stamp[conf["k"]:]
            torch.save(global_model.state_dict(), "./fedac/global_model_"+str(global_epoch)+".pt")

            for client_seq_number in range(workers):
                worker_conf[client_seq_number][4] = './fedac/global_model_'+str(global_epoch)+'.pt'

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
            with open('aflfedacSGD_'+conf["model_name"]+'_'+conf["type"]+'_acc_with'+'_alpha_'+str(conf["alpha"])+'_clip_'+str(conf["clip"])+'.csv', mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([global_epoch, total_acc, total_loss, this_time - start_time,theta_ths, speed_ths,theta_for_workers, speed_for_workers, feedback_for_workers, freq_for_workers, lr_for_workers, moment_for_workers])

            with open('aflfedacSGD_'+conf["model_name"]+'_'+conf["type"]+'_size_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([global_epoch, uploaded_model])
            global_epoch += 1
            

        time.sleep(0.5)

if __name__ == "__main__":
    main()
