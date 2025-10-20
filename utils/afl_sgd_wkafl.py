'''
Experiments on WKAFL.

Reference:
Zihao Zhou, Yanan Li, Xuebin Ren, and Shusen Yang. 2022. Towards efficient and stable K-asynchronous federated learning with unbounded stale gradients on non-IID data. IEEE Transactions on Parallel and Distributed Systems 33, 12 (2022), 3291–3305.
'''

import zmq 
import os
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
import shutil
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)   

mkdir("./WKAFL")

CLASS_NUM = 100

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

def discord_stale(conf, e, recieved_stamp):
    active_stamp = recieved_stamp[:conf["k"]] 

    decay_stamp = []
    for s in active_stamp:
        s_list = s.split(".pt")
        s_list = s_list[0].split("_")
        s_num = int(s_list[-1])
        decay_stamp.append(e-s_num)

    with open('aflWKAFL_'+conf["model_name"]+'_'+conf["type"]+'_staleness_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([e, decay_stamp, sum(decay_stamp)])
    return decay_stamp

def aggregate_model(global_model, recieved_model, conf, e, decay):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    active_recieved = recieved_model[:conf["k"]]

    global_gradient = global_model.state_dict()
    predict_global_gradient = global_model.state_dict()
    
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
        predict_global_gradient[name] = torch.zeros_like(data).to(device).float()
        
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        last_global_gradient = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
        last_global_gradient = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        last_global_gradient = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        last_global_gradient = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass
    if e > 0:
        last_global_gradient.load_state_dict(torch.load("./WKAFL/global_gradient_" + str(e-1) + ".pt"))
        it = 0
        a_i = []
        p = []
        p_prime = []
        sum_loss = 0
        for s in decay:
            a_it = (torch.e/2)**(-s)
            a_i.append(a_it)
        a = sum(a_i)
        if sum_loss < conf["stage_bound"]:
            predict_norm = 0
            for name, value in predict_global_gradient.items():
                try:
                    predict_norm += torch.norm(a[name])
                except:
                    pass

        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            for name, data in gra.state_dict().items():
                if data.type() != last_global_gradient.state_dict()[name].type():
                    last_global_gradient.state_dict()[name] = torch.round(last_global_gradient.state_dict()[name]).to(torch.int64)
                else:
                    pass
                try:
                    data.add_(last_global_gradient.state_dict()[name]*conf["global_lr"])
                except:
                    data.add_((last_global_gradient.state_dict()[name]*conf["global_lr"]).long())
            torch.nn.utils.clip_grad_norm_(gra.parameters(), max_norm=conf["clip"])
            torch.save(gra.state_dict(), gra_way[1])
            for name, data in gra.state_dict().items():
                predict_global_gradient[name] += (a_i[it]/a)*data
            it += 1
        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            cos = []
            gra_norm = 0
            for name, value in gra.state_dict().items():
                if name.find("bn") == -1 & name.find("shortcut") == -1 & name.find("linear.bias") == -1:
                    value = value.reshape(value.shape[0], 1)
                    predict_global_gradient[name] = predict_global_gradient[name].reshape(value.shape[0],1)
                    ccos = torch.norm(torch.cosine_similarity(value, predict_global_gradient[name],1)+1)
                    if torch.isnan(ccos).any():
                        ccos = torch.tensor(1)
                    cos.append(ccos)
                    print("cos: ", ccos, value, predict_global_gradient[name])
                elif value.shape != torch.Size([]):
                    tmp_a = predict_global_gradient[name].reshape(1,-1)
                    tmp_b = value.reshape(1,-1)
                    ccos = torch.norm(torch.cosine_similarity(tmp_a, tmp_b,1)+1)
                    if torch.isnan(ccos).any():
                        ccos = torch.tensor(1)
                    cos.append(ccos)
                    print("cos: ", ccos, tmp_a, tmp_b)
                if name.find("bn") == -1 & name.find("shortcut") == -1 & name.find("linear.bias") == -1:
                    gra_norm += torch.norm(value)
                elif value.shape != torch.Size([]):
                    tmp = value.reshape(1,-1)
                    gra_norm += torch.norm(tmp)
            
            print(gra_norm)
            sim = sum(cos)
            
            print("sim: ", sim)
            if sim >= conf["sim_bound"]:
                p_i_prime = torch.exp(conf["beta"]*sim)
            else:
                p_i_prime = 0
            p_prime.append(p_i_prime)
            sum_loss += gra_way[2]
            print("stage_bound: ",sum_loss)
            print("p_prime: ",p_prime)
            if sum_loss < conf["stage_bound"]:
                if conf["norm_bound"]*predict_norm <= gra_norm:
                    B = conf["norm_bound"]*predict_norm / gra_norm
                    for name, data in gra.state_dict().items():
                        data.copy_(data*B)
        
        sum_prime = sum(p_prime)
        for prime in p_prime:
            p.append(prime/sum_prime)
        for name, data in global_model.state_dict().items():
            it = 0
            for gra_way in active_recieved:
                gra.load_state_dict(torch.load(gra_way[1]))
                gra_state = gra.state_dict()
                update_layer = (gra_state[name] * p[it])
                global_gradient[name] += update_layer
                it += 1
                
            if data.type() != global_gradient[name].type():
                global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
            else:
                pass

            data.add_(global_gradient[name])  
    else:
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

            data.add_(global_gradient[name])

    torch.save(global_gradient, "./WKAFL/global_gradient_" + str(e) + ".pt")
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
            loss.backward()
            optimizer.step()

            if batch_id % 10 == 0:
                print("\t \t Finish ", batch_id, "/", len(data_loader), "batches.")
        
        print("\t Client", seq, " finsh ", e, " epoches train! ")

    for k, v in model.state_dict().items():
        gra_dict[k] = v - gra_dict[k]
    torch.save(gra_dict, "./WKAFL/gradient_" + str(seq) + ".pt")
    torch.save(model.state_dict(), "./WKAFL/local_model" + str(seq) + ".pt")

    return model

def main():
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    eval_loader, eval_loader_list = get_dataset("../data/", conf["type"], "s", conf, -1)
    workers = conf["no_models"]

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
        worker_conf[i] = [resource, loader, 0, 0, "./WKAFL/global_model_0.pt"]
    global_epoch = 0
    have_recieved_model = []
    have_recieved_stamp = []
    time_clock = 0
    uploaded_model = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if conf["model_name"] == "resnet18":
        global_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        global_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        global_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        global_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass
    torch.save(global_model.state_dict(), "./WKAFL/global_model_0.pt")

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
                if using_train_model.find("global") != -1:
                    worker_conf[client_seq_number][3] = using_train_model 
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

                optimizer = torch.optim.SGD(local_model.parameters(), lr=conf['local_lr'], momentum=conf['local_momentum'])
                start_standard_time = time.time()
                local_model = train_model(local_model, optimizer, train_loader, conf, client_seq_number)
                end_standard_time = time.time()
                standard_time = max(standard_time, end_standard_time-start_standard_time)

                worker_conf[client_seq_number][2] += 1
                

            elif time_stamp == resour:
                print("Client ", client_seq_number, "finish train and upload gradient!")
                gra =  "./WKAFL/gradient_" + str(client_seq_number) + ".pt"
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
                local_model.load_state_dict(torch.load("./WKAFL/local_model" + str(client_seq_number) + ".pt"))
                total_acc, total_loss = eval_model(local_model, eval_loader) 
                
                using_global_model = torch.load(worker_conf[client_seq_number][3])
                cos_sim = 0
                cos = {}
                with open('aflWKAFL_local_acc_loss.csv', mode='a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([client_seq_number, worker_conf[client_seq_number][3], global_epoch, total_acc, total_loss, cos_sim, cos])

                have_recieved_model.append([client_seq_number, gra, total_loss])    
                have_recieved_stamp.append(worker_conf[client_seq_number][3])
                worker_conf[client_seq_number][2] = 0        
                worker_conf[client_seq_number][4] = "./WKAFL/local_model" + str(client_seq_number) + ".pt"
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
            print("Having recieved enough models. Need ", conf["k"], ", and recieved ", recieved_amount, (1/conf["CLASS_NUM"])*1.2)
            d_stamp = discord_stale(conf, global_epoch, have_recieved_stamp)
            record_model = global_model.state_dict()
            for name, data in global_model.state_dict().items():
                record_model[name] = data.clone()
            global_model = aggregate_model(global_model, have_recieved_model, conf, global_epoch, d_stamp) 

            total_acc, total_loss = eval_model(global_model, eval_loader) 
            print("Global Epoch ", global_epoch, "\t total loss: ", total_loss, " \t total acc: ")
            have_recieved_model = have_recieved_model[conf["k"]:]
            have_recieved_stamp = have_recieved_stamp[conf["k"]:]
            torch.save(global_model.state_dict(), "./WKAFL/global_model_"+str(global_epoch)+".pt")

            for client_seq_number in range(workers):
                worker_conf[client_seq_number][4] = './WKAFL/global_model_'+str(global_epoch)+'.pt'

            print("Finish aggregate and leave ", len(have_recieved_model), " models!")

        
            this_time = time.time()
            with open('aflWKAFL_'+conf["model_name"]+'_'+conf["type"]+'_acc_with'+'_alpha_'+str(conf["alpha"])+'_clip_'+str(conf["clip"])+'.csv', mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([global_epoch, total_acc, total_loss, this_time - start_time])

            with open('aflWKAFL_'+conf["model_name"]+'_'+conf["type"]+'_size_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([global_epoch, uploaded_model])
            global_epoch += 1
            

        time.sleep(1)

if __name__ == "__main__":
    main()
