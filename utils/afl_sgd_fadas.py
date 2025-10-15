
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
import copy

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)   

mkdir("./fadas")
BETA1 = 0.1
BETA2 = 0.1

# evaluate the model by data from data_loader
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

# discord the staleness
def discord_stale(conf, e, recieved_stamp):
    active_stamp = recieved_stamp[:conf["k"]]  #"./global_model/global_model_square_0.pt"

    decay_stamp = []
    for s in active_stamp:
        s_list = s.split(".pt")
        s_list = s_list[0].split("_")
        s_num = int(s_list[-1])
        decay_stamp.append(e-s_num)

    with open('FADAS_'+conf["model_name"]+'_'+conf["type"]+'_staleness_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([e, decay_stamp, sum(decay_stamp)])


# aggregrate the model
def aggregrate_model(global_model, recieved_model, conf, e, freq_for_workers, global_lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # bring out the first K gradient
    active_recieved = recieved_model[:conf["k"]]
    for gra_way in active_recieved:
        freq_for_workers[gra_way[0]] += 1
    # average without weight
    global_gradient = global_model.state_dict()
    
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        last_m_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        last_v_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
        max_v_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        last_m_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        last_v_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
        max_v_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        last_m_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        last_v_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
        max_v_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass
    
    
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
        if e == 0:
            last_m_model.state_dict()[name].copy_(torch.zeros_like(data).to(device).float())
            last_v_model.state_dict()[name].copy_(torch.zeros_like(data).to(device).float())
            max_v_model.state_dict()[name].copy_(torch.zeros_like(data).to(device).float())
            
    
    if e > 0:
        last_m_model.load_state_dict(torch.load("./fadas/global_moment_"+str(e-1)+".pt"))
        last_v_model.load_state_dict(torch.load("./fadas/global_vector_"+str(e-1)+".pt"))
        max_v_model.load_state_dict(torch.load("./fadas/global_maxvector_"+str(e-1)+".pt"))

    max_staleness = 0
    for gra_way in active_recieved:
        s_list = gra_way[2].split(".pt")
        s_list = s_list[0].split("_")
        s_num = int(s_list[-1])
        current_client_staleness = e-s_num
        if current_client_staleness > max_staleness:
            max_staleness = current_client_staleness

    if max_staleness > 10:
        global_lr = min(global_lr, 1/max_staleness)

    new_v_model = copy.deepcopy(last_v_model)
    for name, data in global_model.state_dict().items():
        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            update_layer = (gra_state[name] / conf["k"]) 
            global_gradient[name] += update_layer

        last_m_model.state_dict()[name].copy_(BETA1*last_m_model.state_dict()[name] + (1 - BETA1)*global_gradient[name])
        new_v_model.state_dict()[name].copy_(BETA2*last_v_model.state_dict()[name] + (1 - BETA2)*(global_gradient[name]**2))
        
        max_v_model.state_dict()[name].copy_(torch.maximum(new_v_model.state_dict()[name], max_v_model.state_dict()[name]))

        update_moment = last_m_model.state_dict()[name]/((max_v_model.state_dict()[name])**0.5 + 1e-8)
        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(update_moment).to(torch.int64)
        else:
            pass

        data.add_(global_gradient[name])


    torch.save(last_m_model.state_dict(), "./fadas/global_moment_"+str(e)+".pt")
    torch.save(last_v_model.state_dict(), "./fadas/global_vector_"+str(e)+".pt")
    torch.save(max_v_model.state_dict(), "./fadas/global_maxvector_"+str(e)+".pt")
        
    return global_model, freq_for_workers, global_lr

# train model
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf["clip"])
            optimizer.step()

            if batch_id % 10 == 0:
                print("\t \t Finish ", batch_id, "/", len(data_loader), "batches.")
        
        print("\t Client", seq, " finsh ", e, " epoches train! ")

    for k, v in model.state_dict().items():
        gra_dict[k] = v - gra_dict[k]
    torch.save(gra_dict, "./fadas/gradient_" + str(seq) + ".pt")

    return model

def angle_between(v1, v2):
    # Compute the dot product
    dot = torch.dot(v1, v2)
    
    # Compute the magnitudes of the vectors
    mag1 = torch.norm(v1)
    mag2 = torch.norm(v2)
    
    # Compute the angle between the vectors
    angle = dot / (mag1 * mag2)
    theta = torch.arccos(angle)/torch.pi*180
    
    return theta, angle
    

# main function
def main():
    # get config
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    # get evaluation dataloader for server
    eval_loader, eval_loader_list = get_dataset("../data/", conf["type"], "s", conf, -1)

    # set workers
    workers = conf["no_models"] # amount
    freq_for_workers = [0]*conf["no_models"]
    speed_for_workers = [0]*conf["no_models"]
    theta_for_workers = [0]*conf["no_models"]
    feedback_for_workers = [0]*conf["no_models"]
    lr_for_workers = [conf["local_lr"]]*conf["no_models"]
    moment_for_workers = [conf["local_momentum"]]*conf["no_models"]

    worker_conf = {} # each worker's config: number(int) : [resource(int), data_lodaer(torch.loader), time_stamp(int), global_stamp(int), newest model(str)](list)

    csv_reader = csv.reader(open('./resources_No_100_max_50.csv'))
    for row in csv_reader:
        r = row

    for i in range(len(r)):
        r[i] = int(r[i])

    # r = [1,2,3,4,5,6,7,8,9,10]
    print(r)

    for i in range(workers):
        resource = r[i]
        print("Client ", i, " has ", resource, " resource.")
        time.sleep(0.5)
        loader, val_loader_list, train_size = get_dataset("../data/", conf["type"], "c", conf, i)
        worker_conf[i] = [resource, loader, 0, 0, "./fadas/global_model_0.pt", val_loader_list]

    # workflow
    global_epoch = 0
    have_recieved_model = []
    have_recieved_stamp = []
    time_clock = 0
    uploaded_model = 0
    theta_ths = 0
    speed_ths = 0
    global_lr = conf["local_lr"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize the model
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
    torch.save(global_model.state_dict(), "./fadas/global_model_0.pt")

    # start training
    start_time = time.time()
    while global_epoch < conf["global_epochs"]:

        print("\nGlobal Epoch ", global_epoch, " Starts! \n")

        for client_seq_number in range(workers):
            print("Waiting for ", client_seq_number, " to train.")
            resour = worker_conf[client_seq_number][0]
            time_stamp = worker_conf[client_seq_number][2]

            if time_stamp == 0:
                # start train
                print("\t Client ", client_seq_number, "start train!")
                # load newest global model and dataloader
                train_loader = worker_conf[client_seq_number][1]
                using_train_model =  worker_conf[client_seq_number][4]
                using_train_epoch = int(using_train_model.split("_")[-1].replace(".pt", ""))
                worker_conf[client_seq_number][3] = worker_conf[client_seq_number][4]   # update using model
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
                if using_train_epoch != 0:
                    last_train_model = "./fadas/global_model_"+str(using_train_epoch-1)+".pt"
                    if conf["model_name"] == "resnet18":
                        last_g_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
                        last_g_gradient = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "vgg16":
                        last_g_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                        last_g_gradient = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "CNN":
                        last_g_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                        last_g_gradient = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "LSTM":
                        last_g_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                        last_g_gradient = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                    else:
                        pass
                    last_g_model.load_state_dict(torch.load(last_train_model))
                    last_g_gradient.load_state_dict(torch.load("./fadas/gradient_" + str(client_seq_number) + ".pt"))

                    for k, v in local_model.state_dict().items():
                        last_g_model.state_dict()[k] = v - last_g_model.state_dict()[k]
                    
                    thetas = []

                    for name, value in last_g_model.state_dict().items():
                        conv1_weight_a = last_g_model.state_dict()[name].view(-1)
                        conv1_weight_b = last_g_gradient.state_dict()[name].view(-1)
                        if name.find("weight") != -1:
                            if conv1_weight_b.all() == 0:
                                continue
                            elif conv1_weight_a.all() == 0:
                                continue
                            theta_a, cos_a = angle_between(conv1_weight_a, conv1_weight_b)
                            thetas.append(theta_a.cpu())
                        elif name.find("bias") != -1:
                            if conv1_weight_b.all() == 0:
                                continue
                            elif conv1_weight_a.all() == 0:
                                continue
                            else:
                                theta_a, cos_a = angle_between(conv1_weight_a, conv1_weight_b)
                                thetas.append(theta_a.cpu())

                    if len(thetas) > 0:
                        theta_for_workers[client_seq_number] = np.mean(thetas)
                    else:
                        theta_for_workers[client_seq_number] = 0
                
                # train
                learning_r = lr_for_workers[client_seq_number]
                moment = moment_for_workers[client_seq_number]
                feedback_for_workers[client_seq_number] = 0
                optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_r, momentum=moment)
                local_model = train_model(local_model, optimizer, train_loader, conf, client_seq_number)

                lr_for_workers[client_seq_number] = learning_r
                moment_for_workers[client_seq_number] = moment

                worker_conf[client_seq_number][2] += 1


            elif time_stamp == resour:
                # compute the updation
                print("Client ", client_seq_number, "finish train and upload gradient!")
                gra =  "./fadas/gradient_" + str(client_seq_number) + ".pt"
                have_recieved_model.append([client_seq_number, gra,worker_conf[client_seq_number][3]])          # update the model to server
                have_recieved_stamp.append(worker_conf[client_seq_number][3])
                worker_conf[client_seq_number][2] = 0                          # reset the time stamp of this client
                uploaded_model += 1
            else:
                print("Client ", client_seq_number, "keep training!") # keep training(idling)
                worker_conf[client_seq_number][2] += 1
        
        time_clock += 1
        recieved_amount = len(have_recieved_model)
        print("\nUsing ", time_clock, " time clocks and recieve ", recieved_amount, " models! \n")

        time.sleep(0.5)

        if recieved_amount < conf["k"]:
            print("Waiting for enough models! Need ", conf["k"], ", but recieved ", recieved_amount)  # have not recieved enough models, keep waiting
        else:
            print("Having recieved enough models. Need ", conf["k"], ", and recieved ", recieved_amount)
            # aggregrate
            discord_stale(conf, global_epoch, have_recieved_stamp)
            global_model, freq_for_workers,global_lr = aggregrate_model(global_model, have_recieved_model, conf, global_epoch, freq_for_workers,global_lr) 

            # evaluation
            total_acc, total_loss = eval_model(global_model, eval_loader) 
            print("Global Epoch ", global_epoch, "\t total loss: ", total_loss, " \t total acc: ", total_acc)
            
            # save global model and add the epoch
            have_recieved_model = have_recieved_model[conf["k"]:]
            have_recieved_stamp = have_recieved_stamp[conf["k"]:]
            torch.save(global_model.state_dict(), "./fadas/global_model_"+str(global_epoch)+".pt")

            # notice the newest global model to each client
            for client_seq_number in range(workers):
                worker_conf[client_seq_number][4] = './fadas/global_model_'+str(global_epoch)+'.pt'

            print("Finish aggregrate and leave ", len(have_recieved_model), " models!")

            # whole_freq = sum(freq_for_workers)
            # for seq in range(len(freq_for_workers)):
            #     if freq_for_workers[seq] == 0:
            #         continue
            #     else:
            #         speed_for_workers[seq] = freq_for_workers[seq] / whole_freq
                
            #     feedback_for_workers[seq] = 0

            
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
            with open('FADAS_'+conf["model_name"]+'_'+conf["type"]+'_acc_with'+'_alpha_'+str(conf["alpha"])+'_clip_'+str(conf["clip"])+'.csv', mode='a+', newline='') as file:
                writer = csv.writer(file)
                # for row in ret:
                writer.writerow([global_epoch, total_acc, total_loss, this_time - start_time,theta_ths, speed_ths,theta_for_workers, speed_for_workers, feedback_for_workers, freq_for_workers, lr_for_workers, moment_for_workers])

            with open('FADAS_'+conf["model_name"]+'_'+conf["type"]+'_size_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([global_epoch, uploaded_model])
            global_epoch += 1
            

        time.sleep(0.5)

if __name__ == "__main__":
    main()
