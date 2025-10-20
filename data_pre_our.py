'''
Prepare dataloader for training
Supporting CV tasks, NLP tasks, and Real-World Tasks

CV tasks: 
    Donwload dataset from public resouce by PyTorch automatically.

NLP tasks: 
    The raw dataset is 'The Complete Works of William Shakespeare', and we process it to a trainable dataset.
    We dealt with the mapping of characters to scripts and store this mapping in 'all_data.json'.
    Please make sure the file 'all_data.json' in the correct path. (You can modify the path in line 68.)

Real World tasks:
    The raw dataset is the US Adult Income Dataset, which you can find from http://www.census.gov/ftp/pub/DES/www/welcome.html.
    We have pre-processed the raw dataset and store them in 'adult_train_x.pt', adult_train_y.pt', 'adult_test_x.pt', and 'adult_test_y.pt".
    You can find them in the folder 'distribution'.
'''


from torchvision import datasets, transforms
import numpy as np
import torch
import random
import json
import csv

def get_dataset(dir, name, roll, conf, user_id):

    if torch.cuda.is_available():
        pin = True
    else:
        pin = False

    # For NLP tasks
    if name == 'Shakespeare':
        class Mydataset(torch.utils.data.Dataset):

            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.idx = list()
                for item in x:
                    self.idx.append(item)
                pass

            def __getitem__(self, index):
                input_data = self.idx[index]
                target = self.y[index]
                return input_data, target

            def __len__(self):
                return len(self.idx)

        ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        NUM_LETTERS = len(ALL_LETTERS)

        def word_to_indices(word):
            indices = []
            for c in word:
                indices.append(ALL_LETTERS.find(c))
            return indices


        def letter_to_vec(letter):
            index = ALL_LETTERS.find(letter)
            return index

        with open("../Shake/all_data.json", "r+") as f:
            data_conf = json.load(f)
        char = data_conf["users"]
        plays = data_conf["hierarchies"]


        train_data_x = []
        train_data_y = []
        val_data_x = []
        val_data_y = []
        test_data_x  = []
        test_data_y  = []
        if conf["non_iid"] == "iid":
            if roll == "s":
                for i in range(200):
                    uid = np.random.randint(0,1129)
                    usr = char[uid]
                    x = data_conf["user_data"][usr]["x"]
                    y = data_conf["user_data"][usr]["y"]
                    data_x = [word_to_indices(word) for word in x]
                    data_y = [letter_to_vec(c) for c in y]

                    test_partition = int(len(data_x) * 0.8)

                    test_data_x += data_x[test_partition:]
                    test_data_y += data_y[test_partition:]

                test_data_x = torch.LongTensor(test_data_x)
                test_data_y = torch.LongTensor(test_data_y)

                test_dataset = Mydataset(test_data_x, test_data_y) 
                test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=conf["batch_size"],shuffle=True, pin_memory = pin)
                test_list = {1:test_loader}

                return test_loader, test_list
            else:
                print("Process the data of ", user_id)
                uid_list = set()
                uid1 = np.random.randint(0,1129)
                uid_list.add(uid1)
                usr1 = char[uid1]
                x = data_conf["user_data"][usr1]["x"] 
                y = data_conf["user_data"][usr1]["y"] 
                for i in range(10):
                    uid = np.random.randint(0,1129)
                    while uid in uid_list:
                        uid = np.random.randint(0,1129)
                    usr = char[uid]
                    x += data_conf["user_data"][usr]["x"] 
                    y += data_conf["user_data"][usr]["y"] 

                data_x = [word_to_indices(word) for word in x]
                data_y = [letter_to_vec(c) for c in y]

                train_partition = int(len(data_x) * 0.8)

                train_data_x += data_x[:train_partition]
                train_data_y += data_y[:train_partition]

                train_data_x = torch.LongTensor(train_data_x)
                train_data_y = torch.LongTensor(train_data_y)

                train_dataset = Mydataset(train_data_x, train_data_y)
                print(user_id, "has data of ", len(train_data_x))
                return torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True)
        else:
            sample_user = torch.load("../Shake/Shakespear_train_"+str(conf["alpha"])+"_frac_08.pt")
            rest_user = torch.load("../Shake/Shakespear_test_"+str(conf["alpha"])+"_frac_08.pt")
            if roll == "s":
                for i in range(len(sample_user)):
                    usr = char[sample_user[i]]
                    x = data_conf["user_data"][usr]["x"]
                    y = data_conf["user_data"][usr]["y"]
                    data_x = [word_to_indices(word) for word in x]
                    data_y = [letter_to_vec(c) for c in y]

                    test_partition = int(len(data_x) * 0.8)

                    test_data_x += data_x[test_partition:]
                    test_data_y += data_y[test_partition:]

                for i in range(len(rest_user)):
                    usr = char[rest_user[i]]
                    x = data_conf["user_data"][usr]["x"]
                    y = data_conf["user_data"][usr]["y"]
                    data_x = [word_to_indices(word) for word in x]
                    data_y = [letter_to_vec(c) for c in y]

                    test_partition = int(len(data_x) * 0.8)

                    test_data_x += data_x[test_partition:]
                    test_data_y += data_y[test_partition:]

                test_data_x = torch.LongTensor(test_data_x)
                test_data_y = torch.LongTensor(test_data_y)

                test_dataset = Mydataset(test_data_x, test_data_y) 

                loader = torch.utils.data.DataLoader(test_dataset,batch_size=conf["batch_size"],shuffle=True, pin_memory = pin)
                test_list = {1:loader}

                return loader, test_list
            else:
                print("Process the data of ", user_id)
                usr1 = char[sample_user[user_id]]
                usr2 = char[sample_user[user_id + conf["no_models"]]]
                x = data_conf["user_data"][usr1]["x"] + data_conf["user_data"][usr2]["x"]
                y = data_conf["user_data"][usr1]["y"] + data_conf["user_data"][usr2]["y"]

                data_x = [word_to_indices(word) for word in x]
                data_y = [letter_to_vec(c) for c in y]
                train_partition = int(len(data_x) * 0.8)

                train_data_x += data_x[:train_partition]
                train_data_y += data_y[:train_partition]

                train_data_x = torch.LongTensor(train_data_x)
                train_data_y = torch.LongTensor(train_data_y)

                train_dataset = Mydataset(train_data_x, train_data_y)

                val_data_x += data_x[:50]
                val_data_y += data_y[:50]

                val_data_x = torch.LongTensor(val_data_x)
                val_data_y = torch.LongTensor(val_data_y)

                val_dataset = Mydataset(val_data_x, val_data_y)
                print(user_id, "has data of ", len(train_data_x))
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True)
                val_list = {1:torch.utils.data.DataLoader(val_dataset, batch_size=1, pin_memory = pin,drop_last=True)}
                return train_loader, val_list, len(train_data_x)


    # For different CV tasks, prepare dataset
    elif name == 'femnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.EMNIST(dir, train=True, download=True, transform=transform_train,split = 'byclass' )
        eval_dataset = datasets.EMNIST(dir, train=False, transform=transform_test,split = 'byclass' )
    elif name == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform_train )
        eval_dataset = datasets.MNIST(dir, train=False, transform=transform_test)
    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.CIFAR100(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)
    # for RWD tasks
    elif name == "adult":
        class adultdataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.input_data = x
                self.targets = y
                self.idx = list()
                for item in x:
                    self.idx.append(item)
                pass

            def __getitem__(self, index):
                input_data = self.idx[index]
                targets = self.y[index]
                return input_data, targets

            def __len__(self):
                return len(self.idx)
        train_data_x = torch.load("../adult/adult_train_x.pt")
        train_data_y = torch.load("../adult/adult_train_y.pt")
        test_data_x = torch.load("../adult/adult_test_x.pt")
        test_data_y = torch.load("../adult/adult_test_y.pt")
        for i in range(len(train_data_x)):
            train_data_x[i] = torch.tensor(train_data_x[i], dtype=torch.float32)
        for i in range(len(test_data_x)):
            test_data_x[i] = torch.tensor(test_data_x[i], dtype=torch.float32)
        train_dataset = adultdataset(train_data_x, train_data_y)
        eval_dataset = adultdataset(test_data_x, test_data_y)

    
    
    if roll == "s":
        if name == "adult":
            eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=conf["batch_size"],shuffle=True, drop_last=True,pin_memory = pin, num_workers=4)
            eval_loader_list = {1:eval_loader}
        else:
            label_to_indices = {}
            eval_loader_list = {}

            for i in range(len(eval_dataset)):
                label = eval_dataset.targets[i]
                if label not in label_to_indices:
                    label_to_indices[label] = []
                
                label_to_indices[label].append(i)

            
            for label, indices in label_to_indices.items():
                eval_loader_label = torch.utils.data.DataLoader(eval_dataset,batch_size=conf["batch_size"],pin_memory = pin,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
                eval_loader_list[label] = eval_loader_label
            eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=conf["batch_size"],shuffle=True, drop_last=True,pin_memory = pin, num_workers=4)
        return eval_loader, eval_loader_list
    else:
        if user_id == 0:
            if conf["redistribution"] == "y":
                if name == "adult":
                    Data_partition = []
                    if conf["non_iid"] == "race":
                        white_idx = []
                        black_idx = []
                        Asian_idx = []
                        Amer_idx = []
                        Other_idx = []
                        for i in range(len(train_dataset)):
                            race = train_dataset[i][0][6]
                            if race == 0:
                                white_idx.append(i)
                            elif race == 1:
                                black_idx.append(i)
                            elif race == 2:
                                Asian_idx.append(i)
                            elif race == 3:
                                Amer_idx.append(i)
                            else:
                                Other_idx.append(i)
                        white_rate = len(black_idx) / len(train_dataset)
                        black_rate = len(white_idx) / len(train_dataset) + white_rate
                        Asian_rate = len(Asian_idx) / len(train_dataset) + black_rate
                        Amer_rate = len(Amer_idx) / len(train_dataset) + Asian_rate
                        Other_rate = len(Other_idx) / len(train_dataset) + Amer_rate
                        y0 = np.random.lognormal(len(train_dataset)/10000, conf["alpha"], conf["no_models"])
                        for i in range(conf["no_models"]):
                            type_random = random.random()
                            if type_random < white_rate:
                                data_indices = random.sample(white_idx, int((y0[i]/sum(y0))*len(white_idx)))
                            elif type_random < black_rate:
                                data_indices = random.sample(black_idx, int((y0[i]/sum(y0))*len(black_idx)))
                            elif type_random < Asian_rate:
                                data_indices = random.sample(Asian_idx, int((y0[i]/sum(y0))*len(Asian_idx)))
                            elif type_random < Amer_rate:
                                data_indices = random.sample(Amer_idx, int((y0[i]/sum(y0))*len(Amer_idx)))
                            else:
                                data_indices = random.sample(list(range(0, len(train_dataset))), int((y0[i]/sum(y0))*len(train_dataset)))
                            Data_partition.append(data_indices)
                    elif conf["non_iid"] == "sex":
                        male_idx = []
                        female_idx = []
                        for i in range(len(train_dataset)):
                            sex = train_dataset[i][0][7]
                            if sex == 0:
                                male_idx.append(i)
                            else:
                                female_idx.append(i)
                        y0 = np.random.lognormal(len(train_dataset)/10000, conf["alpha"], conf["no_models"])
                        for i in range(conf["no_models"]):
                            type_random = random.random()
                            if type_random < conf["alpha"]:
                                data_indices = random.sample(male_idx, int((1 - conf["alpha"])*(y0[i]/sum(y0))*len(male_idx)))
                                data_indices += random.sample(female_idx, int((conf["alpha"])*(y0[i]/sum(y0))*len(female_idx)))
                            else:
                                data_indices = random.sample(female_idx, int((1 - conf["alpha"])*(y0[i]/sum(y0))*len(female_idx)))
                                data_indices += random.sample(male_idx, int((conf["alpha"])*(y0[i]/sum(y0))*len(male_idx)))
                            Data_partition.append(data_indices)

                    elif conf["non_iid"] == "iid":
                        for i in range(conf["no_models"]):
                            data_indices = random.sample(list(range(0, len(train_dataset))), int(conf["alpha"]*len(train_dataset)))
                            Data_partition.append(data_indices)
                        else:
                            pass
                    torch.save(Data_partition, "../bias/data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
                else:
                    label_to_indices = {}
                    eval_loader_list = {}

                    for i in range(len(train_dataset)):
                        label = int(train_dataset.targets[i])
                        if label not in label_to_indices:
                            label_to_indices[label] = []
                        label_to_indices[label].append(i)

                    Data_partition = {}
                    if conf["non_iid"] == "HeteroDiri":
                        partition = np.random.dirichlet([conf["alpha"]]*conf["no_models"], len(label_to_indices))
                        label_partition = {}
                        for label, indices in label_to_indices.items():
                            label_partition[label] = partition[label]

                        label_indices = {}
                        for label, indices in label_to_indices.items():
                            label_indices[label] = np.arange(0, len(indices), 1)

                        Dataset_label_indices = {}
                        for label, parti in label_partition.items():
                            par = []
                            for p in parti:
                                par.append(round(p * len(label_indices[label])))
                            par = np.cumsum(par)
                            par[-1] = len(label_indices[label])
                            Dataset_label_indices[label] = np.split(label_indices[label], par)

                        for label in label_partition:
                            Data_partition[label] = []
                            for indices in Dataset_label_indices[label]:
                                data_indices = []
                                for i in indices:
                                    data_indices.append(label_to_indices[label][i])
                                Data_partition[label].append(data_indices)

                    elif conf["non_iid"] == "Shards":
                        squence_indeices = []
                        for label, indices in label_to_indices.items():
                            for i in indices:
                                squence_indeices.append(i)

                        squence = list(range(len(squence_indeices)))

                        step = int(len(squence)/(conf["no_models"] * conf["alpha"]))
                        shards = [squence[i:i+step] for i in range(0,len(squence),step)]

                        len_shards = list(range(len(shards)))

                        for i in range(conf["no_models"]):
                            id = random.sample(len_shards, conf["alpha"])
                            i_data = []
                            for j in range(conf["alpha"]):
                                i_data += shards[id[j]]
                                len_shards.remove(id[j])
                            s_data = []
                            for s in i_data:
                                s_data.append(squence_indeices[s])
                            Data_partition[i] = s_data

                    elif conf["non_iid"] == "QuanSkew":
                        labels = list(range(len(label_to_indices)))

                        for i in range(conf["no_models"]):
                            client_labels = random.sample(labels, int(conf["alpha"]*len(labels)))
                            client_data = []
                            for cl in client_labels:
                                client_data += label_to_indices[cl]

                            Data_partition[i] = client_data
                    elif conf["non_iid"] == "Unbalance_Diri":
                        n_class = conf["CLASS_NUM"]
                        partition = np.random.dirichlet([50]*n_class, 1) 
                        a = partition[0]
                        ma = max(a)
                        diri_data = []
                        for i in range(n_class):
                            diri_data.append(int((a[i]/ma)*len(label_to_indices[i])))
                        y0 = np.random.lognormal(len(train_dataset)/conf["no_models"], conf["alpha"], conf["no_models"])
                        s = sum(diri_data)/sum(y0)
                        y = y0*s

                        point_indices = [0]*n_class
                        Data_partition = []
                        for i in range(conf["no_models"]):
                            n_label = []
                            c_partition = []
                            data_indices = []
                            for j in range(n_class):
                                n = int(a[j]*y[i])
                                n_label.append(n) 
                                c_partition.append([point_indices[j],point_indices[j]+n])
                                point_indices[j] += n
                                data_indices += label_to_indices[j][c_partition[j][0]: c_partition[j][1]] 
                            Data_partition.append(data_indices)

                    elif conf["non_iid"] == "iid":
                        Data_partition = []
                        for i in range(conf["no_models"]):
                            data_indices = random.sample(list(range(0, len(train_dataset))), int(conf["alpha"]*len(train_dataset)))
                            Data_partition.append(data_indices)
                        else:
                            pass

                    torch.save(Data_partition, "../bias/data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
            else:
                Data_partition = torch.load("../bias/data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
        else:
            Data_partition = torch.load("../bias/data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")

        train_indices = []
        if conf["non_iid"] == "HeteroDiri":
            for label, indices in Data_partition.items():
                train_indices += indices[user_id]
        elif conf["non_iid"] == "Shards":
            train_indices = Data_partition[user_id]
        elif conf["non_iid"] == "QuanSkew":
            train_indices = Data_partition[user_id]
        elif conf["non_iid"] == "Unbalance_Diri":
            train_indices = Data_partition[user_id]
        elif conf["non_iid"] == "iid":
            train_indices = Data_partition[user_id]
        else:
            train_indices = Data_partition[user_id]
        
        val_indices_list = {}
        for i in train_indices:
            label = int(train_dataset.targets[i])
            if label not in val_indices_list:
                val_indices_list[label] = [i]
            else:
                if len(val_indices_list[label]) < 20:
                    val_indices_list[label].append(i)
        
        val_loader_list = {}
        for k, v in val_indices_list.items():
            val_loader_list[k] = torch.utils.data.DataLoader(train_dataset, batch_size=1, pin_memory = pin,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(v), num_workers=4)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices), num_workers=4)
        return train_loader, val_loader_list, len(train_indices)

# test
if __name__ == "__main__":
    import json
    import time
    with open("../tconf.json", 'r', encoding='utf-8') as f:
        conf = json.load(f)

    train_loader = get_dataset("../data/", conf["type"], "c", conf, 0)
