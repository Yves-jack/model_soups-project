import torch
import torch.optim as optim
import numpy as np
import random
from models import LeNet5_Caffe, ResNet18
from tqdm import tqdm
import os
from copy import deepcopy
import torch.nn as nn


def set_seed(seed):
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_config(seed, model_type):
    # Generate different configurations for each model
    # {optimizer, learning rate, and epochs}.
    set_seed(seed)

    if model_type == "LeNet5-Caffe":
        optimizer = random.choice([optim.Adam, optim.RMSprop])
        lr = random.choice([0.005, 0.001, 0.0005])
        epochs = random.choice([3, 5, 8, 10])
        
        return {
            'optimizer': optimizer,
            'lr': lr,
            'epochs': epochs
        }
    elif model_type == "ResNet18":
        optimizer = random.choice([optim.SGD,optim.Adam])
        lr = random.choice([0.001, 0.0005, 0.0001])

        return {
            'optimizer': optimizer,
            'lr': lr,
            'epochs': 5
        }

def train_model(model_type, model_name, model, train_loader, config, device='cuda'): 
    criterion = nn.CrossEntropyLoss()
    if model_type == "LeNet5-Caffe":
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'])
    elif model_type == "ResNet18":
        if config["optimizer"] == optim.SGD:
            optimizer = config["optimizer"](model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = config["optimizer"](model.parameters(), lr=config['lr'], weight_decay=1e-4)
    
    os.makedirs(f'./{model_type}', exist_ok=True)
    save_path = f'./{model_type}/{model_name}.pth'

    for epoch in tqdm(range(config['epochs'])):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), save_path)
    return model

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train(device, dataset_type, train_loader, test_loader, models_num, same_init_flag = True):
    if dataset_type == "mnist":
        model_class = LeNet5_Caffe
        model_template = LeNet5_Caffe() 
        model_type = "LeNet5-Caffe"
    elif dataset_type == "cifar":
        model_class = ResNet18     
        model_template = ResNet18()
        model_type = "ResNet18"
    
    results = {
        "configs": [],
        "models": [],
        "accs": []
    }
    
    for i in range(models_num):
        config = get_config(seed=i, model_type=model_type)
        results["configs"].append(config)

        if same_init_flag:
            model = deepcopy(model_template).to(device) 
            model_name = f'{model_type}_same_init_{i+1}'
        else:
            model = model_class().to(device)
            model_name = f'{model_type}_diff_init_{i+1}'
            
        print(f'{model_name}: {config}')
        model = train_model(model_type, model_name, model, train_loader, config, device)
        results["models"].append(model)

        acc = evaluate(model, test_loader, device)
        results["accs"].append({
            "model_name": model_name,
            "acc": acc
        })
        
    with open(f'{dataset_type}_train_results.txt', 'w') as f:
        for config, acc_dict in zip(results["configs"], results["accs"]):
            name = acc_dict["model_name"]
            acc = acc_dict["acc"]
            f.write(f'{name}: {config}, Test Acc: {acc}\n')
            
    return results
