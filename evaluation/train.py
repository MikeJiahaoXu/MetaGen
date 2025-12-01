"""
Train and evaluation methods
"""

import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

from model import Prediction_Network as PNN
from dataset import MetaDataset
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import pickle
import seaborn as sns
from sklearn.metrics import mean_squared_error
import json

def visualize_line_plots(ys, labels, rows, cols):
    """
    Visulize a group of 1D vectors as line plots and display them in a large figure in a specified row and column manner
    Args:
        vectors: Contains a list of 1D vectors, each vector should have the same length
        rows: Number of rows in the large figure
        cols: Number of columns in the large figure
    """
    num_vectors = len(ys)
    assert num_vectors == rows * cols, "Rows and cols must match the number of vectors"

    fig, axes = plt.subplots(rows, cols, figsize=(25, 25))
    c_sample = np.linspace(60, 80, 50)
    for i, ax in enumerate(axes.flat):
        y = ys[i]
        label = labels[i]
        ax.plot(c_sample, y, label='predicted', color='#845EC2', linewidth=2)
        ax.plot(c_sample, label, label='ground truth',color='#008E9B', linewidth=2)
        ax.set_title(f'Test case {i+1}')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('sample.png')


def bagging_train(modelConfig: Dict):
    """
    Use the ensemble learning method(Bagging) to train the model
    Args:
        modelConfig: Dict: Configuration for the model
    """
    # Ensure reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(modelConfig["device"])

    # dataset setup=======================
    dataset = MetaDataset(modelConfig["meta_path"], modelConfig["base_path"])
    train_size = int(len(dataset) * 0.8)
    valid_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    dataloaders = []
    subsets = []

    # Bootstrap sampling=======================
    for i in range(modelConfig["num_models"]):
        subset_indices = np.random.choice(train_size, size=train_size, replace=True)
        subset = Subset(dataset, subset_indices)
        subsets.append(subset)
    for subset in subsets:
        dataloader = DataLoader(subset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4)
        dataloaders.append(dataloader)
    
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True
    )

    # model setup=======================
    models = []
    optimizers = []
    schedulers = []
    for _ in range(modelConfig["num_models"]):
        model = PNN(modelConfig["condition_dim"]).to(device)
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=modelConfig["lr"], weight_decay=1e-3))
        schedulers.append(StepLR(optimizers[_], step_size=1, gamma=0.1))
    criterion = torch.nn.MSELoss()
    
    # start training=======================
    plt_train = []
    plt_val = []
    models_loss = []
    for i in range(modelConfig["num_models"]):
        models_loss.append([])
    for e in range(modelConfig["epoch"]):
        validation_loss = []
        for i in range(modelConfig["num_models"]):
            ith_train_loss = []
            # Train each model on its own subset
            with tqdm(dataloaders[i], dynamic_ncols=True) as tqdmDataLoader:
                for images, conditions, labels in tqdmDataLoader:
                    x_0 = images.to(device)
                    c_0 = conditions.to(device)
                    labels = labels.to(device)
                    y = models[i](x_0, c_0)
                    optimizers[i].zero_grad()
                    loss = criterion(y, labels)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(models[i].parameters(), modelConfig["grad_clip"])
                    optimizers[i].step()
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": e,
                       "model number": i,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                        "LR": optimizers[i].state_dict()['param_groups'][0]["lr"]
                    })
                    ith_train_loss.append(loss.item())
            models_loss[i].append(sum(ith_train_loss) / len(ith_train_loss))
            if e % 20 == 0:
                # Save model weight
                torch.save(model.state_dict(), os.path.join(
                    modelConfig["save_dir"], 'ckpt_' + str(e) + "_" + str(i)+ "_.pt"))
            if e == 0:
                schedulers[i].step()
        with torch.no_grad():
            # validate the model on the validation set
            for vimages, vconditions, vlabels in valid_dataloader:
                vx_0 = vimages.to(device)
                vc_0 = vconditions.to(device)
                vlabels = vlabels.to(device)
                vbagging_predictions = torch.zeros(vlabels.shape[0], vlabels.shape[1]).to(device)
                for model in models:
                    model.eval()
                    vy = model(vx_0, vc_0)
                    vbagging_predictions += vy
                vbagging_predictions /= modelConfig["num_models"]  # 取平均
                vbagging_loss = criterion(vbagging_predictions, vlabels)
                validation_loss.append(vbagging_loss.item())
            plt_val.append(sum(validation_loss) / len(validation_loss))
        # log the loss
        x = range(len(plt_val))
        plt.plot(x, plt_val, label='val')
        for i in range(modelConfig["num_models"]):
            plt.plot(x, models_loss[i], label=str(i) + '-th train loss')
        with open('Log.txt', 'a') as file:
            file.write(f"Epoch {e}:  Valid Loss {sum(validation_loss) / len(validation_loss)} \n")
        plt.legend()
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('Loss.png')
        plt.clf()


def eval(modelConfig: Dict):
    # Ensure reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(modelConfig["device"])

    # dataset setup=======================
    dataset = MetaDataset(modelConfig["base_path"])
    train_size = int(len(dataset) * 0.8)
    valid_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    test_dataloader = DataLoader(
        test_dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=4, drop_last=True, pin_memory=True
    )
    
    # load model and evaluate
    models = []
    with torch.no_grad():
        for i in range(modelConfig["num_models"]):
            model = PNN(modelConfig["condition_dim"]).to(device)
            ckpt = torch.load(os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]) + str(i) + "_.pt", map_location=device)
            model.load_state_dict(ckpt)
            models.append(model)
            models[i].eval()
        print("models load weight done.")
        # begin to evaluate, final prediction is the average of all models
        for batch, (images, conditions, labels) in enumerate(test_dataloader):
            x_0 = images.to(device)
            c_0 = conditions.to(device)
            labels = labels.to(device)
            vbagging_predictions = torch.zeros(labels.shape[0], labels.shape[1]).to(device)
            for model in models:
                y = model(x_0, c_0)
                vbagging_predictions += y
            vbagging_predictions /= modelConfig["num_models"]
            visualize_line_plots(vbagging_predictions.cpu().detach().numpy(), labels.cpu().detach().numpy(), 8, 8)
            # delete break to evaluate all test data
            break

