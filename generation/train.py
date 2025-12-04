# Description: Train the diffusion model

import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gaussian import GaussianDiffusionSampler, GaussianDiffusionTrainer, DDIMSampler
from model import UNet
from scheduler import GradualWarmupScheduler
from dataset import MetaDataset
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from PIL import Image


def log(loss_list):
    """
    Log the loss and save the loss plot
    """
    plt_train = []
    plt_train.append(sum(loss_list) / len(loss_list))
    x = range(len(plt_train))
    plt.plot(x, plt_train, label='train')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.clf()
    with open('Log.txt', 'a') as file:
        file.write(f"Epoch {len(loss_list)}: Loss {sum(loss_list) / len(loss_list)}\n")


def _load_weights(model: torch.nn.Module, weight_path: str, device: torch.device, strict: bool = False):
    """
    Load a checkpoint and strip a possible DataParallel prefix.
    """
    state_dict = torch.load(weight_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)


def _get_state_dict(model: torch.nn.Module):
    """
    Handle state dict retrieval for both single and DataParallel models.
    """
    return model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()


def train(modelConfig: Dict):
    """
    Train the diffusion model
    Args: modelConfig: Dict: Configuration for the model
    meta_path: str: Path to the meta data (include S-params and other geometric parameters)
    base_path: str: Path to the image data
    """
    # Ensure reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(modelConfig["device"] if torch.cuda.is_available() else "cpu")

    os.makedirs(modelConfig["save_dir"], exist_ok=True)
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)

    # dataset setup=======================
    dataset = MetaDataset(modelConfig["meta_path"], modelConfig["base_path"])
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    # model setup=======================
    net_model = UNet(T=modelConfig["T"], params_dim=modelConfig["params_dim"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        _load_weights(
            net_model,
            os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"]),
            device,
            strict=False,
        )
        print("Model weight load down.")

    use_data_parallel = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_data_parallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
        net_model = torch.nn.DataParallel(net_model)

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    # cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    # warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training=======================
    plt_train = []
    for e in range(modelConfig["epoch"]):
        loss_list = []
        net_model.train()
        with tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels, _ in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                loss_list.append(loss.item())
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        log(loss_list)
        # warmUpScheduler.step()
        # save model weight
        if e % 5 == 0:
            torch.save(
                _get_state_dict(net_model),
                os.path.join(modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"),
            )
        if e < 6:
            scheduler.step()
        

def eval(modelConfig: Dict):
    # Ensure reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(modelConfig["device"] if torch.cuda.is_available() else "cpu")

    # Make sure output directories exist when running evaluation only
    os.makedirs(modelConfig["save_dir"], exist_ok=True)
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)

    # dataset setup=======================
    dataset = MetaDataset(modelConfig["meta_path"], modelConfig["base_path"])
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    # load model and evaluate
    with torch.no_grad():
        model = UNet(T=modelConfig["T"], params_dim=modelConfig["params_dim"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
        _load_weights(
            model,
            os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]),
            device,
            strict=True,
        )
        use_data_parallel = torch.cuda.is_available() and torch.cuda.device_count() > 1
        if use_data_parallel:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs for evaluation.")
            model = torch.nn.DataParallel(model)

        print("model load weight done.")
        model.eval()
        if modelConfig["sample_method"] == "basic":
            sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        else:
            sampler = DDIMSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        for i, (images, labels, file_names) in enumerate(test_dataloader):
            if i == 5:
                break
            images = images.to(device)
            labels = labels.to(device)
            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 1, modelConfig["img_height"], modelConfig["img_width"]], device=device)
            sampledImgs = sampler(noisyImage, labels)
            grayImgs = torch.mean(sampledImgs, dim=1, keepdim=True)
            threshold = 0.5
            binaryImgs = torch.where(grayImgs > threshold, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
            
            grid_padding = modelConfig.get("sample_grid_padding", 2)
            save_image(
                binaryImgs,
                os.path.join(modelConfig["sampled_dir"], modelConfig["sampledImgName"]),
                nrow=modelConfig["nrow"],
                padding=grid_padding,
            )

            image = Image.open(os.path.join(modelConfig["sampled_dir"], modelConfig["sampledImgName"]))
            array = np.expand_dims(np.array(image), axis=-1) if image.mode == "L" else np.array(image)
            tile_h = modelConfig["img_height"]
            tile_w = modelConfig["img_width"]
            nrow = modelConfig["nrow"]

            for idx, name in enumerate(file_names):
                row = idx // nrow
                col = idx % nrow
                top = grid_padding + row * (tile_h + grid_padding)
                left = grid_padding + col * (tile_w + grid_padding)
                sub_arr = array[top:top + tile_h, left:left + tile_w]
                if sub_arr.ndim == 3 and sub_arr.shape[2] == 1:
                    sub_arr = sub_arr[:, :, 0]
                sub_img = Image.fromarray(sub_arr.astype("uint8"))
                sub_img.save(os.path.join(modelConfig["sampled_dir"], name))
