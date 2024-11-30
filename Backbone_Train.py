import logging
import os
import random
import sys
import glob
import matplotlib.pyplot as plt

import numpy as np
import torchvision
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, Grayscale
from PIL import Image
import imageio

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.nn.functional import one_hot
from model import MyResnet

import monai
from monai.data import ImageDataset
from monai.transforms import Activations, AsDiscrete, Compose, ScaleIntensity, EnsureType, AsChannelFirst, AddChannel,\
    RandRotate, RandAdjustContrast, RandFlip, RandAffine, AsChannelLast
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric


# Choose your hyperparameters and the range of the random search
num_workers = 24
BatchSize_train = (16, 128)
BatchSize_val = 500
output = 2
in_channels = 1
LearningRate = (1e-3, 1e-5)
WeightDecay = (1e-5, 1e-7)
prob_1 = (0.1, 0.5)
prob_2 = (0.1, 0.5)
prob_3 = (0.1, 0.5)
prob_4 = (0.1, 0.5)
prob_5 = (0.1, 0.5)
Epochs = 100
num_trials = 5
class_x = 'MCI'
class_y = 'AD'
train_data_type = 'GM_0.2_5p'
val_data_type = 'GM_0.2'
Slice = 'Oblique Coronal'
num_fold = '5-fold'
class_fold = 'fold_5'
device = torch.device("cuda" if torch.cuda.is_available() else print("cuda is not available"))


def main():

    # get data path
    file_x = glob.glob(os.sep.join(["train set path", "*.png"]))
    file_y = glob.glob(os.sep.join(["train set path", "*.png"]))

    val_x = glob.glob(os.sep.join(["validation set path", "*.png"]))
    val_y = glob.glob(os.sep.join(["validation set path", "*.png"]))

    num_x_train = len(file_x)
    num_x_val = len(val_x)

    num_y_train = len(file_y)
    num_y_val = len(val_y)

    img_train = file_x + file_y
    img_val = val_x + val_y
    print(len(img_train), len(img_val))

    # Define label
    train_labels = np.concatenate((np.zeros(num_x_train, dtype=np.int64), np.ones(num_y_train, dtype=np.int64)))
    val_labels = np.concatenate((np.zeros(num_x_val, dtype=np.int64), np.ones(num_y_val, dtype=np.int64)))

    # Convert labels to one-hot vectors
    train_labels = torch.from_numpy(train_labels)
    train_onehot = one_hot(train_labels, num_classes=output).float()
    val_labels = torch.from_numpy(val_labels)
    val_onehot = one_hot(val_labels, num_classes=output).float()

    best_metric = -1
    best_hyperparameters = {}

    # Training and Random Search
    for trial in range(num_trials):

        lr = random.uniform(*LearningRate)
        wd = random.uniform(*WeightDecay)
        bs = random.randint(*BatchSize_train)
        p1 = random.uniform(*prob_1)
        p2 = random.uniform(*prob_2)
        p3 = random.uniform(*prob_3)
        p4 = random.uniform(*prob_4)
        p5 = random.uniform(*prob_5)

        # Define transforms
        train_transforms = Compose([AsChannelFirst(),
                                    AddChannel(),
                                    RandFlip(prob=p1, spatial_axis=1),
                                    RandAdjustContrast(prob=p2, gamma=(0.3, 2.5)),
                                    RandRotate(range_x=np.pi/18, range_y=0, range_z=0, prob=p3),
                                    RandAffine(scale_range=(0.15, 0.15, 0), prob=p4, padding_mode="zeros"),
                                    RandAffine(translate_range=(10, 20), prob=p5, padding_mode="zeros"),
                                    ScaleIntensity(),
                                    EnsureType()
                                    ])
        val_transforms = Compose([AsChannelFirst(),
                                  AddChannel(),
                                  ScaleIntensity(),
                                  EnsureType()])



        # Define image dataset, data loader
        check_dataset = ImageDataset(image_files=img_train, labels=train_onehot, transform=train_transforms)
        weights = [(num_x_train + num_y_train) / num_x_train if label.argmax(dim=0) == 0
                   else (num_x_train + num_y_train) / num_y_train for data, label in check_dataset]
        sampler = WeightedRandomSampler(weights, num_samples=num_x_train * 2 if num_x_train < num_y_train else num_y_train * 2, replacement=True)
        check_loader = DataLoader(check_dataset, batch_size=bs, num_workers=num_workers,
                                  pin_memory=torch.cuda.is_available(), shuffle=None, sampler=sampler)
        im, label = monai.utils.misc.first(check_loader)
        print(type(im), im.shape, label)

        # Create a training data loader
        train_dataset = ImageDataset(image_files=img_train, labels=train_onehot, transform=train_transforms)
        weights = [(num_x_train + num_y_train) / num_x_train if label.argmax(dim=0) == 0
                   else (num_x_train + num_y_train) / num_y_train for data, label in check_dataset]
        sampler = WeightedRandomSampler(weights, num_samples=num_x_train * 2 if num_x_train < num_y_train else num_y_train * 2, replacement=False)
        train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers, shuffle=None,
                                  sampler=sampler, pin_memory=torch.cuda.is_available())

        # Create a validation data loader
        val_dataset = ImageDataset(image_files=img_val, labels=val_onehot, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=BatchSize_val, num_workers=num_workers,
                                pin_memory=torch.cuda.is_available())

        # Creat a Resnet-18 model
        model = torchvision.models.resnet18(num_classes=output).to(device)
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


        # Start training
        val_interval = 1
        loss_interval = 10
        epochs = Epochs
        epoch_loss_values = list()
        metric_values = list()
        writer = SummaryWriter()
        for epoch in range(epochs):
            print("-" * 100)
            print(f"trial {trial + 1}/{num_trials}")
            print(f"epoch {epoch + 1}/{epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_dataset) // train_loader.batch_size
                if step % loss_interval == 0:
                    print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # validate by validation set
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    num_correct = 0.0
                    metric_count = 0
                    for val_data in val_loader:
                        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                        val_outputs = model(val_images)
                        value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                        metric_count += len(value)
                        num_correct += value.sum().item()
                    metric = num_correct / metric_count
                    metric_values.append(metric)

                    if metric > best_metric:
                        best_metric = metric
                        best_trial = trial + 1
                        best_metric_epoch = epoch + 1
                        best_hyperparameters = {'lr': lr, 'wd': wd, 'bs': bs, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5,}
                        torch.save(model.state_dict(), f"best_model_{class_x}{class_y}_{Slice}_{train_data_type}_{class_fold}_res18_newval_randsearch.pth")
                        print("saved new best metric model")
                    print(
                        "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at trial {} epoch {}".format(
                            epoch + 1, metric, best_metric, best_trial, best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at trial {best_trial} epoch {best_metric_epoch}")
    print("Best Hyperparameters:", best_hyperparameters)
    writer.close()

if __name__ == "__main__":
    main()