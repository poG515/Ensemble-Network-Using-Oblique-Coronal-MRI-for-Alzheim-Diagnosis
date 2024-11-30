import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.nn.init as init
from sklearn.metrics import confusion_matrix, roc_auc_score

import os, random
import pandas as pd
import numpy as np
import glob
import monai
from monai.data import ImageDataset
from monai.transforms import Compose, ScaleIntensity, EnsureType, AsChannelFirst, AddChannel,\
    RandRotate, RandAdjustContrast, Rotate90, RandFlip, RandAffine, AsChannelLast


BatchSize_train = (4, 25)
BatchSize_val = 500
Epochs = 50
in_channels = 1
output = 2
LearningRate = (1e-3, 1e-5)
WeightDecay = (1e-5, 1e-7)
prob_1 = (0, 0.5)
prob_2 = (0, 0.5)
prob_3 = (0, 0.5)
prob_4 = (0, 0.5)
prob_5 = (0, 0.5)
fc1 = (128, 256)
fc2 = (64, 128)
fc3 = (32, 64)
dropout = (0, 0.5)
num_per_subject = 3
num_trials = 5
class_x = 'CN'
class_y = 'MCI'
Slice = 'Coronal'
data_type_train = 'GM+WM+CSF_0.2'
data_type_val = 'GM+WM+CSF_0.2'
num_fold = '5-fold'
class_fold = 'fold_5'
class_val = 2
device = torch.device("cuda" if torch.cuda.is_available() else print("cuda is not available"))


# Arrange the data in the data set in order
def get_sequential_set(path, class_name, num_fold, fold_name, split_num):

    file_list = os.listdir(path)
    csvpath = "/media/b607/LiCunhao/Alzheimers_disease_prognosis/ADNI1/{0}/ADNI-1_{1}_{2}.csv"\
        .format(class_name, class_name, num_fold)
    datainf = pd.read_csv(csvpath, usecols=['Image Data ID', '{0}'.format(fold_name)])
    ID_list = datainf['Image Data ID'].tolist()
    split_list = datainf['{0}'.format(fold_name)].tolist()
    index_list = [index for index, value in enumerate(split_list) if value == split_num]

    files = []
    for index in index_list:
        id = ID_list[index]
        files.append(glob.glob(path + id + '_*_CSF.png'))
        files.append(glob.glob(path + id + '_*_GM.png'))
        files.append(glob.glob(path + id + '_*_WM.png'))

    sequential_file = [item for sublist in files if sublist for item in sublist]

    return sequential_file


# Groups images belonging to the same subject in order
class CustomSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        #
        indices = list(range(len(self.data_source)))

        # Divide the index into groups, each containing num_per_subject units
        grouped_indices = [indices[i:i + num_per_subject] for i in range(0, len(indices), num_per_subject)]

        # Shuffle the order of each group
        np.random.shuffle(grouped_indices)

        # Retrieve the index within each group in the order of the scrambled groups
        shuffled_indices = [index for group in grouped_indices for index in group]

        return iter(shuffled_indices)

    def __len__(self):
        return len(self.data_source)


# Architecture of the ensemble network
class EnsembleNet(nn.Module):
    def __init__(self, backbone, backbone_weights, last_layer, num_classes, drop_out):
        super(EnsembleNet, self).__init__()

        # Load initial ResNet model
        self.backbones = nn.ModuleList([backbone(num_classes=num_classes) for _ in range(num_per_subject)])

        # Set the first convolutional layer for each ResNet instance
        for resnet in self.backbones:
            resnet.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Load the pre-training weights
        if backbone_weights is not None:
            for i, resnet in enumerate(self.backbones):
                resnet.load_state_dict(torch.load(backbone_weights[i]))
                print(f'Pretrain weights for backbone {i+1} loaded')

        # Freeze the weights of the Pretrain Network
        for resnet in self.backbones:
            resnet.eval()

        # Obtain the input of the fully connected layer in the Pretrain Model
        self.feature_list = nn.ModuleList([
            nn.Sequential(*list(resnet.children())[:-1],
            nn.Flatten()
            ) for resnet in self.backbones
        ])

        self.last_layer = last_layer

        # Define a fully connected layer with two hidden layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 * num_per_subject, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=drop_out),
            torch.nn.Linear(512, output),
        )

        # Define a 1×1 convolutional layer, followed by a fully connected layers
        self.conv = torch.nn.Conv2d(num_per_subject, 1, 1, bias=None)
        self.fc_after_conv = torch.nn.Linear(512, output)

        # Define the structure of weighted voting
        self.soft_voting_1 = nn.Linear(num_per_subject, 1, bias=False)
        self.soft_voting_2 = nn.Linear(num_per_subject, 1, bias=False)

        ## If you want to use averaging for ensemble,
        ## you just need to set all the weights in the weighted voting structure to 1/num_per_subject
        # init.constant_(self.soft_voting_1.weight, 0.333333)
        # init.constant_(self.soft_voting_2.weight, 0.333333)

    def forward(self, x):

        # Ensemble method using a fully connected layer with two hidden layers
        if self.last_layer == 'fc':
            features = []
            for imgs in x:
                feature = [backbone(torch.unsqueeze(img, dim=0)) for backbone, img in zip(self.feature_list, imgs)]
                feature = [torch.tensor(i) for i in feature]
                feature = torch.cat(feature, dim=1)
                features.append(feature)
            features = torch.squeeze(torch.stack(features, dim=1), dim=0)
            y = self.fc(features)

        # Ensemble method using a 1×1 convolutional layer followed by a fully connected layers
        if self.last_layer == 'conv':
            features = []
            for imgs in x:
                # Gets all the feature values
                feature = [backbone(torch.unsqueeze(img, dim=0)) for backbone, img in zip(self.feature_list, imgs)]
                feature = [torch.tensor(i) for i in feature]
                feature = torch.stack(feature, dim=1) # Stack feature values from different networks
                features.append(feature)
            features = torch.squeeze(torch.stack(features, dim=0), dim=1).unsqueeze(2)
            y = self.conv(features).view(-1, 512)
            y = self.fc_after_conv(y)
            y.requires_grad_(True)

        # Ensemble method using weighted voting
        if self.last_layer == 'weighted_voting':
            y_i_list = []
            for imgs in x:
                # Gets the confidence value of all backbone outputs
                feature = [backbone(torch.unsqueeze(img, dim=0)) for backbone, img in zip(self.backbones, imgs)]
                feature = [torch.tensor(i) for i in feature]
                feature = torch.stack(feature, dim=0).view(num_per_subject, -1)
                feature = torch.softmax(feature, dim=1)
                y_1 = self.soft_voting_1(feature[:, 0].view(1, num_per_subject)).squeeze(-1)
                y_2 = self.soft_voting_2(feature[:, 1].view(1, num_per_subject)).squeeze(-1)
                y_i = torch.stack([y_1, y_2], dim=1)
                y_i_list.append(y_i)
            y = torch.stack(y_i_list, dim=1).squeeze(0)
            y.requires_grad_(True)

        # Ensemble method using majority voting
        if self.last_layer == 'majority_voting':

            y_i_list = []
            for imgs in x:
                feature = [backbone(torch.unsqueeze(img, dim=0)) for backbone, img in zip(self.backbones, imgs)]
                feature = [torch.tensor(i) for i in feature]
                feature = torch.stack(feature, dim=0).view(num_per_subject, -1)
                y_voting = torch.argmax(feature, dim=1).float()
                y_voting = torch.sum(y_voting)
                y_i = torch.where(y_voting >= 2, torch.tensor([0, 1], device='cuda:0'),
                                  torch.tensor([1, 0], device='cuda:0')).float()
                y_i_list.append(y_i)
            y = torch.stack(y_i_list, dim=0)

        return y


# Validate using validation set
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    step = 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_data in val_loader:
            step += 1
            # Example: raw data=15×1×224×224--> reshape=3×5×1×224×224
            img_batch_shape = (int(batch_data[0].shape[0] / num_per_subject), num_per_subject, in_channels, 224, 224)
            img_batch = batch_data[0].view(img_batch_shape)

            label_batch_shape = (int(batch_data[0].shape[0] / num_per_subject), num_per_subject, output)
            label_batch = batch_data[1].view(label_batch_shape)
            label_batch = torch.sum(label_batch, dim=1) / num_per_subject

            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            y = model(img_batch)  # Images are fed into the model to get outputs

            # Calculate validation loss and accuracy
            loss = criterion(y, label_batch)
            val_loss += loss.item()

            _, predicted = torch.max(y.data, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch.argmax(dim=1)).sum().item()

            all_labels.append(label_batch.cpu().numpy())
            all_preds.append(y.argmax(dim=1).cpu().numpy())
            all_probs.append(y.softmax(dim=1).cpu().numpy())

        # Calculate other metrics
        val_loss /= step
        accuracy = correct / total

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

        cm = confusion_matrix(all_labels.argmax(axis=1), all_preds)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    return val_loss, accuracy, sensitivity, specificity, auc


def main():

    # get path of training set
    file_x = get_sequential_set(f"/data_path/{Slice}/ensemble_train/{num_fold}/{class_fold}/{class_x}_{data_type_train}/",
                                class_x, num_fold, class_fold, 1)
    file_y = get_sequential_set(f"/data_path/{Slice}/ensemble_train/{num_fold}/{class_fold}/{class_y}_{data_type_train}/",
                                class_y, num_fold, class_fold, 1)

    # get path of validation set
    val_x = get_sequential_set(f"/data_path/{Slice}/validation/{num_fold}/{class_fold}/{class_x}_{data_type_train}/",
                                class_x, num_fold, class_fold, 2)
    val_y = get_sequential_set(f"/data_path/{Slice}/validation/{num_fold}/{class_fold}/{class_y}_{data_type_train}/",
                                class_y, num_fold, class_fold, 2)


    num_x_train = len(file_x)
    num_x_val = len(val_x)

    num_y_train = len(file_y)
    num_y_val = len(val_y)

    img_train = file_x + file_y
    img_val = val_x + val_y

    # Get labels in the form of one-hot vector
    train_labels = np.concatenate((np.zeros(num_x_train, dtype=np.int64), np.ones(num_y_train, dtype=np.int64)))
    val_labels = np.concatenate((np.zeros(num_x_val, dtype=np.int64), np.ones(num_y_val, dtype=np.int64)))
    train_labels = torch.from_numpy(train_labels)
    train_onehot = one_hot(train_labels, num_classes=output).float()
    val_labels = torch.from_numpy(val_labels)
    val_onehot = one_hot(val_labels, num_classes=output).float()

    best_accuracy = -1
    best_hyperparameters = {}

    # Start Random Search
    for trial in range(num_trials):

        # Define the range of the Random search
        lr = random.uniform(*LearningRate)
        wd = random.uniform(*WeightDecay)
        bs = random.randint(*BatchSize_train) * num_per_subject
        p1 = random.uniform(*prob_1)
        p2 = random.uniform(*prob_2)
        p3 = random.uniform(*prob_3)
        p4 = random.uniform(*prob_4)
        p5 = random.uniform(*prob_5)
        DropOut = random.uniform(*dropout)

        # Define transforms
        train_transforms = Compose([AsChannelFirst(),
                                    AddChannel(),
                                    RandFlip(prob=p1, spatial_axis=1),
                                    RandAdjustContrast(prob=p2, gamma=(0.3, 2.5)),
                                    RandRotate(range_x=np.pi / 18, range_y=0, range_z=0, prob=p3),
                                    RandAffine(scale_range=(0.15, 0.15, 0), prob=p4, padding_mode="zeros"),
                                    RandAffine(translate_range=(10, 20), prob=p5, padding_mode="zeros"),
                                    ScaleIntensity(),
                                    EnsureType()
                                    ])
        val_transforms = Compose([AsChannelFirst(),
                                  AddChannel(),
                                  ScaleIntensity(),
                                  EnsureType()])

        # Create training aset and test if it to correct and feasible
        train_dataset = ImageDataset(image_files=img_train, labels=train_onehot, transform=train_transforms)
        sampler = CustomSampler(train_dataset, batch_size=bs)
        train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=24, shuffle=None,
                                  sampler=sampler, pin_memory=torch.cuda.is_available())
        im, label = monai.utils.misc.first(train_loader)
        print(type(im), im.shape, label)

        # Create validation set
        val_dataset = ImageDataset(image_files=img_val, labels=val_onehot, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=BatchSize_val, num_workers=24, shuffle=None,
                                pin_memory=torch.cuda.is_available())

        # Define the model and loss function
        model = EnsembleNet(backbone=models.resnet18,
                            backbone_grad=False,
                            backbone_weights=[f'best_model_{class_x}{class_y}_{Slice}_CSF_0.2_5p_res18.pth',
                                              f'best_model_{class_x}{class_y}_{Slice}_GM_0.2_5p_res18.pth',
                                              f'best_model_{class_x}{class_y}_{Slice}_WM_0.2_5p_res18.pth'],
                            drop_out=DropOut,
                            last_layer='conv',
                            num_classes=output).to(device)

        ## If you want to test with a test set, you just need to load the training weights here
        # model.load_state_dict(torch.load('best_ensemble_model_CNAD_Coronal_CE_3type_wv.pth'))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lr=lr, weight_decay=wd)

        # Test the model's performance
        model_valuation = validate(model, val_loader, criterion, device)
        print(f'Accuracy of pretrain model: {model_valuation[1]}\n'
              f'Sensitivity: {model_valuation[2]}\n'
              f'Specificity: {model_valuation[3]}\n'
              f'AUC: {model_valuation[4]}')

        # Start training
        num_epochs = Epochs
        for epoch in range(num_epochs):
            avg_epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                # Example: raw data=15×1×224×224--> reshape=3×5×1×224×224
                img_batch_shape = (int(batch_data[0].shape[0] / num_per_subject), num_per_subject, in_channels, 224, 224)
                img_batch = batch_data[0].view(img_batch_shape)

                # Example: raw label=15×2--> reshape=3×5×2
                label_batch_shape = (int(batch_data[0].shape[0] / num_per_subject), num_per_subject, output)
                label_batch = batch_data[1].view(label_batch_shape)
                label_batch = torch.sum(label_batch, dim=1) / num_per_subject  # Example: 5×[1,0]-->[1,0]

                img_batch, label_batch = img_batch.to(device), label_batch.to(device)

                optimizer.zero_grad()
                y = model(img_batch)
                loss = criterion(y, label_batch)
                loss.backward()
                optimizer.step()
                avg_epoch_loss += loss.item()

            avg_epoch_loss /= step
            train_accuracy = validate(model, train_loader, criterion, device)
            val_valuation = validate(model, val_loader, criterion, device)
            val_accuracy = val_valuation[1]
            print(f'trial {trial + 1}/{num_trials}\n'
                  f'Epoch [{epoch + 1}/{num_epochs}]\n'
                  f'Train Loss: {avg_epoch_loss}, '
                  f'Train Accuracy: {train_accuracy[1]}, \n'
                  f'Validation Loss: {val_valuation[0]}, '
                  f'Validation Accuracy: {val_valuation[1]}')

            # Save the best model on the validation set
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                sen = val_valuation[2]
                spe = val_valuation[3]
                auc = val_valuation[4]
                best_epoch = epoch + 1
                best_trial = trial + 1
                torch.save(model.state_dict(), f'best_ensemble_model_{class_x}{class_y}_{Slice}_CE_3type_conv.pth')
                best_hyperparameters = {'lr': lr, 'wd': wd, 'bs': bs, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
                                        'p5': p5, 'fc1': NumFc1, 'fc2': NumFc2, 'fc3': NumFc3}
            print(f'Best Accuracy: {best_accuracy} at trial {best_trial} at Epoch {best_epoch} \n'
                  f'sen: {sen} spe: {spe} auc: {auc}')

        print(f'Training Completed\n'
              f'Best Accuracy: {best_accuracy} at trial {best_trial} at Epoch {best_epoch}\n'
              f'sen: {sen} spe: {spe} auc: {auc}')


if __name__ == "__main__":
    main()