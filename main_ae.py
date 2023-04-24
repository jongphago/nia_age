import argparse
import os
import random
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18

from data import NiaDataset
from mean_variance_loss import MeanVarianceLoss

LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
START_AGE = 0
END_AGE = 90
NUM_AGE_GROUPS = 9
VALIDATION_RATE = 0.1

random.seed(2019)
np.random.seed(2019)
torch.manual_seed(2019)


class AgeModel(nn.Module):
    def __init__(self, num_ages, num_age_groups):
        super(AgeModel, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.5))

        self.age_classifier = nn.Linear(512, num_ages)
        self.age_group_classifier = nn.Sequential(nn.Linear(512, 256),
                                                  nn.ReLU(),
                                                  nn.BatchNorm1d(256),
                                                  nn.Dropout(0.5),
                                                  nn.Linear(256, num_age_groups))

    def forward(self, x):
        feature = self.backbone(x)
        age_pred = self.age_classifier(feature)
        age_group_pred = self.age_group_classifier(feature)

        return age_pred, age_group_pred


def train(train_loader, model, criterion1, criterion2, optimizer, epoch, result_directory):
    model.train()
    running_loss = 0.
    running_mean_loss = 0.
    running_variance_loss = 0.
    running_softmax_loss = 0.
    interval = 50
    for i, sample in enumerate(train_loader):
        images = sample['image'].cuda()
        labels = sample['age'].cuda()
        age_group_labels = sample['age_class'].cuda()
        # print("train-label---", sample['label'])

        output1, output2 = model(images)
        # age loss
        dta = np.array(sample['data_type'])
        age_sample_indices = dta == 'kinship'
        output1 = output1[age_sample_indices]
        labels = labels[age_sample_indices]
        mean_loss, variance_loss = criterion1(output1, labels)
        softmax_loss = criterion2(output1, labels)
        # age group loss
        output2 = output2[np.logical_not(age_sample_indices)]
        age_group_labels = age_group_labels[np.logical_not(age_sample_indices)]
        softmax_loss2 = criterion2(output2, age_group_labels)
        loss = mean_loss + variance_loss + softmax_loss + 10 * softmax_loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_softmax_loss += softmax_loss.data + softmax_loss2.data
        running_mean_loss += mean_loss.data
        running_variance_loss += variance_loss.data
        if (i + 1) % interval == 0:
            print('[%d, %5d] mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f'
                  % (epoch, i, running_mean_loss / interval,
                     running_variance_loss / interval,
                     running_softmax_loss / interval,
                     running_loss / interval))
            running_loss = 0.
            running_mean_loss = 0.
            running_variance_loss = 0.
            running_softmax_loss = 0.


def train_softmax(train_loader, model, criterion2, optimizer, epoch, result_directory):
    model.train()
    running_loss = 0.
    running_softmax_loss = 0.
    interval = 1
    for i, sample in enumerate(train_loader):
        images = sample['image'].cuda()
        labels = sample['age'].cuda()
        output = model(images)
        loss = criterion2(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        if (i + 1) % interval == 0:
            print('[%d, %5d] loss: %.3f'
                  % (epoch, i, running_loss / interval))
            with open(os.path.join(result_directory, 'log'), 'a') as f:
                f.write('[%d, %5d] loss: %.3f\n'
                        % (epoch, i, running_loss / interval))
            running_loss = 0.


def evaluate_1(val_loader, model):
    log_dict = dict()
    log_dict['No.'] = []
    log_dict['image_path'] = []
    log_dict['data_type'] = []
    log_dict['GT_age'] = []
    log_dict['predicted_age'] = []
    log_dict['GT_age_group'] = []
    log_dict['predicted_age_group'] = []
    log_dict['absolute_error'] = []
    log_dict['mean_absolute_error'] = []
    log_dict['age_group_classification_accuracy'] = []

    # model.cuda()
    model.eval()
    ae_sum = 0.
    kinship_counter = 0
    age_counter = 0
    age_group_success = 0
    # print(val_loader)
    tics = time.time()
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['image'].cuda()

            # print("val-label---", sample['label'])
            # print("image---", image)
            # print(count)
            output, output2 = model(image)
            # print(output)
            m = nn.Softmax(dim=1)
            output_softmax = m(output)
            # print(output_softmax)

            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)[0][0]
            pred_age_group = np.argmax(output2.cpu().data.numpy())
            gt_age_group = sample['age_class'].cpu().item()
            # print("count---", len(val_loader))
            gt_age = sample['age'].cpu().item()
            ae = np.absolute(pred - gt_age)

            filename = sample['file'][0]
            log_dict['No.'].append(i)
            log_dict['image_path'].append(filename)
            data_type = sample['data_type'][0]
            log_dict['data_type'].append(data_type)
            if data_type == 'kinship':
                log_dict['GT_age'].append(gt_age)
                log_dict['predicted_age'].append(pred)
                log_dict['absolute_error'].append(ae)
                log_dict['GT_age_group'].append('-')
                log_dict['predicted_age_group'].append('-')
                ae_sum += ae
                kinship_counter += 1
            else:
                log_dict['GT_age'].append('-')
                log_dict['predicted_age'].append('-')
                log_dict['absolute_error'].append('-')
                log_dict['GT_age_group'].append(gt_age_group)
                log_dict['predicted_age_group'].append(pred_age_group)
                if gt_age_group == pred_age_group:
                    age_group_success += 1
                age_counter += 1
            mae = ae_sum / kinship_counter if kinship_counter > 0 else '-'
            ag_acc = age_group_success / age_counter if age_counter > 0 else '-'
            log_dict['mean_absolute_error'].append(mae)
            log_dict['age_group_classification_accuracy'].append(ag_acc)
    print("# validation ----", len(val_loader))
    return ae_sum / kinship_counter, age_group_success / age_counter, log_dict


def evaluate(val_loader, model, criterion1, criterion2):
    model.cuda()
    model.eval()
    loss_val = 0.
    mean_loss_val = 0.
    variance_loss_val = 0.
    softmax_loss_val = 0.
    mae = 0.
    count = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['image'].cuda()
            label = sample['age'].cuda()
            age_class = sample['age_class'].cuda()
            # print("val-label---", sample['label'])
            # print("image---", image)
            # print(count)
            output, output2 = model(image)
            # print(output)
            mean_loss, variance_loss = criterion1(output, label)
            softmax_loss = criterion2(output, label)
            softmax_loss2 = criterion2(output2, age_class)
            loss = mean_loss + variance_loss + softmax_loss + softmax_loss2
            loss_val += loss.data
            mean_loss_val += mean_loss.data
            variance_loss_val += variance_loss.data
            softmax_loss_val += softmax_loss.data + softmax_loss2.data
            m = nn.Softmax(dim=1)
            output_softmax = m(output)
            # print(output_softmax)

            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            # print("images---", i)
            # print("pred----",  pred)
            # print("label----", label)
            # print("count---", len(val_loader))
            mae += np.absolute(pred - sample['age'].cpu().data.numpy())
    print("# validation ----", len(val_loader))
    return mean_loss_val / len(val_loader), \
           variance_loss_val / len(val_loader), \
           softmax_loss_val / len(val_loader), \
           loss_val / len(val_loader), \
           mae / len(val_loader)


def evaluate_softmax(val_loader, model, criterion2):
    model.cuda()
    model.eval()
    loss_val = 0.
    softmax_loss_val = 0.
    mae = 0.
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['image'].cuda()
            label = sample['age'].cuda()
            age_class = sample['age_class'].cuda()
            output, output2 = model(image)
            loss = criterion2(output, label) + criterion2(output2, age_class)
            loss_val += loss.data
            m = nn.Softmax(dim=1)
            output_softmax = m(output)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            print("-------pred", pred)
            print("-------label", label)
            mae += np.absolute(pred - sample['age'].cpu().data.numpy())
    return loss_val / len(val_loader), mae / len(val_loader)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-i', '--train_meta_path', type=str, default='nia_cropped/train_0.npy')
    parser.add_argument('-v', '--test_meta_path', type=str, default='nia_cropped/test_0.npy')
    parser.add_argument('-ls', '--leave_subject', type=int, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('-rd', '--result_directory', type=str, default='result_model')
    parser.add_argument('-pi', '--pred_image', type=str, default=None)
    parser.add_argument('-pm', '--pred_model', type=str, default=None)
    parser.add_argument('-loss', '--is_mean_variance', action='store_true')
    return parser.parse_args()


def validation(model_path, meta_path):
    model = AgeModel(END_AGE - START_AGE + 1, NUM_AGE_GROUPS)
    model.cuda()
    model.load_state_dict(torch.load(model_path))

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    val_gen = NiaDataset(meta_path, transforms)
    val_loader = DataLoader(val_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    mae, ag_acc, log_dict = evaluate_1(val_loader, model)
    print("Mae---", mae)

    return mae, ag_acc, log_dict


def main(*, train_meta_path: str, test_meta_path: str, leave_subject: int, learning_rate: float, exp_name: str = '',
         is_mean_variance: bool = False, pred_image: str = None, pred_model: str = None, result_directory: str = None,
         resume: str = None, epoch: int = 0, batch_size: int = 16):
    save_model_path = os.path.join(result_directory, "model_{}".format(exp_name))
    # print(args)
    if epoch > 0:
        if result_directory is not None:
            if not os.path.exists(result_directory):
                os.mkdir(result_directory)

        transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.RandomAffine(degrees=10, shear=16),
                 torchvision.transforms.RandomHorizontalFlip(p=1.0),
                 ], p=0.5),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        train_gen = NiaDataset(train_meta_path, transforms_train)
        train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        val_gen = NiaDataset(test_meta_path, transforms)
        val_loader = DataLoader(val_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

        model = AgeModel(END_AGE - START_AGE + 1, NUM_AGE_GROUPS)
        model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE).cuda()
        criterion2 = torch.nn.CrossEntropyLoss().cuda()
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8, 9], gamma=0.1)

        tic = time.time()
        for epoch in range(epoch):
            scheduler.step(epoch)
            if is_mean_variance:
                train(train_loader, model, criterion1, criterion2, optimizer, epoch, result_directory)
                mae, accuracy, log_dict = evaluate_1(val_loader, model)
                print('epoch: %d, mae: %.3f, accuracy: %.3f' % (epoch, mae, accuracy))
            else:
                train_softmax(train_loader, model, criterion2, optimizer, epoch, result_directory)
                loss_val, mae = evaluate_softmax(val_loader, model, criterion2)

            # option 2
            print("model saving----")
            torch.save(model.state_dict(), save_model_path)

    # if pred_image and pred_model:
    mae, ag_acc, log_dict = validation(save_model_path, test_meta_path)
    return mae, ag_acc, log_dict


if __name__ == "__main__":
    args = get_args()

    main(batch_size=args.batch_size,
         train_meta_path=args.train_meta_path,
         test_meta_path=args.test_meta_path,
         leave_subject=args.leave_subject,
         learning_rate=args.learning_rate,
         epoch=args.epoch,
         resume=args.resume,
         result_directory=args.result_directory,
         pred_image=args.pred_image,
         pred_model=args.pred_model,
         is_mean_variance=args.is_mean_variance
         )
