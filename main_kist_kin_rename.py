import os 
import time 
import json 
import argparse
import torch 
import torchvision
import random
import numpy as np 
from data import FaceDataset
from tqdm import tqdm 
from torch import nn
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet18
from mean_variance_loss import MeanVarianceLoss
import cv2

LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
START_AGE = 0
END_AGE = 90
VALIDATION_RATE= 0.1

random.seed(2019)
np.random.seed(2019)
torch.manual_seed(2019)


def ResNet18(num_classes):

    model = resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


def train(train_loader, model, criterion1, criterion2, optimizer, epoch, result_directory):

    model.train()
    running_loss = 0.
    running_mean_loss = 0.
    running_variance_loss = 0.
    running_softmax_loss = 0.
    interval = 1
    for i, sample in enumerate(train_loader):
        images = sample['image'].cuda()
        labels = sample['label'].cuda()
        #print("train-label---", sample['label'])

        output = model(images)
        mean_loss, variance_loss = criterion1(output, labels)
        softmax_loss = criterion2(output, labels)
        loss = mean_loss + variance_loss + softmax_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_softmax_loss += softmax_loss.data
        running_mean_loss += mean_loss.data
        running_variance_loss += variance_loss.data
        if (i + 1) % interval == 0:
            print('[%d, %5d] mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f'
                  % (epoch, i, running_mean_loss / interval,
                     running_variance_loss / interval,
                     running_softmax_loss / interval,
                     running_loss / interval))
            with open(os.path.join(result_directory, 'log'), 'a') as f:
                f.write('[%d, %5d] mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f\n'
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
        labels = sample['label'].cuda()
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
    #model.cuda()
    model.eval()
    fr = open('./test_mae4.txt','w')
    mae = 0.
    print(val_loader)
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['image'].cuda()
            label = sample['label'].cuda()
           
            #print("val-label---", sample['label'])
            #print("image---", image)
            #print(count) 
            output = model(image)
            #print(output)
            m = nn.Softmax(dim=1)
            output_softmax = m(output)
            #print(output_softmax)

            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            filename = ''.join(map(str, sample['file']))
            file_mae = np.absolute(pred -sample['label'].cpu().data.numpy())  
            fr.write("%s\t%d\n"%(filename, file_mae))
            #print("count---", len(val_loader))
            mae += np.absolute(pred - sample['label'].cpu().data.numpy())
    print("# validation ----", len(val_loader))
    fr.close()
    return mae / len(val_loader)


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
            label = sample['label'].cuda()
            #print("val-label---", sample['label'])
            #print("image---", image)
            #print(count) 
            output = model(image)
            #print(output)
            mean_loss, variance_loss = criterion1(output, label)
            softmax_loss = criterion2(output, label)
            loss = mean_loss + variance_loss + softmax_loss
            loss_val += loss.data
            mean_loss_val += mean_loss.data
            variance_loss_val += variance_loss.data
            softmax_loss_val += softmax_loss.data
            m = nn.Softmax(dim=1)
            output_softmax = m(output)
            #print(output_softmax)

            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            #print("images---", i)
            #print("pred----",  pred)
            #print("label----", label)
            #print("count---", len(val_loader))
            mae += np.absolute(pred - sample['label'].cpu().data.numpy())
    print("# validation ----", len(val_loader))
    return mean_loss_val / len(val_loader),\
        variance_loss_val / len(val_loader),\
        softmax_loss_val / len(val_loader),\
        loss_val / len(val_loader),\
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
            label = sample['label'].cuda()
            output = model(image)
            loss = criterion2(output, label)
            loss_val += loss.data
            m = nn.Softmax(dim=1)
            output_softmax = m(output)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            print("-------pred", pred)
            print("-------label", label)
            mae += np.absolute(pred - sample['label'].cpu().data.numpy())
    return loss_val / len(val_loader), mae / len(val_loader)


def test(test_loader, model):
    model.cuda()
    model.eval()
    mae = 0.

    #print(model)
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample['image'].cuda()
            label = sample['label'].cuda()
            output = model(image)
            m = nn.Softmax(dim=1)
            output = m(output)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            print("pred---, ground----", pred, sample['label'])  
            mae += np.absolute(pred - sample['label'].cpu().data.numpy())
    return mae / len(test_loader)


def predict(model, image):
    model.cuda()
    model.eval()
    with torch.no_grad():
        image = image.astype(np.float32) / 255.
        image = np.transpose(image, (2,0,1))#2,0,1
        img = torch.from_numpy(image).cuda()
        print("image---", img)
        output = model(img[None])
        print("predict output-----", output)
        m = nn.Softmax(dim=1)
        output = m(output)
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]

        #pred = np.around(mean)
    return pred


def get_image_list(image_directory, leave_sub, validation_rate):
    
    train_val_list = []
    test_list = []
    for fn in os.listdir(image_directory):
        filepath = os.path.join(image_directory, fn)
        #print("filepath--------\n", filepath)
        #subject = int(fn[:3])
        str_fn = str(fn)
        
        filenames = str_fn.split('/')[-1]
        #print(filenames)
        subject = int(filenames.split('_')[2])
        #print("subject-----", subject)
        if subject == leave_sub:
            test_list.append(filepath)
            #print("test_list---", test_list)
        else:
            train_val_list.append(filepath)
    num = len(train_val_list)
    index_val = np.random.choice(num, int(num * validation_rate), replace=False)
    print("random_index----",index_val)
    train_list = []
    val_list = []
    for i, fp in enumerate(train_val_list):
        if i in index_val:
            val_list.append(fp)
        else:
            train_list.append(fp)

    return train_list, val_list, test_list

def get_new_list(image_directory, leave_sub, validation_directory):
    train_list = []
    test_list = []
    val_list = []
    for fn in os.listdir(image_directory):
        filepath = os.path.join(image_directory, fn)
        #print("filepath--------\n", filepath)
        #subject = int(fn[:3])
        str_fn = str(fn)
        
        filenames = str_fn.split('/')[-1]
        #print(filenames)
        subject = int(filenames.split('_')[2])
        #print("subject-----", subject)
        if subject == leave_sub:
            test_list.append(filepath)
            #print("test_list---", test_list)
        else:
            train_list.append(filepath)

    #print("validation directory name---", validation_directory)

    for fns in os.listdir(validation_directory):
        filepaths = os.path.join(validation_directory, fns)
        #print("val list---", filepaths)
        val_list.append(filepaths)
    #print("validation_list---", val_list)

    return train_list, val_list, test_list



def get_test_list(image_directory, leave_sub):
    test_list = []
    age_list = []
    for fn in os.listdir(image_directory):
        filepath = os.path.join(image_directory, fn)
        str_fn = str(fn)
        print(str_fn)
        filenames = str_fn.split('/')[-1]
        subject = int(filenames.split('_')[2])
        g_age = int(filenames.split('_')[0])
        #if subject == leave_sub:
        test_list.append(filepath)
        age_list.append(g_age)
    return test_list, age_list         

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-i', '--image_directory', type=str)
    parser.add_argument('-v', '--val_directory', type=str)
    parser.add_argument('-ls', '--leave_subject', type=int)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    parser.add_argument('-pi', '--pred_image', type=str, default=None)
    parser.add_argument('-pm', '--pred_model', type=str, default=None)
    parser.add_argument('-loss', '--is_mean_variance', action='store_true')
    return parser.parse_args()


def main():
    
    args = get_args()
    #print(args)
    if args.epoch > 0:
        batch_size = args.batch_size
        if args.result_directory is not None:
            if not os.path.exists(args.result_directory):
                os.mkdir(args.result_directory)

        #train_filepath_list, val_filepath_list, test_filepath_list\
            #= get_image_list(args.image_directory, args.leave_subject, VALIDATION_RATE)
        train_filepath_list, val_filepath_list, test_filepath_list\
            = get_new_list(args.image_directory, args.leave_subject, args.val_directory)
   
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
        train_gen = FaceDataset(train_filepath_list, transforms_train)
        train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        val_gen = FaceDataset(val_filepath_list, transforms)
        val_loader = DataLoader(val_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

        test_gen = FaceDataset(test_filepath_list, transforms)
        test_loader = DataLoader(test_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
        #print(train_filepath_list)
        #print(val_filepath_list)
        print(test_filepath_list)
        model = ResNet18(END_AGE - START_AGE + 1)
        model.cuda()

        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
        criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE).cuda()
        criterion2 = torch.nn.CrossEntropyLoss().cuda()
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)

        #print(model)


        best_val_mae = np.inf
        best_val_loss = np.inf
        best_test_mae = np.inf
        best_mae_epoch = -1
        best_loss_epoch = -1
        best_test_epoch = -1
        for epoch in range(args.epoch):
            scheduler.step(epoch)
            if args.is_mean_variance:
                train(train_loader, model, criterion1, criterion2, optimizer, epoch, args.result_directory)
                mean_loss, variance_loss, softmax_loss, loss_val, mae = evaluate(val_loader, model, criterion1, criterion2)
                print('epoch: %d, mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f, mae: %3f' %
                      (epoch, mean_loss, variance_loss, softmax_loss, loss_val, mae))
                with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                    f.write('epoch: %d, mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f, mae: %3f\n' %
                        (epoch, mean_loss, variance_loss, softmax_loss, loss_val, mae))
            else:
                train_softmax(train_loader, model, criterion2, optimizer, epoch, args.result_directory)
                loss_val, mae = evaluate_softmax(val_loader, model, criterion2)
                #print('epoch: %d, loss: %.3f, mae: %3f' % (epoch, loss_val, mae))
                #with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                    #f.write('epoch: %d, loss: %.3f, mae: %3f\n' % (epoch, loss_val, mae))

            #mae_test = test(test_loader, model)
            #print('epoch: %d, test_mae: %3f' % (epoch, mae_test))
            #option 1
            
            #if best_test_mae > mae_test:
                #print("best_test_mae, mae_test----", best_test_mae, mae_test)
                #best_test_mae = mae_test
                #best_test_epoch = epoch

                #torch.save(model.state_dict(), os.path.join(args.result_directory,"model_best_mae"))
                #print('best_epoch :%d, best_test_mae:%f'%(best_test_epoch, best_test_mae))

            #with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                #f.write('epoch: %d, mae_test: %3f\n' % (epoch, mae_test))

            #option 2

            if best_val_mae > mae:
                best_val_mae = mae
                print("best_val_mae---", best_val_mae)
                print("mae---", mae)
                print("model saving----")
                best_mae_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.result_directory, "mae_4"))
            if best_val_loss > loss_val:
                best_val_loss = loss_val
                best_loss_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.result_directory, "loss_4"))            
        with open(os.path.join(args.result_directory, 'log'), 'a') as f:
            f.write('best_loss_epoch: %d, best_val_loss: %f, best_mae_epoch: %d, best_val_mae: %f\n'
                    % (best_loss_epoch, best_val_loss, best_mae_epoch, best_val_mae))
        print('best_loss_epoch: %d, best_val_loss: %f, best_mae_epoch: %d, best_val_mae: %f'
              % (best_loss_epoch, best_val_loss, best_mae_epoch, best_val_mae))

    #if args.pred_image and args.pred_model:
    if args.pred_model:
        model = ResNet18(END_AGE - START_AGE + 1)
        model.cuda()
        model.load_state_dict(torch.load(args.pred_model))

        test_list = []
        age_list = []

        test_list, age_list = get_test_list(args.image_directory, args.leave_subject)
  
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        val_gen = FaceDataset(test_list, transforms)
        val_loader = DataLoader(val_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

        mae = evaluate_1(val_loader, model)
        print("Mae---", mae)

        #mae = 0    
        #for i, files in enumerate(test_list):
            #print(i)
            #img = cv2.imread(test_list[i])
            #resized_img = cv2.resize(img, (224,224))
            #pred = predict(model, resized_img)
            #print('Age:' + str(int(pred)))
            #ga = age_list[i]
            #pred = int(pred)
            #ga = int(ga)
            #print(ga, pred)
            #mae += abs(ga - pred)
         
        #print(mae/len(test_list))a
        #img = cv2.imread(args.pred_image)
        #resized_img = cv2.resize(img, (224, 224))
        #model.load_state_dict(torch.load(args.pred_model))
        
        #pred = predict(model, resized_img)
        #print('Age: ' + str(int(pred)))
        #cv2.putText(img, 'Age: ' + str(int(pred)), (int(img.shape[1]*0.1), int(img.shape[0]*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        #name, ext = os.path.splitext(args.pred_image)
        #names = "./result_pic/" + name
        #cv2.imwrite(names + '_result.jpg', img)
        
if __name__ == "__main__":
    main()
