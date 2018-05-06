# coding:utf-8
'''
python 3.5
pytorch 0.3.0
gym 0.9.4
author: helloholmes
'''

import os
import torch
import torchvision as tv
import fire
from torch import nn
from tqdm import tqdm
from torchnet import meter
from config import DefaultConfig
from models import model
from torch.utils.data import DataLoader
from torchvision import transforms as T
from utils.visualize import Visualizer

def write_csv(ids, results, breed, file_name):
    import csv
    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(['id']+breed)
        for i, output in zip(ids, results):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')

def dataloader(data_path, train, batch_size, shuffle, num_workers, drop_last=True):
    transform_train = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    transform_test = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    if train:
        dataset = tv.datasets.ImageFolder(data_path, transform=transform_train)
    else:
        dataset = tv.datasets.ImageFolder(data_path, transform=transform_test)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    return dataloader

def train(opt):
    model_train = getattr(model, opt.model)()
    vis = Visualizer(opt.env)

    if opt.load_model_path:
        model_train.load(opt.load_model_path)
    if opt.use_gpu:
        model_train.cuda()

    train_dataloader = dataloader(opt.train_data_root,
                            train=True,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers)
    val_dataloader = dataloader(opt.valid_data_root,
                            train=False,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers)
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model_train.classifier.parameters(),
                                lr=lr,
                                weight_decay=opt.weight_decay)

    # meter
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(120)
    previous_loss = 1e100

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        if epoch == 20:
            model_train.set_requires_grad()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            if opt.use_gpu:
                data = data.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            score = model_train(data)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, label.data)

            if ii % opt.print_freq:
                # print(ii, ' loss: ', loss_meter.value()[0])
                vis.plot('loss', loss_meter.value()[0])

        model_train.save(opt.save_model_path+opt.model+'_'+str(epoch))

        # validate and visualize
        val_cm, val_accuracy = val(model_train, val_dataloader, opt)
        vis.plot('val_accuracy', val_accuracy)
        # vis.log()

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param in optimizer.param_groups:
                param['lr'] = lr

        previous_loss = loss_meter.value()[0]

def val(model_train, dataloader, opt):
    model_train.eval()
    confusion_matrix = meter.ConfusionMeter(120)
    total_num = 0
    correct_num = 0
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        if opt.use_gpu:
            data = data.cuda()
            # label = data.cuda()

        score = model_train(data)

        confusion_matrix.add(score.data.squeeze(), label.type(torch.LongTensor))
        _, predict = torch.max(score.data, 1)
        total_num += label.size(0)
        correct_num += (predict.cpu() == label).sum()

    model_train.train()
    accuracy = 100 * correct_num / total_num

    return confusion_matrix, accuracy

def test(opt):
    model_test = getattr(model, opt.model)().eval()
    model_test.load(opt.load_model_path)

    breed = os.listdir(opt.train_data_root)
    breed.sort()

    ids = os.listdir(opt.test_data_root+'/unknown')
    ids.sort()

    if opt.use_gpu:
        model_test.cuda()

    test_dataloader = dataloader(opt.test_data_root,
                                train=False,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                drop_last=False)

    results = []

    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        if opt.use_gpu:
            data = data.cuda()

        score = model_test(data)
        probability = nn.functional.softmax(score, dim=1).data

        # batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability)]

        results.extend(probability.cpu().numpy())

    write_csv(ids, results, breed, opt.result_file)




if __name__ == '__main__':
    opt = DefaultConfig()
    # opt.parse()
    # m = getattr(model, opt.model)()
    # # print(m)
    # for param in m.parameters():
    #     print(param.requires_grad)
    opt.parse({'model': 'ResNet50','max_epoch': 30, 'load_model_path': './checkpoints/ResNet50_19'})
    # train(opt)
    test(opt)
    '''
    m = getattr(model, opt.model)().cuda()
    train_dataloader = dataloader(opt.train_data_root,
                            train=True,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers)
    for (data, label) in train_dataloader:
        m.train()
        data = data.cuda()
        label = label.cuda()
        score = m(data)
        print('train: ', score.size())
        m.eval()
        score = m(data)
        print('eval: ', score.size())
        break
    '''