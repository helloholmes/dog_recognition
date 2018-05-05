# coding:utf-8
'''
python 3.5
pytorch 0.4.0
auther: helloholmes
'''
import os
import math
import shutil
from collections import Counter

def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    # label_file is '.csv'
    # the generated dataset dir is input_dir
    # train_dir, test_dir are the origin data
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # skip first line
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        # idx_label: index: the name of the picture, label: the label
        idx_label = dict((idx, label) for idx, label in tokens)
    # labels include 120 varieties of dogs, length is 120
    labels = set(idx_label.values())
    '''
    for l in labels:
        print(l)
        break
    print(len(labels))
    '''
    # num_train = train + valid
    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    # the minimium number of one label
    min_num_train_per_label = (
        Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    num_valid_per_label = math.floor(min_num_train_per_label*valid_ratio)
    label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # make train and valid dataset
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        # data for valid
        if label not in label_count or label_count[label] < num_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))

    # make test dataset
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))


if __name__ == '__main__':
    data_dir = '/home/qinliang/Desktop/kaggle/dog_recognition'
    label_file = 'labels.csv'
    train_dir = 'data_train_origin'
    test_dir = 'data_test_origin'
    input_dir = 'train_valid_test'
    valid_ratio = 0.1
    reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)