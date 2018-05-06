# coding:utf-8
'''
python 3.5
pytorch 0.3.0
gym 0.9.4
author: helloholmes
'''
import warnings

class DefaultConfig(object):
    env = 'default'
    model = 'Vgg16'

    train_data_root = './train_valid_test/train'
    test_data_root = './train_valid_test/test'
    valid_data_root = './train_valid_test/valid'

    load_model_path = ''

    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20

    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001
    lr_decay = 0.5
    weight_decay = 0e-5

def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn('warning: opt has not attribut %s'%k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))

DefaultConfig.parse = parse

if __name__ == '__main__':
    opt = DefaultConfig()
    print(opt.env)
    new_config = {'env': 'new'}
    opt.parse(new_config)
    print(opt.env)