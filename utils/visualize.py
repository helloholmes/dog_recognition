# coding:utf-8
'''
python 3.5
pytorch 0.4.0
visdom 0.1.7
torchnet 0.0.2
auther: helloholmes
'''
import visdom
import time
import numpy as np

class Visualizer(object):
    def __init__(self, env='default'):
        self.vis = visdom.Visdom(env=env)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='default'):
        self.vis = visdom.Visdom(env=env)
        return self

    def plot_many(self, d):
        '''
        d: dict(name, value)
        '''
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        '''
        d: dict(name, image)
        '''
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(X=np.array([x]),
                    Y=np.array([y]),
                    win=name,
                    update=None if x == 0 else 'append')
        # update index of name
        self.index[name] = x + 1

    def img(self, name, img_):
        '''
        self.img('input_img', torch.Tensor(64, 64))
        '''
        self.vis.images(image=img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name))

    def log(self, info, win='log_text'):
        '''
        self.log({'loss': 1, 'lr': 0.0001})
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),
                            info=info))
        self.vis.text(self.log_text)

    def __getattr__(self, name):
        return getattr(self.vis, name)