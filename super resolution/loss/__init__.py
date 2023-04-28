import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel_loss(nn.Module):
    def __init__(self):
        super(Sobel_loss,self).__init__()
    def forward(self,output,target):

        b, c, h, w = output.shape
        filter_x  = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        filter_x = np.expand_dims(filter_x, 0)
        filter_x = np.tile(np.expand_dims(filter_x, 0), [c, 1, 1, 1])
        filter_x = torch.tensor(filter_x).cuda()

        filter_y = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        filter_y = np.expand_dims(filter_y, 0)
        filter_y = np.tile(np.expand_dims(filter_y, 0), [c, 1, 1, 1])
        filter_y = torch.tensor(filter_y).cuda()

        filter_a = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
        filter_a = np.expand_dims(filter_a, 0)
        filter_a = np.tile(np.expand_dims(filter_a, 0), [c, 1, 1, 1])
        filter_a = torch.tensor(filter_a).cuda()

        filter_b = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32)
        filter_b = np.expand_dims(filter_b, 0)
        filter_b = np.tile(np.expand_dims(filter_b, 0), [c, 1, 1, 1])
        filter_b = torch.tensor(filter_b).cuda()
        # print('1111111111111111111111111111111111111')
        # print(output.device)
        # print(filter_x.device)
        # print('1111111111111111111111111111111111111')

        output_gradient_x = torch.square(F.conv2d(output, filter_x, groups=c, stride = 1, padding = 1))
        output_gradient_y = torch.square(F.conv2d(output, filter_y, groups=c, stride = 1, padding = 1))
        output_gradient_a = torch.square(F.conv2d(output, filter_a, groups=c, stride = 1, padding = 1))
        output_gradient_b = torch.square(F.conv2d(output, filter_b, groups=c, stride = 1, padding = 1))

        output_gradients = torch.sqrt(output_gradient_x + output_gradient_y +  output_gradient_a +  output_gradient_b + 1e-6)

        target_gradient_x = torch.square(F.conv2d(target, filter_x, groups=c, stride = 1, padding = 1))
        target_gradient_y = torch.square(F.conv2d(target, filter_y, groups=c, stride = 1, padding = 1))
        target_gradient_a = torch.square(F.conv2d(target, filter_a, groups=c, stride = 1, padding = 1))
        target_gradient_b = torch.square(F.conv2d(target, filter_b, groups=c, stride = 1, padding = 1))

        target_gradients = torch.sqrt(target_gradient_x + target_gradient_y + target_gradient_a + target_gradient_b + 1e-6)

        mse = nn.MSELoss()

        loss = mse(output_gradients, target_gradients)
        
        return loss


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'sobel':
                loss_function = Sobel_loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()


        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss

                losses.append(effective_loss)
   
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

