import math

import matplotlib.pyplot as plt
import numpy as np

import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision as TV

from figure import Figure, Accumulator
import video_utils as VU

import mnist_models as mm

NAME = 'gp_prelu'
NUM_EPOCHS = 20000
EPOCH_LEN = 100
BATCH_SIZE = 24
LAMBDA = 10
D_TRAIN_RATIO = 5

NUM_TST_SAMPLES = 40
NUM_TST_ROWS = 5

mnist = TV.datasets.MNIST('MNIST_DATA', download=True, transform=TV.transforms.Compose([TV.transforms.ToTensor()]))
num_digits = 10

loader = T.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)


class GradientHistory:
    def __init__(self, model, tau=0.9):
        self.model = model
        self.tau = tau
        self.mean_grads = None
        self.disp_grads = None
        #for p in model.parameters():
        #    self.mean_grads.append(T.zeros_like(p.data))

    def flat_grads(self):
        return T.cat(list(map(lambda p: p.grad.data.view(-1), self.model.parameters())), dim=0)

    def update(self):
        grads = self.flat_grads()
        if self.mean_grads is None:
            self.mean_grads = grads
        else:
            self.mean_grads += (grads - self.mean_grads) * self.tau

        disp = (grads - self.mean_grads)**2
        if self.disp_grads is None:
            self.disp_grads = disp
        else:
            self.disp_grads += (disp - self.disp_grads) * self.tau

    def get_mean_norm(self):
        return self.mean_grads.norm(2)

    def get_std_norm(self):
        return math.sqrt(self.disp_grads.norm(2))


czero = Variable(T.cuda.FloatTensor([0]))

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = T.rand(BATCH_SIZE, 1, 1, 1).cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = T.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=T.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (T.max(czero, gradients.view(BATCH_SIZE, -1).norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


#model = LinearModel(l2=100).cuda()

netG = mm.Generator(NAME).resume().cuda()
netD = mm.Discriminator(NAME).resume().cuda()

#gen = ConvGen(NAME).resume().cuda()
#dis = ConvDisc(NAME).resume().cuda()

gen_opt = T.optim.Adam(params=netG.parameters(), lr=0.0001, betas=(0.5, 0.9))
dis_opt = T.optim.Adam(params=netD.parameters(), lr=0.0001, betas=(0.5, 0.9))

#opt = T.optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)

vu = VU.VideoWriter('%s_evolution.mp4' % NAME, show=True)


g_costs = Accumulator(with_std=True)
w_dists = Accumulator(with_std=True)

d_grad_norms = Accumulator(with_std=True)
g_grad_norms = Accumulator(with_std=True)

g_grad = GradientHistory(netG)
d_grad = GradientHistory(netD)

fig = Figure(accums={'G_cost': g_costs,
                     'W_dist': w_dists,
                     'g_grad': g_grad_norms, 'd_grad': d_grad_norms})


def batches_gen():
    while True:
        for b, _ in loader:
            yield b


def gen_noise(num_samples=BATCH_SIZE):
    return T.rand(num_samples, netG.inp_size).cuda()

def train_loop():
    example = gen_noise(NUM_TST_SAMPLES)
    batch_gen = batches_gen()
    for ep in range(NUM_EPOCHS):
        for k in range(EPOCH_LEN):
            batch = next(batch_gen)
            train_d = k % D_TRAIN_RATIO != 0

            noise = Variable(T.rand((BATCH_SIZE, netG.inp_size)).cuda())
            fake = netG(noise)

            real = Variable(batch.cuda())

            if train_d:
                dis_opt.zero_grad()
                w_dist = T.mean(netD(real)) - T.mean(netD(fake))
                w_loss = -w_dist
                w_loss.backward()

                penalty = calc_gradient_penalty(netD, real.data, fake.data)
                penalty.backward()

                dis_opt.step()

                d_grad.update()
                w_dists.append(w_dist.data.cpu())
            else:
                gen_opt.zero_grad()
                g_loss = - T.mean(netD(fake))
                g_loss.backward()
                gen_opt.step()

                g_grad.update()
                g_costs.append(-g_loss.data.cpu())

        g_grad_norms.accumulate_raw(g_grad.get_mean_norm(), g_grad.get_std_norm())
        d_grad_norms.accumulate_raw(d_grad.get_mean_norm(), d_grad.get_std_norm())
        g_costs.accumulate()
        w_dists.accumulate()

        fig.plot_accums()
        fig.draw()

        print('ep %d; w_dist %f; g_cost %f' % (ep, w_dists.history[-1], g_costs.history[-1]))
        tst_out = netG(Variable(example))
        sz = tst_out.size()
        one_line_pic = tst_out.data.permute(1,2,0,3).contiguous()[0].view((sz[2],-1))
        pics = one_line_pic.chunk(NUM_TST_ROWS, dim=1)
        pic = T.cat(pics, dim=0).cpu()

        vu.consume(pic.numpy())

        if ep > 0 and ep % 100 == 0:
            netG.save()
            netD.save()


try:
    train_loop()
except KeyboardInterrupt:
    pass

netG.save()
netD.save()

# ax.plot(xs, g_losses, 'r', xs, d_losses, 'b')
# ax.fill_between(xs, g_losses - g_losses_std, g_losses + g_losses_std, alpha=0.3, color='r')
# ax.fill_between(xs, d_losses - d_losses_std, d_losses + d_losses_std, alpha=0.3, color='b')
# plt.ioff()
# plt.show()
