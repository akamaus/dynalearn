import matplotlib.pyplot as plt
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision as TV

from torch.autograd import Variable

import video_utils as VU
from figure import Figure

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

class LinearModel(nn.Module):
    def __init__(self, l2):
        super(LinearModel, self).__init__()
        self.lin1 = nn.Linear(28*28, l2)
        self.lin2 = nn.Linear(l2, 28 * 28)

    def forward(self, inp):
        sz = inp.size()
        inp = inp.view(sz[0], -1)
        x = F.relu(self.lin1(inp))
        x = self.lin2(x)
        return x.view(sz)


class ConvDisc(mm.PersistentModule):
    def __init__(self, name):
        super(ConvDisc, self).__init__('ConvDisc_' + name)
        self.pos_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.neg_pad = nn.ZeroPad2d((0, -1, 0, -1))
        self.r = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 16, 3, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 2)

        f = 64
        s = 3
        self.lin1 = nn.Linear(f*s*s, 1)

    def forward(self, x):
        x = self.r(self.conv1(self.pos_pad(x)))
        x = self.r(self.conv2(self.pos_pad(x)))
        x = self.r(self.conv3(x))

        s = x.size()
        x = self.lin1(x.view([s[0], -1]))
        return x


class ConvGen(mm.PersistentModule):
    def __init__(self, name):
        super(ConvGen, self).__init__('ConvGen_' + name)

        self.f = 64
        self.s = 3
        self.inp_size = self.f * self.s ** 2

        self.r = nn.ReLU()
        self.pos_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.neg_pad = nn.ZeroPad2d((0, -1, 0, -1))

        #self.lin1 = nn.Linear(inp_size, self.f * self.s ** 2)

        self.tconv3 = nn.ConvTranspose2d(64, 32, 3, 2)
        self.tconv2 = nn.ConvTranspose2d(32, 16, 3, 2)
        self.tconv1 = nn.ConvTranspose2d(16, 1, 3, 2)

    def forward(self, x):
        sz = x.size()
        assert sz[1] == self.inp_size

        x = x.view((sz[0], self.f, self.s, self.s))

        x = self.r(self.tconv3(x))
        x = self.r(self.neg_pad(self.tconv2(x)))
        x = self.neg_pad(self.tconv1(x))
#        x = F.sigmoid(x)
        return x

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


class Accumulator:
    def __init__(self):
        self.history = []
        self.history_std = []
        self.last = []

    def append(self, y):
        self.last.append(y)

    def accumulate(self):
        self.history.append(np.mean(self.last))
        self.history_std.append(np.std(self.last))
        self.last.clear()


#model = LinearModel(l2=100).cuda()

netG = mm.Generator(NAME).resume().cuda()
netD = mm.Discriminator(NAME).resume().cuda()

#gen = ConvGen(NAME).resume().cuda()
#dis = ConvDisc(NAME).resume().cuda()

gen_opt = T.optim.Adam(params=netG.parameters(), lr=0.0001, betas=(0.5, 0.9))
dis_opt = T.optim.Adam(params=netD.parameters(), lr=0.0001, betas=(0.5, 0.9))

#opt = T.optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)

vu = VU.VideoWriter('%s_evolution.mp4' % NAME, show=True)

fig = Figure(['g_loss', 'd_loss', 'W_dist'])

g_xs = []
d_xs = []

g_losses = Accumulator()
d_losses = Accumulator()
w_dists = Accumulator()

one = T.FloatTensor([1]).cuda()
mone = one * -1


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

                d_losses.append(w_loss.data.cpu())
                w_dists.append(w_dist.data.cpu())
            else:
                gen_opt.zero_grad()
                g_loss = - T.mean(netD(fake))
                g_loss.backward()
                gen_opt.step()

                g_losses.append(g_loss.data.cpu())

        g_losses.accumulate()
        d_losses.accumulate()
        w_dists.accumulate()

        fig.plot('g_loss', g_losses.history)
        fig.plot('d_loss', d_losses.history)
        fig.plot('W_dist', w_dists.history)

        fig.draw()

        print('ep %d; gloss %f; dloss %f' % (ep, g_losses.history[-1], d_losses.history[-1]))
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
