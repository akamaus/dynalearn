import matplotlib.pyplot as plt
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision as TV

from torch.autograd import Variable

import video_utils as VU

mnist = TV.datasets.MNIST('MNIST_DATA', download=True, transform=TV.transforms.Compose([TV.transforms.ToTensor()]))
num_digits = 10

loader = T.utils.data.DataLoader(mnist, batch_size=4, shuffle=True)

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


class ConvDisc(nn.Module):
    def __init__(self):
        super(ConvDisc, self).__init__()
        self.pos_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.neg_pad = nn.ZeroPad2d((0, -1, 0, -1))
        self.r = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 8, 3, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, 2)

        f = 32
        s = 3
        self.lin1 = nn.Linear(f*s*s, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight.data)

    def forward(self, x):
        x = self.r(self.conv1(self.pos_pad(x)))
        x = self.r(self.conv2(self.pos_pad(x)))
        x = self.r(self.conv3(x))

        s = x.size()
        x = self.lin1(x.view([s[0], -1]))
        return x


class ConvGen(nn.Module):
    def __init__(self, inp_size):
        super(ConvGen, self).__init__()

        self.inp_size = inp_size
        self.f = 32
        self.s = 3

        self.r = nn.ReLU()
        self.pos_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.neg_pad = nn.ZeroPad2d((0, -1, 0, -1))

        self.lin1 = nn.Linear(inp_size, self.f * self.s ** 2)

        self.tconv3 = nn.ConvTranspose2d(32, 16, 3, 2)
        self.tconv2 = nn.ConvTranspose2d(16, 8, 3, 2)
        self.tconv1 = nn.ConvTranspose2d(8, 1, 3, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight.data)

    def forward(self, x):
        sz = x.size()
        assert sz[1] == self.inp_size

        x = self.r(self.lin1(x)).view((sz[0], self.f, self.s, self.s))

        x = self.r(self.tconv3(x))
        x = self.r(self.neg_pad(self.tconv2(x)))
        x = self.neg_pad(self.tconv1(x))
        x = F.sigmoid(x)
        return x


#model = LinearModel(l2=100).cuda()

gen = ConvGen(inp_size=32).cuda()
dis = ConvDisc().cuda()

gen_opt = T.optim.RMSprop(params=gen.parameters(), lr=0.0001)
dis_opt = T.optim.RMSprop(params=dis.parameters(), lr=0.0001)

#opt = T.optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)

vu = VU.VideoWriter('tst.mp4', show=True)

g_xs = []
d_xs = []

g_losses = []
d_losses = []

num_epochs = 100
batch_size = 24

example = T.rand((10, gen.inp_size)).cuda()

for ep in range(num_epochs):
    loop_g_losses = []
    loop_d_losses = []

    k = 0
    for batch in loader:
        noise = T.rand((batch_size, gen.inp_size)).cuda()

        noise = Variable(noise)
        fake = gen(noise)
        real = batch[0].cuda()

        if k % 3 != 0:
            dis_opt.zero_grad()
            w_loss = T.mean(dis(fake)) - T.mean(dis(real))
            w_loss.backward()
            dis_opt.step()
            loop_d_losses.append(w_loss.data.cpu())
        else:
            g_loss = - T.mean(dis(fake))
            g_loss.backward()
            gen_opt.step()
            loop_g_losses.append(g_loss.data.cpu())

        k += 1
        if k == 50:
            break

    for p in dis.parameters():
        p.data[...] = p.data.clamp(-0.01, 0.01)

    g_losses.append(np.mean(loop_g_losses))
    d_losses.append(np.mean(loop_d_losses))
    #losses_std.append(np.std(loop_losses))

    print('ep %d; gloss %f; dloss %f' % (ep, g_losses[-1], d_losses[-1]))
    tst_out = gen(Variable(example))
    sz = tst_out.size()
    tst_pic = tst_out.data.permute(1,2,0,3).contiguous()[0].view((sz[2],-1)).cpu()
    vu.consume(tst_pic.numpy())

g_losses = np.array(g_losses)
d_losses = np.array(d_losses)

#losses_std = np.array(losses_std)

xs = np.arange(len(g_losses))
f = plt.figure(2)
ax = f.add_subplot(111)
ax.plot(xs, g_losses, 'r', xs, d_losses, 'b')
#ax.fill_between(xs, losses - losses_std, losses + losses_std, alpha=0.5)
plt.ioff()
plt.show()

