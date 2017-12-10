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

samples = []
for i in range(num_digits):
    k = 0
    for img, lbl in mnist:
        if lbl == i:
            print('found digit %d at %d' % (i, k))
            samples.append(img.cuda())
            break
        k += 1

targets = samples[1:] + [samples[0]]

dataset = T.utils.data.TensorDataset(T.stack(samples), T.stack(targets))
loader = T.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

fs = 8

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

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.pos_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.neg_pad = nn.ZeroPad2d((0, -1, 0, -1))
        self.r = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 8, 3, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, 2)

        s = 3
        self.lin1 = nn.Linear(32*s*s, 32*s*s)

        self.tconv3 = nn.ConvTranspose2d(32, 16, 3, 2)
        self.tconv2 = nn.ConvTranspose2d(16, 8, 3, 2)
        self.tconv1 = nn.ConvTranspose2d(8, 1, 3, 2)

    def forward(self, x):
        x = self.r(self.conv1(self.pos_pad(x)))
        x = self.r(self.conv2(self.pos_pad(x)))
        x = self.r(self.conv3(x))

        s = x.size()
        x = self.r(self.lin1(x.view([s[0], -1]))).view(s)

        x = self.r(self.tconv3(x))
        x = self.r(self.neg_pad(self.tconv2(x)))
        x = self.r(self.neg_pad(self.tconv1(x)))
        return x


model = LinearModel(l2=100).cuda()
#model = ConvModel().cuda()


for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)

opt = T.optim.Adam(params=model.parameters(), lr=0.0001)

vu = VU.VideoWriter('tst.mp4', show=True)
losses = []
losses_std = []

smpl = T.unsqueeze(samples[0], 0)
for ep in range(1000):
    outs = []
    loop_losses = []
    for si in range(num_digits):
        if ep < 120:
            smpl = T.unsqueeze(samples[si], 0)
        inp = Variable(smpl)
        tgt = Variable(T.unsqueeze(targets[si], 0))

        opt.zero_grad()
        out = model(inp)
        loss = T.mean((out - tgt)**2)
        loss.backward()
        opt.step()

        loop_losses.append(loss.cpu().data)
        outs.append(out.data.cpu()[0])
        smpl = out.data.clamp(0,1)

    losses.append(np.mean(loop_losses))
    losses_std.append(np.std(loop_losses))
    print('ep %d; loss %f +- %f' % (ep, losses[-1], losses_std[-1]))
    t_outs = T.stack(outs, 0)
    t_tgts = T.stack(targets).cpu()

    sz = samples[0].size()
    frame = T.cat([t_outs, t_tgts, T.zeros_like(t_outs)], dim=1).permute(1,2,0,3).contiguous().view([3, sz[1], sz[2] * len(outs)])
    n_frame = (frame.permute(1,2,0).clamp(min=0, max=1) * 255).numpy().astype(np.uint8)
    vu.consume(n_frame)

losses = np.array(losses)
losses_std = np.array(losses_std)

xs = np.arange(len(losses))
f = plt.figure(2)
ax = f.add_subplot(111)
ax.plot(xs, losses, 'r')
ax.fill_between(xs, losses - losses_std, losses + losses_std, alpha=0.5)
plt.ioff()
plt.show()

