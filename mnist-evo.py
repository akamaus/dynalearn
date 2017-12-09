import torch as T
import torch.nn as nn
import torchvision as TV
from torch.autograd import Variable

import video_utils as VU

mnist = TV.datasets.MNIST('MNIST_DATA', download=True, transform=TV.transforms.Compose([TV.transforms.ToTensor()]))

samples = []
for i in range(10):
    k = 0
    for img, lbl in mnist:
        if lbl == i:
            print('found digit %d at %d' % (i, k))
            samples.append(img)
            break
        k += 1

dataset = T.utils.data.TensorDataset(T.stack(samples).cuda(), T.stack(samples[1:] + [samples[0]]).cuda())
loader = T.utils.data.DataLoader(dataset, batch_size=4)

fs = 8

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.lin1 = nn.Linear(28*28, 28*28)

    def forward(self, inp):
        s = inp.size()
        inp = inp.view(s[0], -1)
        x = self.lin1(inp)
        x = x.view(s)
        return x

model = LinearModel().cuda()

# model = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(1, 8, 3, 2), nn.ReLU(),
#                       nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(8, 16, 3, 2), nn.ReLU(),
#                       nn.Conv2d(16, 32, 3, 2), nn.ReLU(),
#                       nn.ConvTranspose2d(32, 16, 3, 2), nn.ReLU(),
#                       nn.ConvTranspose2d(16, 8, 3, 2), nn.ZeroPad2d((0, -1, 0, -1)), nn.ReLU(),
#                       nn.ConvTranspose2d(8, 1, 3, 2), nn.ZeroPad2d((0, -1, 0, -1)), nn.ReLU()).cuda()


for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)

opt = T.optim.Adam(params=model.parameters())

vu = VU.VideoWriter('tst.mp4', show=True)

for ep in range(1000):

    for inp, tgt in loader:
        inp = Variable(inp)
        tgt = Variable(tgt)

        opt.zero_grad()
        out = model(inp)
        loss = T.mean((out - tgt)**2)
        loss.backward()
        opt.step()

    if ep % 10 == 0:
        print('ep %d; loss %f' % (ep, loss.cpu().data))
        t_inp, t_tgt = dataset[:]
        s = t_inp.size()
        t_out = model(Variable(t_inp)).data
        frame = T.cat([t_out, t_tgt, T.zeros_like(t_out)], dim=1).cpu().permute(1,2,0,3).contiguous().view([3, s[2], s[3]*s[0]])
        vu.consume(frame.numpy().transpose(1,2,0))
