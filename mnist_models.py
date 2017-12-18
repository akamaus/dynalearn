import torch
import torch.nn as nn

DIM = 64 # Model dimensionality
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

INP_SIZE = 128

class PersistentModule(nn.Module):
    def __init__(self, name):
        super(PersistentModule, self).__init__()
        self.name = name

    def savepath(self):
        return self.name + '.mdl'

    def save(self):
        sp = self.savepath()
        torch.save(self.state_dict(), sp)
        print('Saving model weights to %s' % sp)

    def resume(self):
        import os
        sp = self.savepath()
        if os.path.exists(sp):
            print('Loading model parameters from %s' % sp)
            self.load_state_dict(torch.load(sp))
        else:
            print("Warning, no data found at %s, starting afresh" % sp)
            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            #         nn.init.xavier_uniform(m.weight.data)
        return self


# ==================Definition Start======================
class Generator(PersistentModule):
    def __init__(self, name):
        super(Generator, self).__init__(name + '_Generator')

        preprocess = nn.Sequential(
            nn.Linear(INP_SIZE, 4*4*4*DIM),
            nn.PReLU(),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.PReLU(),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.PReLU(),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.inp_size = INP_SIZE
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output


class Discriminator(PersistentModule):
    def __init__(self, name):
        super(Discriminator, self).__init__(name + '_Discriminator')

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.PReLU(),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.PReLU(),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.PReLU(),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

# MINE old models

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


class ConvDisc(PersistentModule):
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


class ConvGen(PersistentModule):
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
