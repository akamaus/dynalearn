#!/usr/bin/env python3
from argparse import ArgumentParser
import math
import os
import random

import PIL.Image as PI
import PIL.ImageDraw as PD
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as AG

from tqdm import tqdm
import video_utils as VU


class DynaModel(nn.Module):
    def __init__(self, name, num_frames=2, num_filters=8):
        super(DynaModel, self).__init__()

        self.conv1 = nn.Conv2d(num_frames, num_filters, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(num_filters, 2, kernel_size=3, stride=2)

        self.deconv1 = nn.ConvTranspose2d(2, num_filters, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(num_filters, 1, kernel_size=3, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform(m.weight.data)

    def forward(self, batch_inp):
        x = F.relu(self.conv1(batch_inp))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x


class Renderer:
    def __init__(self, img_size):
        self.img_size = img_size
        self.img = None
        self.new_frame()

    def new_frame(self):
        self.img = PI.new('L', (self.img_size, self.img_size))

    def polar_circle(self, angle, dist, radius, oblate):
        x = math.cos(angle) * dist + self.img_size / 2
        y = math.sin(angle) * dist + self.img_size / 2

        self.carthesian_circle(x, y, radius, oblate)

    def carthesian_circle(self, x, y, radius, oblate):
        draw = PD.Draw(self.img)

        hr = radius // 2
        vr = hr * oblate
        draw.ellipse((x - hr, y - vr, x + hr, y + vr), fill='white', outline='white')

    def render_spiral_move(self, num_frames=10000, radius_mean=10, radius_std=0, oblate_std=0):
        for frame in range(0, num_frames):
            rad = np.random.normal(radius_mean, radius_std)
            if rad <= 1:
                rad = 1
            obl = np.random.normal(1, oblate_std)
            if obl < 0.01:
                obl = 0.01

            self.new_frame()
            self.polar_circle(angle=frame * (2 * math.pi) / 100,
                              dist=10 + (self.img_size / 2 - 20) / (1 + num_frames) * frame, radius=rad, oblate=obl)
            yield self.img

    # def render_linear_move(num_frames=100):
    #     for fr in range(num_frames):
    #         img = carthesian_circle(10 + (IMG_SIZE - 20) / num_frames * fr, IMG_SIZE / 2)
    #         yield img


class Trainer:
    def __init__(self, args):
        self.name = args.name
        self.num_frames = args.num_frames
        self.img_size = args.img_size

        self.dyna = DynaModel(args.name, num_frames=args.num_frames, num_filters=args.num_filters)
        self.tensor = torch.FloatTensor(1)
        if args.cuda:
            print("Using Cuda")
            self.dyna = self.dyna.cuda()
            self.tensor = self.tensor.cuda()

        self.renderer = Renderer(args.img_size)

        if args.video_name:
            video_name = args.video_name
        else:
            video_name = args.name
        video_name += '.mp4'

        self.vu = VU.VideoWriter(video_name, show=args.gui)

    def checkpoint_path(self):
        return os.path.join('models', self.name + ".pth")

    def draw_frames(self, frames):
        """ Render some frames """
        for fr in frames:
            self.vu.consume(fr)

    def run_training(self, num_epochs, training_episode_len, test_episode_len, eval_period):
        frames = []
        for img in tqdm(self.renderer.render_spiral_move(num_frames=training_episode_len),
                        desc='preparing frames', ascii=True):
            frames.append(np.array(img) / 512.0)

        rnd_rad_frames = []
        for img in tqdm(self.renderer.render_spiral_move(num_frames=training_episode_len, radius_mean=10, radius_std=1, oblate_std=0.2),
                        desc='preparing distorted frames', ascii=True):
            noise = np.random.normal(0, 0.01, (self.renderer.img_size, self.renderer.img_size))
            rnd_rad_frames.append(np.array(img) / 512.0 * np.random.normal(1, 0.05) + noise)

        try:
            for ep in tqdm(range(num_epochs), ascii=True, desc='Epoch'):
                self.train_for_epoch(rnd_rad_frames, frames, batch_size=64)
                if ep % eval_period == 0:
                    self.test_episode(frames, test_episode_len)
                    torch.save(self.dyna, self.checkpoint_path())

        finally:
            self.vu.close()

    def test_episode(self, frames, episode_len):
        episode = episode_len

        start_frame = random.randint(0, len(frames) - episode*2)

        pred_one_step = []
        for f_i in range(start_frame, start_frame + episode):
            pred_one_step.append(self.predict(frames[f_i: f_i+self.num_frames]))

        pred_far = self.far_prediction(frames[start_frame: start_frame + self.num_frames], episode)

        gap = np.zeros((self.renderer.img_size, 16), dtype=np.float32)
        self.draw_frames(map(lambda ps: np.concatenate([ps[0], gap, ps[1]], axis=1), zip(pred_one_step, pred_far)))

    def far_prediction(self, frs, episode_len):
        for i in range(episode_len):
            fr = self.predict(frs)
            frs.append(fr)
            frs = frs[1:]
            yield fr

    def run_demo(self, num_episodes, episode_len):
        frames = []
        for img in tqdm(self.renderer.render_spiral_move(num_frames=1000),
                        desc='preparing frames', ascii=True):
            frames.append(np.array(img) / 512.0)

        for ep in tqdm(range(num_episodes), desc='Episode', ascii=True):
                s = random.randint(0, 500)
                for frame in tqdm(self.far_prediction(frames[s:s+self.num_frames], episode_len),
                                  desc='Frame', ascii=True):
                    self.vu.consume(frame)

    def train_for_epoch(self, inp_frames, tgt_frames, batch_size=8):
        import torch.optim

        opt = torch.optim.Adam(self.dyna.parameters(), lr=0.0001)

        n = len(inp_frames)

        starts = list(range(n - self.num_frames - 1))
        random.shuffle(starts)

        for si in range(0, len(starts), batch_size):
            inp_batch = []
            tgt_batch = []

            for bi in range(0, batch_size):
                if si + bi >= len(starts):
                    break
                fi = starts[si + bi]

                inp_frame = np.stack(list(map(lambda f: inp_frames[f], range(fi, fi+self.num_frames))), axis=2)
                tgt_frame = tgt_frames[fi + self.num_frames]

                inp_batch.append(inp_frame)
                tgt_batch.append(tgt_frame)

            x = AG.Variable(self.tensor.new(np.array(inp_batch).transpose([0,3,1,2])))
            y = self.dyna(x)
            y_truth = AG.Variable(self.tensor.new(np.array(tgt_batch)[:, None, :, :]))
            loss = torch.mean((y - y_truth)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss = loss.cpu().data.numpy()[0]

        tqdm.write('loss: %f' % loss)

    def predict(self, inp_frames):
        expanded = False
        if isinstance(inp_frames, list) and np.ndim(inp_frames[0]) == 2: # this is a sequence of frames
            inp_frames = np.stack(inp_frames, axis=2)
        elif isinstance(inp_frames, np.ndarray) and np.ndim(inp_frames) in [3, 4]:  # already prepared frame stack
            pass
        else:
            raise Exception("unknown dims")

        if np.ndim(inp_frames) == 3:
            inp_frames = np.expand_dims(inp_frames, 0)
            expanded = True

        res = self.dyna(AG.Variable(self.tensor.new(inp_frames.transpose([0,3,1,2]))))
        res = res.data.cpu().numpy() # .transpose([0,3,1,2])
        res = np.maximum(np.minimum(res, 1), 0)

        if expanded:
            res = res[0,0]

        return res


def test():
    dyna = DynaModelMF('tst', 5)
    dyna.restore()

    frames = []
    for i in range(10):
        frames.append(np.ones((IMG_SIZE, IMG_SIZE))*i)

    rnd_rad_frames = []
    for i in range(10):
        rnd_rad_frames.append(np.ones((IMG_SIZE, IMG_SIZE)) * i * (-1))

    dyna.train_for_epoch(rnd_rad_frames, frames, batch_size=3, num_epochs=100000)


parser = ArgumentParser()
add_arg = parser.add_argument
add_arg('--name', default=None)
add_arg('--num-frames', type=int, default=2)
add_arg('--num-filters', type=int, default=8)
add_arg('--img-size', type=int, default=127)
add_arg('--video-name', default=None, help='path to output video')
add_arg('--no-cuda', dest='cuda', default=True, action='store_false')
add_arg('--no-gui', dest='gui', default=True, action='store_false', help='path to output video')

cmds = parser.add_subparsers(dest='cmd')
train_parser = cmds.add_parser('train', help='train on spirals')
add_arg = train_parser.add_argument
add_arg('--num-epochs', type=int, default=1000)
add_arg('--training-episode-len', type=int, default=1000)
add_arg('--test-episode-len', type=int, default=100)
add_arg('--eval-period', type=int, default=5, help='number of epochs before evaluation')

demo_parser = cmds.add_parser('demo', help='run demo')
add_arg = demo_parser.add_argument
add_arg('--num-episodes', type=int, default=3)
add_arg('--episode-len', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()
    tr = Trainer(args)

    if args.cmd == 'train':
        tr.run_training(training_episode_len=args.training_episode_len,
                        test_episode_len=args.test_episode_len,
                        num_epochs=args.num_epochs,
                        eval_period=args.eval_period)
    elif args.cmd == 'demo':
        tr.run_demo(num_episodes=args.num_episodes,
                    episode_len=args.episode_len)
    else:
        raise(Exception('unknown command'))
