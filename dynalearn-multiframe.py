import math
import os
import random

import PIL.Image as PI
import PIL.ImageDraw as PD
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.contrib.keras as K

import video_utils as VU

IMG_SIZE = 128


def polar_circle(angle, dist=IMG_SIZE / 2.5, radius=IMG_SIZE // 10):
    x = math.cos(angle) * dist + IMG_SIZE/2
    y = math.sin(angle) * dist + IMG_SIZE/2

    return carthesian_circle(x, y, radius)


def carthesian_circle(x, y, rad=IMG_SIZE//10, img=None):
    if img is None:
        img = PI.new('L', (IMG_SIZE, IMG_SIZE))
    draw = PD.Draw(img)

    hr = rad//2
    draw.ellipse((x-hr, y-hr, x+hr, y+hr), fill='white', outline='white')
    return img


def render_spiral_move(num_frames=10000, radius_mean=10, radius_std=0):
    for frame in range(0, num_frames):
        rad = np.random.normal(radius_mean, radius_std)
        if rad <= 1:
            rad = 1
        img = polar_circle(angle=frame * (2 * math.pi) / 100, dist=10 + (IMG_SIZE / 2 - 20) / (1 + num_frames) * frame, radius=rad)
        yield img


def render_linear_move(num_frames=100):
    for fr in range(num_frames):
        img = carthesian_circle(10 + (IMG_SIZE-20)/num_frames*fr, IMG_SIZE/2)
        yield img


class DynaModelMF:
    def __init__(self, name, num_frames=2):
        self.name = name
        self.num_frames = num_frames
        self.step = tf.Variable(0, trainable=False, name='step')

        inp_batch_sh = [None, IMG_SIZE, IMG_SIZE, num_frames]
        out_batch_sh = [None, IMG_SIZE, IMG_SIZE]

        self.p_inp_frames = tf.placeholder(dtype=tf.float32, shape=inp_batch_sh)
        self.p_tgt_frame = tf.placeholder(dtype=tf.float32, shape=out_batch_sh)

        self.t_frame_predict = self.build_model(self.p_inp_frames)

        self.loss = tf.losses.mean_squared_error(self.t_frame_predict, self.p_tgt_frame)
        tf.summary.scalar('loss', self.loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, global_step=self.step)

        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cfg)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=20)
        self.writer = tf.summary.FileWriter(os.path.join('logs', self.name))

    def build_model(self, inp):
        ks = 3
        fs = 8

        x = K.layers.Input(tensor=inp)
        x = K.layers.Conv2D(filters=fs, kernel_size=ks, strides=2, activation='relu', padding='same')(x)
        x = K.layers.Conv2D(filters=fs, kernel_size=ks, strides=2, activation='relu', padding='same')(x)
        x = K.layers.Conv2D(filters=2, kernel_size=ks, strides=2, activation='relu', padding='same')(x)

        # x = K.layers.Flatten()(x)
        # x = K.layers.Dense(512)(x)
        # x = K.layers.Reshape((IMG_SIZE // 8, IMG_SIZE // 8, 2))(x)

        x = K.layers.Conv2DTranspose(filters=fs, kernel_size=ks, strides=2, activation='relu', padding='same')(x)
        x = K.layers.Conv2DTranspose(filters=fs, kernel_size=ks, strides=2, activation='relu', padding='same')(x)
        x = K.layers.Conv2DTranspose(filters=1, kernel_size=ks, strides=2, activation='relu', padding='same')(x)

        x = K.layers.Reshape((IMG_SIZE, IMG_SIZE))(x)
        return x

    def predict(self, frame):
        if np.ndim(frame) == 3:
            expanded = True
            frame = np.expand_dims(frame, axis=0)
        elif np.ndim(frame) == 4:
            expanded = False
        else:
            raise Exception("unknown dims")

        res = self.sess.run(self.t_frame_predict, feed_dict={self.p_inp_frames: frame, K.backend.learning_phase(): False})
        res = np.maximum(np.minimum(res, 1), 0)

        if expanded:
            res = np.squeeze(res, axis=0)

        return res

    def get_step(self):
        return self.sess.run(self.step)

    def train(self, inp_frames, tgt_frames, num_epochs=10, batch_size=8, on_epoch_finish=None):
        frame_pile = np.stack(inp_frames)

        n = len(inp_frames)
        inp_frame_seqs = []
        for k in range(self.num_frames):
            inp_frame_seqs.append(frame_pile[k:n - self.num_frames + k])
        inp_frame_seqs.append(tgt_frames[k+1:n - self.num_frames + k + 1])

        frames = np.stack(inp_frame_seqs)
        frames = frames.transpose((1,2,3,0))

        for ep in range(num_epochs):
            np.random.shuffle(frames)

            num_batches = frames.shape[0] // batch_size
            batches = frames[0:num_batches*batch_size]

            print("Epoch", ep)
            for batch in np.split(batches, batch_size):
                inp_frames = batch[:, :, :, :-1]
                tgt_frames = batch[:, :, :, -1]

                step = self.get_step()
                targets = {'loss': self.loss, 'train_op': self.train_op}
                if step % 10 == 0:
                    targets['summary'] = tf.summary.merge_all()

                res = self.sess.run(targets,
                                    feed_dict={self.p_inp_frames: inp_frames, self.p_tgt_frame: tgt_frames, K.backend.learning_phase(): True})

                if 'summary' in res:
                    self.writer.add_summary(res['summary'], global_step=step)

                if step % 100 == 0:
                    print('iter', step, 'loss', res['loss'])

            if on_epoch_finish:
                on_epoch_finish(ep)

    def checkpoint_dir(self):
        return os.path.join('models', self.name)

    def save(self):
        print("Saved checkpoint")
        path = self.checkpoint_dir()
        os.makedirs(path, exist_ok=True)
        self.saver.save(self.sess, os.path.join(path, 'model'))

    def restore(self):
        path = tf.train.latest_checkpoint(self.checkpoint_dir())
        if path:
            print("Restoring checkpoint", path)
            self.saver.restore(self.sess, path)
        else:
            print("Starting afresh")


name = 'conv3_dense512_deconv3_fs8_ks3_spiral_noise'

vu = VU.VideoWriter(name + ".mp4", show=True)


def draw_frames(frames):
    global vu

    for fr in frames:
        vu.consume(fr)

def experiment():
    dyna = DynaModelMF(name)
    dyna.restore()

    frames = []
    for img in render_spiral_move(num_frames=1024):
        frames.append(np.array(img) / 512.0)

    rnd_rad_frames = []
    for img in render_spiral_move(num_frames=8192, radius_mean=10, radius_std=1):
        noise = np.random.normal(0, 0.01, (IMG_SIZE, IMG_SIZE))
        rnd_rad_frames.append(np.array(img) / 512.0 * np.random.normal(1, 0.05) + noise)

    #draw_frames(frames)
    #exit(0)

    def test_prediction(ep):
        if ep % 5 != 0:
            return

        episode = 500

        start_frame = random.randint(0, len(frames) - episode*2)
        pred_one_step = dyna.predict(frames[start_frame: start_frame+episode])
        pred_far = far_prediction(rnd_rad_frames[start_frame], episode)
        gap = np.zeros((IMG_SIZE, 10), dtype=np.float32)
        draw_frames(map(lambda ps: np.concatenate([ps[0], gap, ps[1]], axis=1), zip(pred_one_step, pred_far)))

        dyna.save()

    def far_prediction(fr, episode_len):
        pred_frames = []
        for i in range(episode_len):
            fr = dyna.predict(fr)
            pred_frames.append(fr)
        return pred_frames


    dyna.train(rnd_rad_frames, frames, on_epoch_finish=test_prediction, batch_size=64, num_epochs=100000)


def demo():
    m = 10

    dyna = DynaModelMF(name)
    dyna.restore()

    for i in range(100):
        img = None

        for k in range(np.random.randint(1,5)):
            x = np.random.randint(m, IMG_SIZE - m)
            y = np.random.randint(m, IMG_SIZE - m)
            img = carthesian_circle(x, y, rad=np.random.randint(8, 12), img=img)

        frame = np.array(img) / 512

        for i in range(20):
            print(i)
            vu.consume(frame)
            frame = dyna.predict(frame)


def test():
    dyna = DynaModelMF(name,5)
    dyna.restore()

    frames = []
    for i in range(10):
        frames.append(np.ones((IMG_SIZE, IMG_SIZE))*i)

    rnd_rad_frames = []
    for i in range(10):
        rnd_rad_frames.append(np.ones((IMG_SIZE, IMG_SIZE)) * i * (-1))

    dyna.train(rnd_rad_frames, frames, batch_size=3, num_epochs=100000)


if __name__ == '__main__':
    try:
        #test()
        experiment()
        #demo()
    finally:
        print("Closing video")
        vu.close()