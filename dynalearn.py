import math
import os
import random

import PIL.Image as PI
import PIL.ImageDraw as PD
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.contrib.keras as K

IMG_SIZE=128


def polar_circle(angle, dist=IMG_SIZE / 2.5, radius=IMG_SIZE // 10):
    x = math.cos(angle) * dist + IMG_SIZE/2
    y = math.sin(angle) * dist + IMG_SIZE/2

    return carthesian_circle(x, y, radius)


def carthesian_circle(x, y, rad=IMG_SIZE//10):
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
        img = polar_circle(angle=frame * (2 * math.pi) / 100, dist=20 + IMG_SIZE / 3 / (1 + num_frames - frame), radius=rad)
        yield img


def render_linear_move(num_frames=100):
    for fr in range(num_frames):
        img = carthesian_circle(10 + (IMG_SIZE-20)/num_frames*fr, IMG_SIZE/2)
        yield img


class DynaModel:
    def __init__(self, name):
        self.name = name

        self.step = tf.Variable(0, trainable=False, name='step')
        self.p_frames1 = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE])
        self.p_frames2 = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE])
        self.t_frame_predict = self.build_model(self.p_frames1)

        self.loss = tf.losses.mean_squared_error(self.t_frame_predict, self.p_frames2)
        tf.summary.scalar('loss', self.loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, global_step=self.step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(os.path.join('logs', self.name))

    def build_model(self, inp):
        ks = 3
        fs = 8

        x = K.layers.Input(tensor=inp)
        x = K.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)
        #x = K.layers.BatchNormalization()(x)
        x = K.layers.Conv2D(filters=fs, kernel_size=ks, strides=2, activation='relu', padding='same')(x)
#        x = K.layers.BatchNormalization()(x)
        x = K.layers.Conv2D(filters=fs, kernel_size=ks, strides=2, activation='relu', padding='same')(x)
#        x = K.layers.BatchNormalization()(x)
        x = K.layers.Conv2D(filters=2, kernel_size=ks, strides=2, activation='relu', padding='same')(x)

        # x = K.layers.Flatten()(x)
        # x = K.layers.Dense(100)(x)
        # x = K.layers.Dense(2 * IMG_SIZE ** 2 // 4 // 4 // 4)(x)
        # x = K.layers.Reshape((IMG_SIZE // 8, IMG_SIZE // 8, 2))(x)

        x = K.layers.Conv2DTranspose(filters=fs, kernel_size=ks, strides=2, activation='relu', padding='same')(x)
        x = K.layers.Conv2DTranspose(filters=fs, kernel_size=ks, strides=2, activation='relu', padding='same')(x)
        x = K.layers.Conv2DTranspose(filters=1, kernel_size=ks, strides=2, activation='relu', padding='same')(x)

        x = K.layers.Reshape((IMG_SIZE, IMG_SIZE))(x)

        #x = K.layers.Conv2D(filters=16, kernel_size=3, strides=2, activation='relu')(x)
        #x = K.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='relu')(x)
        #x = K.layers.Flatten()(x)
        #x = K.layers.Dense(100, activation='relu')(x)
        #x = K.layers.Dense(IMG_SIZE*IMG_SIZE, activation='relu')(x)
#        x = K.layers.Reshape((IMG_SIZE, IMG_SIZE))(x)
        return x

    def predict(self, frame):
        if np.ndim(frame) == 2:
            expanded = True
            frame = np.expand_dims(frame, axis=0)
        else:
            expanded = False
        res = self.sess.run(self.t_frame_predict, feed_dict={self.p_frames1: frame, K.backend.learning_phase(): False})

        if expanded:
            res = np.squeeze(res, axis=0)

        return res

    def get_step(self):
        return self.sess.run(self.step)

    def train(self, frames1, frames2, num_epochs=10, batch_size=8, on_epoch_finish=None):
        frames = list(zip(frames1, frames2))
        random.shuffle(frames)

        for ep in range(num_epochs):
            print("Epoch", ep)
            for idx in range(0, len(frames), batch_size):
                step = self.get_step()
                targets = {'loss': self.loss, 'train_op': self.train_op}
                if step % 10 == 0:
                    targets['summary'] = tf.summary.merge_all()

                batch = frames[idx:idx+batch_size]
                bfs1 = list(map(lambda f: f[0], batch))
                bfs2 = list(map(lambda f: f[1], batch))
#                plt.imshow(bfs1[10] - bfs2[10])

                res = self.sess.run(targets,
                                    feed_dict={self.p_frames1: bfs1, self.p_frames2: bfs2, K.backend.learning_phase(): True})

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


axis = None
def draw_frames(frames):
    global axis

    plt.ion()
    for fr in frames:
        if axis is None:
            axis = plt.imshow(fr)
            plt.show()
        else:
            axis.set_data(fr)
        plt.pause(0.01)


def experiment():
    dyna = DynaModel('conv3_deconv3_fs8_ks3_spiral_std_1')
    dyna.restore()

    frames = []
    for img in render_spiral_move(num_frames=8192):
        frames.append(np.array(img) / 255.0)

    rnd_rad_frames = []
    for img in render_spiral_move(num_frames=8192, radius_mean=10, radius_std=1):
        rnd_rad_frames.append(np.array(img) / 255.0)

    #draw_frames(rnd_rad_frames)
    #exit(0)

    def test_prediction(ep):
        if ep % 5 != 0:
            return

        pred_one_step = dyna.predict(frames[0:100])
        pred_far = far_prediction()
        gap = np.zeros((IMG_SIZE, 10), dtype=np.float32)
        draw_frames(map(lambda ps: np.concatenate([ps[0], gap, ps[1]], axis=1), zip(pred_one_step, pred_far)))

        dyna.save()

    def far_prediction():
        fr = frames[10]
        pred_frames = []
        for i in range(100):
            fr = dyna.predict(fr)
            pred_frames.append(fr)
        return pred_frames


    #draw_frames(frames[::10])

    dyna.train(rnd_rad_frames[:-1], frames[1:], on_epoch_finish=test_prediction, batch_size=64, num_epochs=100000)


if __name__ == '__main__':
    experiment()
