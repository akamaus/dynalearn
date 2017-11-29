import math
import random

import PIL.Image as PI
import PIL.ImageDraw as PD
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.contrib.keras as K

IMG_SIZE=128


def render_circle(angle, dist=IMG_SIZE/2.5, rad=IMG_SIZE//10):
    img = PI.new('L', (IMG_SIZE, IMG_SIZE))
    draw = PD.Draw(img)

    x = math.cos(angle) * dist + IMG_SIZE/2
    y = math.sin(angle) * dist + IMG_SIZE/2

    hr = rad//2
    draw.ellipse((x-hr, y-hr, x+hr, y+hr), fill='white', outline='white')
    return img


def render_spiral(num_frames=10000):
    for frame in range(0, num_frames):
        img = render_circle(angle=frame * (2 * math.pi) / 100, dist=20 + IMG_SIZE/3/(1+num_frames - frame))
        yield img


class DynaModel:
    def __init__(self):
        self.p_frames1 = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE])
        self.p_frames2 = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE])
        self.t_frame_predict = self.build_model(self.p_frames1)

        self.loss = tf.losses.mean_squared_error(self.t_frame_predict, self.p_frames2)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.iters = 0


    def build_model(self, inp):
        x = K.layers.Input(tensor=inp)
        x = K.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)
        x = K.layers.Conv2D(filters=4, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        x = K.layers.Conv2D(filters=8, kernel_size=3, strides=2, activation='relu', padding='same')(x)

        x = K.layers.Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        x = K.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, activation='relu', padding='same')(x)

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
            frame = np.expand_dims(frame, axis=0)

        return self.sess.run(self.t_frame_predict, feed_dict={self.p_frames1: frame})

    def train(self, frames1, frames2, num_epochs=10, batch_size=8, on_epoch_finish=None):
        frames = list(zip(frames1, frames2))
        random.shuffle(frames)

        for ep in range(num_epochs):
            print("Epoch", ep)
            for idx in range(0, len(frames), batch_size):
                res = self.sess.run({'loss': self.loss, 'train_op': self.train_op},
                                    feed_dict={self.p_frames1: frames[idx:idx+batch_size][0],
                                               self.p_frames2: frames[idx:idx+batch_size][1]})
                if self.iters % 100 == 0:
                    print('iter', self.iters, 'loss', res['loss'])
                self.iters += 1

            if on_epoch_finish:
                on_epoch_finish(ep)

    def save(self):
        print("Saved checkpoint")
        self.saver.save(self.sess, "models/model1")



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
    dyna = DynaModel()

    frames = []
    for img in render_spiral():
        frames.append(np.array(img) / 255.0)

    def test_prediction(ep):
        if ep % 10 != 0:
            return

        pred_frames = dyna.predict(frames[0:100])
        draw_frames(pred_frames)
        dyna.save()

    #draw_frames(frames[::10])

    dyna.train(frames[:-1], frames[1:], on_epoch_finish=test_prediction, batch_size=64, num_epochs=100)


if __name__ == '__main__':
    experiment()
