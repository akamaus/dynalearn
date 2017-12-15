import imageio as ii
import numpy as np
import matplotlib.pyplot as plt


def concat_images(img1, img2):
    padder = np.zeros([img1.shape[0], 20, 3])
    return np.concatenate([img1, padder, img2], axis=1)


def video_reader(path, num_frames = None, rotate=False):
    reader = ii.get_reader(path)
    for k, img in enumerate(reader):
        #print("reading {} of {}".format(k, num_frames))
        if rotate:
            img = np.transpose(img, axes=[1, 0, 2])
            img = np.fliplr(img)

        if num_frames is None or k < num_frames:
            yield img
        else:
            raise StopIteration


class VideoWriter:
    def __init__(self, path=None, show=False):
        if path:
            self.writer = ii.get_writer(path, fps=30)
        else:
            self.writer = None
        self.show = show

        self.ax = None
        self.im = None
        if show:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.ion()

    def consume(self, img):
        if self.writer:
            self.writer.append_data(img)
        if self.show:
            if self.im is None:
                self.im = self.ax.imshow(img)
            else:
                self.im.set_data(img)
            plt.pause(0.01)

    def close(self):
        if self.writer:
            self.writer.close()
