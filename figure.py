from typing import Dict

from matplotlib import pyplot as plt
import numpy as np

class Figure:
    """ Window with learning stats"""

    def __init__(self, graph_names, title=None):
        if not plt.isinteractive():
            plt.ion()

        self.fig = plt.figure()
        self.graphs = {}

        for i, gn in enumerate(graph_names):
            ax = self.fig.add_subplot(len(graph_names), 1, i + 1)
            ax.set_ylabel(gn)
            pl = None
            self.graphs[gn] = [ax, pl]

        if title is not None:
            self.fig.canvas.set_window_title(title)
        plt.show()

    def plot(self, graph_name, data):
        """ Update single graph"""
        graph = self.graphs[graph_name]
        assert graph
        if graph[1] is None:
            graph[1] = graph[0].plot(data)[0]
        else:
            graph[1].set_xdata(np.arange(0,len(data)))
            graph[1].set_ydata(data)
            graph[0].relim()
            graph[0].autoscale_view()

    def draw(self):
        """ Redraw after updates"""
        self.fig.canvas.draw()
        plt.pause(0.00001)  # http://stackoverflow.com/a/24228275
