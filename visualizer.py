import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

class Visualizer(object):
    def __init__(self):
        self.results = []

        self.num = 40
        self.range = 6.0
        x = np.linspace(-self.range, self.range, self.num)
        y = np.linspace(-self.range, self.range, self.num)
        self.xx, self.yy = np.meshgrid(x, y)
        xy = np.vstack((np.reshape(self.xx, -1), np.reshape(self.yy, -1)))
        self.input = tf.convert_to_tensor(np.transpose(xy), tf.float32)

    def store_results(self, nn):
        result = nn(self.input).numpy()
        label = np.maximum(np.sign(result, dtype=np.float32), 0.0)
        self.results.append(label)
        if len(self.results) > 1000:
            self.results.pop(0)

    def retrieve_results(self):
        prob = np.average(self.results, axis=0)
        return np.flip(np.reshape(prob, [self.num, self.num]), 1)

    def save_results(self, log_dir, dataset):
        fig = plt.figure()
        log_file = log_dir + "heatmap.png"

        prob = self.retrieve_results()
        xticks = np.arange(-self.range, self.range+1)
        yticks = np.arange(-self.range, self.range+1)
        ax = fig.add_subplot(111, aspect='equal')
        sns.heatmap(prob, cbar=False, cmap="RdYlGn", square=True, ax=ax,
                    xticklabels=xticks, yticklabels=yticks[::-1])
        ax.set_xticks((xticks+self.range)*ax.get_xlim()[1]/(2*self.range))
        ax.set_yticks((yticks+self.range)*ax.get_xlim()[1]/(2*self.range))

        train_dataset_iter = dataset.batch(10)
        for dataset in train_dataset_iter:
            xy = np.transpose(dataset["data"].numpy())
            ax.scatter((xy[0]+self.range)*ax.get_xlim()[1]/(2*self.range),
                        (-xy[1]+self.range)*ax.get_ylim()[0]/(2*self.range), c="b", marker=".")
            print((xy[0]+self.range)*ax.get_xlim()[1]/(2*self.range))
            print((-xy[1]+self.range)*ax.get_ylim()[0]/(2*self.range))

        fig.savefig(log_file)
