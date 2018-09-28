import numpy as np
import tensorflow as tf

class Visualizer(object):
    def __init__(self):
        self.results = []

        self.num = 20
        x = np.linspace(-4, 4, self.num)
        y = np.linspace(-4, 4, self.num)
        self.xx, self.yy = np.meshgrid(x, y)
        xy = np.vstack((np.reshape(self.xx, -1), np.reshape(self.yy, -1)))
        self.input = tf.convert_to_tensor(np.transpose(xy), tf.float32)

    def store_results(self, nn):
        result = nn(self.input).numpy()
        label = np.maximum(np.sign(result, dtype=np.float32), 0.0)
        self.results.append(label)
        if len(self.results) > 10000:
            self.results.pop(0)

    def retrieve_results(self):
        return np.reshape(np.average(self.results, axis=0), [self.num, self.num])
