import os

import numpy as np

size = 100

def make_dataset():
    np.random.seed(1202)

    for type in ["train", "validation"]:
        if not os.path.isdir("dataset/"):
            os.makedirs("dataset/")

        x = np.random.normal(1.0, 0.5, 100)
        y = np.random.normal(1.0, 0.5, 100)

        with open("dataset/SGLD_LR/%s.csv" % type, "w") as f:
            for i in range(size):
                f.write("%f,%f,%f\n" % (x[i], y[i], -1))

        x = np.random.normal(-1.0, 0.5, 100)
        y = np.random.normal(-1.0, 0.5, 100)

        with open("dataset/SGLD_LR/%s.csv" % type, "a") as f:
            for i in range(size):
                f.write("%f,%f,%f\n" % (x[i], y[i], 1))

if __name__ == "__main__":
	make_dataset()
