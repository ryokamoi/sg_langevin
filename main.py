import os
import shutil
import argparse
import datetime

import tensorflow as tf

import model
from get_dataset import get_dataset
from visualizer import Visualizer

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Stochastic Gradient Langevin Dynamics')
parser.add_argument('--hparams', type=str, default=None,
                    help='The name of a file containing comma separated list of "name=value" pairs.')
args = parser.parse_args()

tf.set_random_seed(1202)

def main():
    # train logistic regression with stocastic gradient Langevin Gradient

    if not os.path.isdir("log/"):
        os.makedirs("log/")
    now = datetime.datetime.today()
    logdir = "log/log%s/" % now.strftime("%Y%m%d-%H%M")
    os.makedirs(logdir)

    # tensorboard
    writer = tf.contrib.summary.create_file_writer(logdir)
    global_step=tf.train.get_or_create_global_step()
    writer.set_as_default()

    # read hyperparameters from file
    hparams = tf.contrib.training.HParams(
                lr=0.1,
                model="SGLD_LR",
                epoch=30,
                batch_size=10)

    if args.hparams:
        shutil.copyfile(args.hparams, logdir + "params")
        hparams_from_file = ""
        with open(args.hparams, "r") as f:
            for l in f.readlines():
                hparams_from_file += l

    hparams.parse(hparams_from_file)

    # choose model
    if hparams.model == "SGLD_LR":
        nn = model.SGLD_LR(hparams)
        train_dataset, train_dataset_size = get_dataset(hparams.model, "train")
        val_dataset, val_dataset_size = get_dataset(hparams.model, "validation")
    else:
        raise "Invalid parameter for hparams.model"

    visualizer = Visualizer()

    # train
    epsilon_ = hparams.lr
    step = 0
    for epoch in range(hparams.epoch):
        train_dataset_iter = train_dataset.shuffle(train_dataset_size).batch(hparams.batch_size)

        for batch, data in enumerate(train_dataset_iter):
            global_step.assign_add(1)
            step += 1

            epsilon_ = hparams.lr / (1 + 0.05 * step)
            epsilon = tf.convert_to_tensor(epsilon_, tf.float32)

            loss = nn.loss(data["data"], data["label"]).numpy()
            accuracy = nn.accuracy(data["data"], data["label"]).numpy()

            visualizer.store_results(nn)

            nn.update(data["data"], data["label"], epsilon, train_dataset_size)

            with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('accuracy', accuracy)
                tf.contrib.summary.scalar('epsilon', epsilon)

                grads_vars = nn.grads_variances()
                for i in range(len(grads_vars)):
                    tf.contrib.summary.scalar('grads_var%d' % (i+1), grads_vars[i])

        print("epoch %2d\tbatch %4d\tloss %f\taccuracy %f" % (epoch+1, batch+1, loss, accuracy))

    for l_epoch in range(100):
        print("langevin epoch %d" % (l_epoch+1))
        train_dataset_iter = train_dataset.shuffle(train_dataset_size).batch(hparams.batch_size)

        for batch, data in enumerate(train_dataset_iter):
            visualizer.store_results(nn)
            nn.update(data["data"], data["label"], epsilon, train_dataset_size)

    # visualize
    result = visualizer.retrieve_results()
    print(result)

if __name__ == "__main__":
    main()
