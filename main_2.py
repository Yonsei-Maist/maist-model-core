from model.examples.gan import DCGanExample, GanExample, Pix2pixExample

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf

example = Pix2pixExample("F:\\data\\cityscapes")

epochs = [1, 100, 250, 500]

with tf.device("/cpu:0"):
    loss, res = example.run(index=1)
    for epoch in epochs:
        seed = example.create_fake_seed(16)
        loss, res = example.run(index=epoch, train=False, seed=seed)

        plot.figure(figsize=(10, 10))
        for i in range(res.shape[0]):
            plot.subplot(4, 4, i + 1)
            img = res[i, :, :, :]
            plot.imshow((img - np.min(img)) / (np.max(img) - np.min(img)))
            plot.axis("off")

        plot.tight_layout()
        plot.savefig("F:\\result\\pix2pix\\{}_epoch.png".format(epoch))
        plot.close('all')
