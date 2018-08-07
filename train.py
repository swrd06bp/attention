from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import tensorflow as tf
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import logging
import numpy as np

import utils
from model import RecurrentAttentionModel
from metrics import get_recall, get_precision

logging.getLogger().setLevel(logging.INFO)

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97,
        "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 8, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 128, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 256, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("cell_size", 256, "Size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 6, "Number of glimpses.")
tf.app.flags.DEFINE_float("variance", 0.22, "Gaussian variance for Location Network.")
tf.app.flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")

FLAGS = tf.app.flags.FLAGS

training_steps_per_epoch = mnist.train.num_examples // FLAGS.batch_size




ram = RecurrentAttentionModel(img_shape=(28, 28, 3), # MNIST: 28 * 28 
    pth_size=FLAGS.patch_window_size,
    g_size=FLAGS.g_size,
    l_size=FLAGS.l_size,
    glimpse_output_size=FLAGS.glimpse_output_size,
    loc_dim=3,   # (x, y, number of images)
    variance=FLAGS.variance, 
    cell_size=FLAGS.cell_size,
    num_glimpses=FLAGS.num_glimpses,
    num_classes=10,
    learning_rate=FLAGS.learning_rate,
    learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
    min_learning_rate=FLAGS.min_learning_rate,
    training_steps_per_epoch=training_steps_per_epoch,
    max_gradient_norm=FLAGS.max_gradient_norm, 
    is_training=True)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "/tmp/model.ckpt")
    # print("Model restored.")

    for step in xrange(FLAGS.num_steps):
        ram.is_training = True
        images, labels = mnist.train.next_batch(FLAGS.batch_size)
        images = np.reshape(images, [-1, 28, 28, 1])

        # labels_bak = deepcopy(labels)
        # labels[np.where(labels_bak>1)[0]] = 0
        # labels[np.where(labels_bak<=1)[0]] = 1
        images = np.tile(images, [FLAGS.M, 1, 1, 3])
        labels = np.tile(labels, [FLAGS.M])

        output_feed = [ram.train_op, ram.loss, ram.xent, ram.reward, ram.advantage, ram.baselines_mse, ram.learning_rate, ram.locs]
        _, loss, xent, reward, advantage, baselines_mse, learning_rate, locs = sess.run(output_feed,
                feed_dict={
                ram.img_ph: images,
                ram.lbl_ph: labels,
                })
        if step and step % 100 == 0:
            logging.info(
                'step {}: lr = {:3.6f}\tloss = {:3.4f}\txent = {:3.4f}\treward = {:3.4f}\tadvantage = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
                    step, learning_rate, loss, xent, reward, advantage, baselines_mse))
            locs_first_image  = np.array([loc[0, :] for loc in locs])
            requests.post("http://127.0.0.1:8888/bboxes", data=str(utils.transform_locs_to_bboxes(locs_first_image, (8,8), (32, 32), 3)))
            requests.post("http://127.0.0.1:8888/image", data=utils.transform_images(images[0]))
            requests.post("http://127.0.0.1:8888/prediction", data=str(labels[0]))

                        # Evaluationif step and step % training_steps_per_epoch ==0:
        if step and step % training_steps_per_epoch ==0:
            
            for dataset in [mnist.validation, mnist.test]:
                ram.is_training = False
                steps_per_epoch = dataset.num_examples // FLAGS.batch_size
                correct_cnt = 0
                all_labels = []
                all_predictions = []
                num_samples = steps_per_epoch * FLAGS.batch_size
                for test_step in xrange(steps_per_epoch):
                    images, labels = dataset.next_batch(FLAGS.batch_size)
                    images = np.reshape(images, [-1, 28, 28, 1])
                    images = np.tile(images, [FLAGS.M, 1, 1, 3])
                    labels_bak = labels
                    labels = np.tile(labels, [FLAGS.M])

                    softmax, locs = sess.run([ram.softmax, ram.locs],
                          feed_dict={
                              ram.img_ph: images,
                              ram.lbl_ph: labels
                              })
                    softmax = np.reshape(softmax, [FLAGS.M, -1, 10])
                    softmax = np.mean(softmax, 0)
                    prediction = np.argmax(softmax, 1).flatten()
                    correct_cnt += np.sum(prediction == labels_bak)
                acc = correct_cnt / num_samples
                if dataset == mnist.validation:
                    logging.info('valid accuracy = {}'.format(acc))
                else:
                    logging.info('test accuracy = {}'.format(acc))

# requests.post("http://127.0.0.1:8888/image", data=base64.b64encode(open('test.jpg', 'rb').read()))

          # labels_bak = deepcopy(labels)
          # labels[np.where(labels_bak>1)[0]] = 0
          # labels[np.where(labels_bak<=1)[0]] = 1
          # labels_bak = labels

          # Duplicate M times
          # images = np.tile(images, [FLAGS.M, 1, 1, 3])
          # labels = np.tile(labels, [FLAGS.M])
          # softmax = sess.run(ram.softmax,
                                # feed_dict={
                                  # ram.img_ph: images,
                                  # ram.lbl_ph: labels
                                # })
          # softmax = np.reshape(softmax, [FLAGS.M, -1, 2])
          # softmax = np.mean(softmax, 0)
          # prediction = np.argmax(softmax, 1).flatten()
          # correct_cnt += np.sum(prediction == labels_bak)
          # all_predictions.extend(prediction)
          # all_labels.extend(labels_bak)

        # acc = correct_cnt / num_samples
        # precision = get_precision(all_predictions, all_labels)
        # recall = get_recall(all_predictions, all_labels)
        # if dataset == mnist.validation:
          # logging.info('valid accuracy = {}'.format(acc))
          # logging.info('valid precision = {}'.format(precision))
          # logging.info('valid recall = {}'.format(recall))
        # else:
          # logging.info('test accuracy = {}'.format(acc))
          # logging.info('test precision = {}'.format(precision))
          # logging.info('test recall = {}'.format(recall))
      # save_path = saver.save(sess, "/tmp/model.ckpt")
      # print("Model saved in file: %s" % save_path)
