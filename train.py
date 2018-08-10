from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import utils
import log_activity
from test import Tester
from get_data import get_data
from model import RecurrentAttentionModel



tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97,
        "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 32, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 128, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 256, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("cell_size", 256, "Size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 10, "Number of glimpses.")
tf.app.flags.DEFINE_float("variance", 0.22, "Gaussian variance for Location Network.")
tf.app.flags.DEFINE_integer("M", 30, "Monte Carlo sampling, see Eq(2).")

FLAGS = tf.app.flags.FLAGS




ram = RecurrentAttentionModel(img_shape=(240, 320, 3), # MNIST: 28 * 28 
    pth_size=FLAGS.patch_window_size,
    g_size=FLAGS.g_size,
    l_size=FLAGS.l_size,
    glimpse_output_size=FLAGS.glimpse_output_size,
    loc_dim=3,   # (x, y, number of images)
    variance=FLAGS.variance, 
    cell_size=FLAGS.cell_size,
    num_glimpses=FLAGS.num_glimpses,
    num_classes=2,
    learning_rate=FLAGS.learning_rate,
    learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
    min_learning_rate=FLAGS.min_learning_rate,
    training_steps_per_epoch=600,
    max_gradient_norm=FLAGS.max_gradient_norm, 
    is_training=True)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()



def train():
    #initialisation
    rewards = 0
    advantages = 0
    baselines_mses = 0
    losses = 0
    xents = 0
    epochs = 0
    number_examples = 600
    
    utils.reset()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "/tmp/model.ckpt")
        # print("Model restored.")

        for step, data in enumerate(get_data()):
            
            ram.is_training = True
            images, labels = data
            
            images = np.tile(images, [FLAGS.M, 1, 1, 1])
            labels = np.tile(labels, [FLAGS.M])

            output_feed = [
                ram.train_op,
                ram.prediction,
                ram.loss,
                ram.xent,
                ram.reward,
                ram.advantage,
                ram.baselines_mse,
                ram.learning_rate,
                ram.locs
            ]
            
            _, prediction_model, loss, xent, reward, advantage, baselines_mse,\
                learning_rate, locs = sess.run(output_feed,
                    feed_dict={
                    ram.img_ph: images,
                    ram.lbl_ph: labels,
                    })

            losses += loss
            advantages += advantage
            baselines_mses += baselines_mse
            rewards += reward 
            xents += xent

            if step and step % 10 == 0:
                utils.render_results(images, locs, prediction_model, labels)

            if step and step % 100 == 0:
                rewards /= 100
                advantages /= 100
                xents /= 100
                losses /= 100
                baselines_mses /= 100
                
                log_activity.render_training_steps(step, learning_rate,
                    losses, xents, rewards, advantages, baselines_mses)

                rewards = 0
                advantages = 0
                baselines_mses = 0
                losses = 0
                xents = 0

            if step and step % number_examples == 0:
                reduction, recall, accuracy = Tester(sess, ram).evaluate_model()
                log_activity.render_evaluation(epochs, reduction, recall, accuracy) 
                epochs += 1
    return sess
    
def save_model(ram):
      save_path = saver.save(sess, "/tmp/model.ckpt")
      print("Model saved in file: %s" % save_path)

if __name__ == "__main__":
    sess = train()
    save_model(sess)
