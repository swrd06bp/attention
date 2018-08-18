import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal


def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def _log_likelihood(loc_means, locs, variance):
    loc_means = tf.stack(loc_means)  # [timesteps, batch_sz, loc_dim]
    locs = tf.stack(locs)
    gaussian = Normal(loc_means, variance)
    logll = gaussian._log_prob(x=locs)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    return tf.transpose(logll)      # [batch_sz, timesteps]

class RetinaSensor(object):
    # one scale
    def __init__(self, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size

    def __call__(self, img_ph, loc, init=False):
        # # change loc[:, 2] to be between 0 and 2
        # # ex nb_loc = 1.2
        nb_loc =  tf.add(loc[:, 2], 1)
        # # change nb_loc to be between 0 and 2*img_ph.shape[-1]
        # # ex if img_ph.shape[-1] = 3, nb_loc = 3.6
        nb_loc = tf.multiply(tf.cast(img_ph.shape[-1], tf.float32), nb_loc)
        # # change nb_loc to be between 0 and img_ph.shape[-1]
        # # ex: nb_loc = 1.8
        # # change nb_loc to be an int which could be either
        # # 0, 1, .. , img_ph.shape[-1] - 1
        # # ex: nb_loc = 1
        nb_loc = tf.cast(tf.divide(nb_loc, 2), tf.uint8)
        # # change nb_loc to be on hot with a depth of 1
        # # ex nb_loc = [0, 1, 0]
        nb_loc = tf.one_hot(nb_loc, 3)
        nb_loc = tf.expand_dims(nb_loc, 1)  
        nb_loc = tf.expand_dims(nb_loc, 1)  
        nb_loc = tf.expand_dims(nb_loc, 1)  
        nb_loc = tf.tile(nb_loc, [1, self.pth_size, self.pth_size, 3, 1])
        # img_ph = tf.multiply(img_ph, nb_loc)
        # img_ph = tf.reduce_sum(img_ph, -1)

        img = tf.reshape(img_ph, [
          tf.shape(img_ph)[0],
          self.img_size[0],
          self.img_size[1],
          3
        ])
        if not init:
            pths = []
            for i in [1, 2, 4]:
                pth = tf.image.extract_glimpse(
                        img,
                        [int(self.pth_size), int(self.pth_size)],
                        loc[:, :2])
                pth = tf.image.resize_images(
                        pth,
                        [self.pth_size, self.pth_size]
                        )
                pth = tf.expand_dims(pth, -1)
                pths.append(pth)

            pth = tf.concat((pths[0], pths[1]), axis=4)
            pth = tf.concat((pth, pths[2]), axis=4)
            pth = tf.multiply(pth, nb_loc)
            pth = tf.reduce_sum(pth, -1)
        else:
            pth = tf.image.resize_images(
                img,
                [self.pth_size, self.pth_size],
            )

        return tf.reshape(pth, [tf.shape(loc)[0], 3*self.pth_size*self.pth_size])

class GlimpseNetwork(object):
    def __init__(self, img_size, pth_size, loc_dim, g_size, l_size, output_size):
        self.retina_sensor = RetinaSensor(img_size, pth_size)

        # layer 1
        self.g1_w = _weight_variable((3*pth_size*pth_size, g_size))
        self.g1_b = _bias_variable((g_size,))
        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))
        # layer 2 
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))
        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, imgs_ph, locs, init=False):
        
        pths = self.retina_sensor(imgs_ph, locs, init=init)

        g = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(pths, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        l = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(locs, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

        return tf.nn.relu(g + l)

class LocationNetwork(object):
    def __init__(self, loc_dim, rnn_output_size, variance=0.22, is_sampling=False):
        self.loc_dim = loc_dim
        self.variance = variance
        self.w = _weight_variable((rnn_output_size, loc_dim))
        self.b = _bias_variable((loc_dim,))

        self.is_sampling = is_sampling

    def __call__(self, cell_output, sampling=False):
        mean = tf.nn.xw_plus_b(cell_output, self.w, self.b)
        mean = tf.clip_by_value(mean, -1., 1.)
        mean = tf.stop_gradient(mean)

        if self.is_sampling or sampling:
            loc = mean + tf.random_normal(
                  (tf.shape(cell_output)[0], self.loc_dim), 
                  stddev=self.variance)
            loc = tf.clip_by_value(loc, -1., 1.)
        else:
            loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean

class RecurrentAttentionModel(object):
    def __init__(self, img_shape, pth_size, g_size, l_size, glimpse_output_size, 
               loc_dim, variance, 
               cell_size, num_glimpses, num_classes, 
               learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch,
               max_gradient_norm, 
               is_training=False):
        self.is_training = is_training
        self.img_ph = tf.placeholder(tf.float32, [None, img_shape[0], img_shape[1], img_shape[2]])
        self.lbl_ph = tf.placeholder(tf.int64, [None])

        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.maximum(tf.train.exponential_decay(
                                            learning_rate, self.global_step,
                                            training_steps_per_epoch,
                                            learning_rate_decay_factor,
                                            staircase=True),
                                        min_learning_rate)

        cell = BasicLSTMCell(cell_size)

        with tf.variable_scope('InitGlimpseNetwork'):
            init_glimpse_network = GlimpseNetwork(img_shape, pth_size, loc_dim, g_size, l_size, glimpse_output_size)
        with tf.variable_scope('GlimpseNetwork'):
            glimpse_network = GlimpseNetwork(img_shape, pth_size, loc_dim, g_size, l_size, glimpse_output_size)
        with tf.variable_scope('LocationNetwork'):
            location_network = LocationNetwork(loc_dim=loc_dim, rnn_output_size=cell.output_size, variance=variance, is_sampling=self.is_training)

    # Core Network
        batch_size = tf.shape(self.img_ph)[0]
        init_loc = tf.random_uniform((batch_size, loc_dim), minval=0, maxval=0)
        init_state = cell.zero_state(batch_size, tf.float32)

        init_glimpse = init_glimpse_network(self.img_ph, init_loc, init=True)
        rnn_inputs = [init_glimpse]
        rnn_inputs.extend([0] * num_glimpses)
        
        locs, loc_means = [], []
        def loop_function(prev, _):
            loc, loc_mean = location_network(prev, self.is_training)
            locs.append(loc)
            loc_means.append(loc_mean)
            glimpse = glimpse_network(self.img_ph, loc)
            return glimpse
        rnn_outputs, _ = rnn_decoder(rnn_inputs, init_state, cell, loop_function=loop_function)
        # to be displyed
        self.locs = locs 
    # Time independent baselines
        with tf.variable_scope('Baseline'):
            baseline_w = _weight_variable((cell.output_size, 1))
            baseline_b = _bias_variable((1,))
        baselines = []
        for output in rnn_outputs[1:]:
            baseline = tf.nn.xw_plus_b(output, baseline_w, baseline_b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)
        baselines = tf.stack(baselines)        # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)   # [batch_sz, timesteps]

        # Classification. Take the last step only.
        rnn_last_output = rnn_outputs[-1]
        with tf.variable_scope('Classification'):
            logit_w = _weight_variable((cell.output_size, num_classes))
            logit_b = _bias_variable((num_classes,))
        logits = tf.nn.xw_plus_b(rnn_last_output, logit_w, logit_b)
        self.prediction = tf.argmax(logits, 1)
        self.softmax = tf.nn.softmax(logits)

        if self.is_training:
            # classification loss
            #self.xent = focal_loss(logits, self.lbl_ph)#
            self.xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
            # RL reward
            reward = tf.cast(tf.equal(self.prediction, self.lbl_ph), tf.float32)
            # reward = tf.multiply(tf.cast(tf.equal(self.prediction, self.lbl_ph), tf.float32),0.1) + tf.multiply(tf.cast(tf.multiply(self.prediction, self.lbl_ph), tf.float32),0.9)
            rewards = tf.expand_dims(reward, 1)             # [batch_sz, 1]
            rewards = tf.tile(rewards, (1, num_glimpses))   # [batch_sz, timesteps]
            advantages = rewards - tf.stop_gradient(baselines)
            self.advantage = tf.reduce_mean(advantages)
            logll = _log_likelihood(loc_means, locs, variance)
            logllratio = tf.reduce_mean(logll * advantages)
            self.reward = tf.reduce_mean(reward)
            # baseline loss
            self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
            # hybrid loss
            self.loss = -logllratio + self.xent + self.baselines_mse
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)
