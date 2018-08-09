import numpy as np

import utils
import metrics
from get_data import get_data


class Tester(object):
    def __init__(self, sess, ram):
        self.ram = ram
        self.sess = sess
        self.number_examples = 600
    

    def evaluate_model(self):
        
        self.ram.is_training = False
        all_predictions = []
        all_labels = []

        for steps, data in enumerate(get_data()):
            images, labels = data
            
            predictions, locs = self.sess.run(
                [self.ram.prediction, self.ram.locs],
                feed_dict={
                    self.ram.img_ph: images,
                    self.ram.lbl_ph: labels,
                }
            )
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            if steps > self.number_examples:
                reduction = metrics.get_reduction(all_predictions, all_labels)
                recall = metrics.get_recall(all_predictions, all_labels)
                accuracy = metrics.get_accuracy(all_predictions, all_labels)
                break
        return reduction, recall, accuracy


