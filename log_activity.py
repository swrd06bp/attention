import logging

import utils

logging.getLogger().setLevel(logging.INFO)

def render_evaluation(epoch, reduction, recall, accuracy):
    text_log = "epoch {epoch} -- reduction: {reduction}% - recall: {recall}% - accuracy: {accuracy}%"\
        .format(
            epoch=epoch,
            reduction=reduction,
            recall=recall,
            accuracy=accuracy,
        )
    logging.info(text_log)
    utils.add_logging(text_log)
    utils.add_accuracy(epoch, reduction, recall, accuracy)

def render_training_steps(steps, learning_rate, losses, xents, rewards,
    advantages, baselines_mses):
    
    text_log = 'step {}: lr = {:3.6f}\tloss = {:3.4f}\txent = {:3.4f}\treward = {:3.4f}\tadvantage = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
        steps,
        learning_rate,
        losses, xents,
        rewards, advantages,
        baselines_mses
    )

    logging.info(text_log)
    utils.add_reward(steps, rewards)
    utils.add_logging(text_log)

