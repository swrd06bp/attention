import json
import base64

from requests_futures.sessions import FuturesSession
import numpy as np
from PIL import Image

port = 8888
session = FuturesSession()

def transform_images(images):
    transformed_images = np.vstack((images[:, :, 0], images[:, :, 1])) 
    transformed_images = np.vstack((transformed_images, images[:, :, 2]))
    transformed_images = np.expand_dims(transformed_images, -1)
    transformed_images = np.tile(transformed_images, [1, 1, 3])
    transformed_images *= 255
    img = Image.fromarray(transformed_images.astype(np.uint8))
    img.save("image.png")
    transformed_images = base64.b64encode(open('image.png').read())
    return transformed_images


def transform_locs_to_bboxes(loc, focal_shape, image_shape, number_images):
    x_original = loc[:, 0]
    y_original = loc[:, 1]
    n_original = loc[:, 2]

    x_transformed_center = image_shape[1]*(x_original + 1)/2 
    y_transformed_center = image_shape[0]*(y_original + 1)/2 

    x_transform_1 = (x_transformed_center - focal_shape[1]/2)
    y_transform_1 = (y_transformed_center - focal_shape[0]/2)
    
    n_transform_1 = number_images*(n_original + 1)/2 

    return np.transpose(np.array([x_transform_1, y_transform_1, n_transform_1]).astype(int)).tolist()

def render_results(images, locs, predictions, labels):
    locs_first_image  = np.array([loc[0, :] for loc in locs])
    session.post("http://127.0.0.1:{}/bboxes".format(port), json=transform_locs_to_bboxes(locs_first_image, (8,8), (32, 32), 3))
    session.post("http://127.0.0.1:{}/image".format(port), data=transform_images(images[0]))
    session.post("http://127.0.0.1:{}/prediction".format(port), json=[predictions[0], labels[0]])

def add_logging(logs):
    session.post("http://127.0.0.1:{}/logging".format(port), data="{}\n".format(logs))

def add_reward(steps, reward):
    session.post("http://127.0.0.1:{}/reward".format(port), json=[steps, reward])

def add_accuracy(epoch, reduction, recall, accuracy):
    session.post("http://127.0.0.1:{}/accuracy".format(port), json=[epoch, reduction, recall, accuracy])

def reset():
    session.get("http://127.0.0.1:{}/reset".format(port))


