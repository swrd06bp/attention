import os
import json
import email
import mimetypes

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

DATA_FOLDER = 'data_management'
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

def get_filetype_filename(part):
    filename = part.get_filename()
    if filename:
        # guess file type using the name
        type_subtype, encoding = mimetypes.guess_type(filename)
        # split the type and subtype
        file_type, file_subtype = type_subtype.split('/')
    else:
        file_type = None
    return file_type, filename

def extract_images_from_email_attachment(msg, transform=True):
    all_attachments = []
    for part in msg.walk():
        # get file type for the email part
        file_type, filename = get_filetype_filename(part)
        # add to attachments only if it's an image
        if file_type == 'image':
            attachment = part.get_payload(decode=True)
            arr = np.frombuffer(attachment, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if transform:
                # Convert colours to RGB because TF works on RGB files
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            # concatenate all the images of the different emails
            all_attachments.append(img)

    return all_attachments

def get_mask(masks_list):
    # by default the mask lets everything pass
    mask = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

    # iterate over all the masks for an alarm
    for mask_idx in range(len(masks_list)):
        # polygon for a mask as a list of (x, y) tuples
        polygon = [(IMAGE_WIDTH*x, IMAGE_HEIGHT*y)
                   for x, y in masks_list[mask_idx]]

        # initial mask is drawn with ones representing the polygon and
        # zeros for the background
        img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

        # invert and multiply elementwise with the existing mask
        polygon_mask = 1-np.array(img)
        mask = np.multiply(mask, polygon_mask)
    return mask

def apply_masks(images, masks):
    mask = get_mask(json.loads(masks)) 
    return np.array([np.multiply(mask, image) for image in images])


def get_number_examples(dataset="train"):
    df = pd.read_csv(open(os.path.join(DATA_FOLDER, 'alarms_{}.csv'.format(dataset))), delimiter='--')
    return df.shape[0]



def get_data(dataset="train", batch_size=1):
    df = pd.read_csv(open(os.path.join(DATA_FOLDER, 'alarms_{}.csv'.format(dataset))), delimiter='--')
    nb_data = 1
    all_images = []
    all_labels = []
    while True:
        df = df.sample(frac=1).reset_index(drop=True)
        for i, key in enumerate(df['s3key']):
            try:
                msg = email.message_from_file(open(os.path.join(DATA_FOLDER, dataset, key)))
                images = extract_images_from_email_attachment(msg)
                images = apply_masks(images, df['masks'][i])
                images = np.expand_dims(images, -1)
                images_temp = np.concatenate((images[0, :, :], images[1, :, :]), axis=2)
                images = np.concatenate((images_temp, images[2, :, :]), axis=2)
                images = np.expand_dims(images, 0)
                images = images.astype(float)
                images /= 255
            except:
                continue
            
            labels = np.array([int(df['ground_truth'][i])])
            all_images.append(images[0])
            all_labels.append(labels[0])
            if nb_data % batch_size == 0:
                all_images_temp = all_images
                all_labels_temp = all_labels
                all_images = []
                all_labels = []
                yield np.array(all_images_temp), np.array(all_labels_temp)
                
            nb_data += 1
