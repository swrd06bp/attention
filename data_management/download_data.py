import os
import email
from email.generator import Generator
from email.mime.image import MIMEImage

import cv2
import boto3
import numpy as np
import pandas as pd

import data_getter
from bgs import BackGroundSubtractor


DATA_FOLDER = '.'

def apply_bgs_images(images):
    background_subtractor = BackGroundSubtractor(0.2, 15, images[0])
    new_images = [images[0]]
    for image in images[1:]:
        mask_area = background_subtractor.get_motion_image(image)
        mask_area[mask_area == 255] = 1
        mask_area = np.expand_dims(mask_area, -1)
        mask_area = np.tile(mask_area, [1, 1, 3])
        new_images.append(np.multiply(mask_area, image))
    return new_images

def modify_data(dataset="train"):
    df = pd.read_csv(open(os.path.join(DATA_FOLDER, 'alarms_{}.csv'.format(dataset))), delimiter='--')
    
    for i, key in enumerate(df['s3key']):
        print("Downloading email: {}".format(key))
        s3 = boto3.resource('s3')
        s3.meta.client.download_file('monitoring-station', 'emails/{}'.format(key), '{}'.format(os.path.join(dataset, key)))
        print("Modifying email: {}".format(key))
        msg = email.message_from_file(open(os.path.join(DATA_FOLDER, dataset, key)))
        try:
            images = data_getter.extract_images_from_email_attachment(msg, transform=False)
            images = apply_bgs_images(images)
        except:
            os.system("rm {}".format(os.path.join(dataset, key)))
            continue
        msg.set_payload([])
        for name, image in enumerate(images):
            full_name = "{}.jpg".format(name)
            img_str = cv2.imencode(full_name, image)[1].tostring()
            img = MIMEImage(img_str)
            img.add_header('Content-Disposition', 'attachment', filename=full_name)
            msg.attach(img)
        generator = Generator(open(os.path.join(DATA_FOLDER, dataset, key), "w"))
        generator.flatten(msg)

    

if __name__ == "__main__":
    datasets = ["train"]
    for dataset in datasets:
        modify_data(dataset)
