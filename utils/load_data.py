import csv
import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.utils import shuffle
from tqdm import tqdm
from PIL import Image

def get_class_filenames(csv_path, body_part):
    df = pd.read_csv(csv_path, names=["filepath", "class"])
    df = df[df["filepath"].str.contains(body_part) == True]

    pos_shoulder_df = df[df["class"] == 1]
    neg_shoulder_df = df[df["class"] == 0]

    pos_filepaths = pos_shoulder_df["filepath"]
    neg_filepaths = neg_shoulder_df["filepath"]

    pos_filenames = []
    neg_filenames = []

    for fp in pos_filepaths:
        files = os.listdir(fp)
        for f in files:
            pos_filenames.append(os.path.join(fp, f))

    for fp in neg_filepaths:
        files = os.listdir(fp)
        for f in files:
            neg_filenames.append(os.path.join(fp, f))

    return pos_filenames, neg_filenames

def get_filenames(f_path):
    with open(f_path, 'r') as f:
        filenames = f.readlines()
    filenames = [x.strip() for x in filenames]

    return filenames

def get_labels(f_path):
    filenames, labels = [], []
    with open(f_path, 'r') as csvfile:
        label_reader = csv.reader(csvfile)
        for row in label_reader:
            filenames.append(row[0])
            labels.append(row[1])

    return filenames, labels 

def pad_image(img_data, target_dims):
    left_pad = round(float(target_dims[0] - img_data.shape[0]) / 2 )
    right_pad = round(float(target_dims[0] - img_data.shape[0]) - left_pad)
    top_pad = round(float(target_dims[1] - img_data.shape[1]) / 2 )
    bottom_pad = round(float(target_dims[1] - img_data.shape[1]) - top_pad)

    num_channels = 1#img_data.shape[-1]
    new_img = np.zeros((*target_dims, num_channels),
                        dtype=np.float16)

    if len(img_data.shape) == 3:
        front_pad = round(float(target_dims[2] - img_data.shape[2]) / 2 )
        back_pad = round(float(target_dims[2] - img_data.shape[2]) - front_pad)

        pads = ((left_pad, right_pad),
                (top_pad, bottom_pad),
                (front_pad, back_pad))


        for c in range(num_channels):
            new_img[:,:,:,c] = np.pad(img_data[:,:,:,c], pads, 'constant', constant_values=0)

    elif len(img_data.shape) == 2:
        pads = ((left_pad, right_pad),
                (top_pad, bottom_pad),)


        for c in range(num_channels):
            new_img[:,:,c] = np.pad(img_data[:,:,c], pads, 'constant', constant_values=0)

    return new_img 


def load_image(img_path):
    img = nib.load(img_path).get_data()
    img = np.divide(img, np.max(img))
    img = np.reshape(img, img.shape + (1,))
    
    return img 

def load_images(studies):
    target_dims = (96,96,128) 
    num_studies = len(studies)
    num_channels = 1

    X = np.empty((num_studies, *target_dims, num_channels),
                  dtype=np.float16)
    filenames = [None] * num_studies

    indices = np.arange(num_studies)
    indices = shuffle(indices, random_state=0)

    for i in tqdm(range(len(indices))):
        X[indices[i]] = pad_image(load_image(studies[i]), target_dims) 
        filenames[indices[i]] = studies[i]

    return X, filenames 
