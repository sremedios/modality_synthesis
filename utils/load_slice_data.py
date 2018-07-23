'''
Samuel Remedios
NIH CC CNRM
Load images from file
'''

import os
from tqdm import *
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from sklearn.utils import shuffle

def load_image(filename):
    '''
    Loads a single-channel image and adds a dimension for the implicit "1" dimension
    '''
    img = nib.load(filename).get_data()
    img = np.reshape(img, (1,)+img.shape+(1,))
    #MAX_VAL = 255  # consistent maximum intensity in preprocessing

    # linear scaling so all intensities are in [0,1]
    #return np.divide(img, MAX_VAL)
    return  img

def load_slices(filename):
    '''
    Loads a single-channel image and adds a dimension for the implicit "1" dimension
    Returns a np.array of slices: [slice_idx, height, width, channels==1]
    '''
    img = nib.load(filename).get_data()
    IMG_MAX = np.max(img)
    img_slices = np.zeros((img.shape[-1], img.shape[0], img.shape[1], 1), dtype=np.float16)
    for idx in range(len(img_slices)):
        img_slices[idx, :, :, 0] = img[:,:,idx] / IMG_MAX

    return img_slices

def load_middle_slice(filename):
    img = nib.load(filename).get_data()
    img_slice = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=np.float16)
    middle_idx = img.shape[2]//2
    img_slice[0,:,:,0] = img[:,:,middle_idx]
    return img_slice

def load_multiclass_slice_data(data_dir, middle_only=False):
    '''
    Loads all 2D image slices from 3D images and returns them.
    Loads specifically from both T1 and T2 folders and returns them
    '''

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    t1_data_dir = os.path.join(data_dir, "T1")
    t2_data_dir = os.path.join(data_dir, "T2")

    tmp_file = os.path.join(t1_data_dir, os.listdir(t1_data_dir)[0])
    img_shape = nib.load(tmp_file).get_data().shape

    if middle_only:
        total_num_slices = len(os.listdir(t1_data_dir))
    else:
        total_num_slices = len(os.listdir(t1_data_dir)) * img_shape[-1]

    t1_data = np.zeros(shape=((total_num_slices,) + img_shape[:-1] + (1,)), 
                            dtype=np.float16)
    t1_slice_filenames = [None] * total_num_slices

    t2_data = np.zeros(shape=((total_num_slices,) + img_shape[:-1] + (1,)), 
                            dtype=np.float16)
    t2_slice_filenames = [None] * total_num_slices

    combined_data_dict = {t1_data_dir: t1_data,
                          t2_data_dir: t2_data}
    combined_filenames_dict = {t1_data_dir: t1_slice_filenames,
                               t2_data_dir: t2_slice_filenames}

    indices = np.arange(len(t1_data))
    indices = shuffle(indices, random_state=0)

    for cur_dir in (t1_data_dir, t2_data_dir):
        # set cur_idx back to start for both sets of T1 and T2
        cur_idx = 0
        filenames = [os.path.join(cur_dir, x) for x in os.listdir(cur_dir)
                     if not os.path.isdir(os.path.join(cur_dir, x))]
        filenames.sort()

        for f in tqdm(filenames):
            if middle_only:
                img_slice = load_middle_slice(f)
                combined_data_dict[cur_dir][indices[cur_idx]] = img_slice
                combined_filenames_dict[cur_dir][indices[cur_idx]] = f

            else:
                img_slices = load_slices(f)

                for img_slice in img_slices:
                    combined_data_dict[cur_dir][indices[cur_idx]] = img_slice
                    combined_filenames_dict[cur_dir][indices[cur_idx]] = f
            cur_idx += 1

        print(cur_dir,"shape:",combined_data_dict[cur_dir].shape)

    return (combined_data_dict[t1_data_dir], 
           combined_data_dict[t2_data_dir],
           combined_filenames_dict[t1_data_dir], 
           combined_filenames_dict[t2_data_dir], 
           combined_data_dict[t1_data_dir][0].shape)


def load_slice_data(data_dir, middle_only=False):
    '''
    Loads all 2D image slices from 3D images and returns them.

    Naively appends lists, then converts to ndarrays and shuffles
    '''

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir)
                 if not os.path.isdir(os.path.join(data_dir, x))]
    filenames.sort()

    data = []
    slice_filenames = []

    for f in tqdm(filenames):
        if middle_only:
            img_slice = load_middle_slice(f)
            data.append(img_slice)
            slice_filenames.append(f)

        else:
            img_slices = load_slices(f)

            for img_slice in img_slices:
                if not np.any(np.isnan(img_slice)) and np.sum(img_slice) != 0:
                    data.append(img_slice)
                    slice_filenames.append(f)

    data = np.array(data, dtype=np.float16)
    data, slice_filenames = shuffle(data, slice_filenames)

    return (data, 
           slice_filenames,
           data[0].shape)



def load_data(data_dir, classes=None):
    '''
    Loads in datasets and returns the labeled preprocessed patches for use in the model.

    Determines the number of classes for the problem and assigns labels to each class,
    sorted alphabetically.

    Params:
        - data_dir: string, path to all training class directories
        - task: string, one of modality, T1-contrast, FL-contrast'
        - labels_known: boolean, True if we know the labels, such as for training or
                                 validation.  False if we do not know the labels, such
                                 as loading in data to classify in production
    Returns:
        - data: list of 3D ndarrays, the patches of images to use for training
        - labels: list of 1D ndarrays, one-hot encoding corresponding to classes
        - all_filenames: list of strings, corresponding filenames for use in validation/test
        - num_classes: integer, number of classes
        - img_shape: ndarray, shape of an individual image
    '''

    labels = []

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    if classes is None:
        all_filenames = []
        data = []
        filenames = [x for x in os.listdir(data_dir)
                     if not os.path.isdir(os.path.join(data_dir, x))]
        filenames.sort()

        for f in tqdm(filenames):
            img = nib.load(os.path.join(data_dir, f)).get_data()
            img = np.reshape(img, img.shape+(1,))
            data.append(img)
            all_filenames.append(f)

        data = np.array(data)

        return data, all_filenames

    #################### TRAINING OR VALIDATION ####################

    # determine number of classes
    class_directories = [os.path.join(data_dir, x)
                         for x in os.listdir(data_dir)]
    class_directories.sort()

    print(classes)
    num_classes = len(classes)

    # set up all_filenames and class_labels to speed up shuffling
    all_filenames = []
    class_labels = {}
    i = 0
    for class_directory in class_directories:

        if not os.path.basename(class_directory) in classes:
            print("{} not in {}; omitting.".format(
                os.path.basename(class_directory),
                classes))
            continue

        class_labels[os.path.basename(class_directory)] = i
        i += 1
        for filename in os.listdir(class_directory):
            filepath = os.path.join(class_directory, filename)
            all_filenames.append(filepath)

    img_shape = nib.load(all_filenames[0]).get_data().shape
    data = np.empty(shape=((len(all_filenames),) +
                           img_shape + (1,)), dtype=np.float_16)

    # shuffle data
    all_filenames = shuffle(all_filenames, random_state=0)

    data_idx = 0  # pointer to index in data

    for f in tqdm(all_filenames):
        img = nib.load(f).get_data()
        img = np.asarray(img, dtype=np.float_16)

        # place this image in its spot in the data array
        data[data_idx] = np.reshape(img, (1,)+img.shape+(1,))
        data_idx += 1

        cur_label = f.split(os.sep)[-2]
        labels.append(to_categorical(
            class_labels[cur_label], num_classes=num_classes))

    labels = np.array(labels, dtype=np.float_16)
    print(data.shape)
    print(labels.shape)
    return data, labels, all_filenames, num_classes, data[0].shape


