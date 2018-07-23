from utils.load_data import *
from utils.load_slice_data import *
from utils.display import *

from keras.models import load_model, model_from_json
import os
import sys
import json

if __name__ == '__main__':

    ########## DIRECTORY SETUP ##########

    ROOT_DIR = "data"
    TEST_DIR = os.path.join(ROOT_DIR, "test", sys.argv[1])
    RECON_DIR = os.path.join(ROOT_DIR, "reconstructions")

    for d in [ROOT_DIR, TEST_DIR, RECON_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    ########## MODEL SETUP ##########

    weight_path = sys.argv[2]
    model_path = sys.argv[3]

    with open(model_path) as json_data:
        model = model_from_json(json.load(json_data))
    model.load_weights(weight_path)

    ########## LOAD DATA ##########

    #X_T1, X_T2, filenames_T1, filenames_T2, dims = load_slice_data(TEST_DIR)

    # test on a subset
    filenames = [os.path.join(TEST_DIR, x) for x in os.listdir(TEST_DIR)]
    filenames.sort()
    filenames = filenames[0:5]

    # tmp affine to use for all images
    #affine = nib.load(filenames[0]).affine

    ########## TEST ##########

    for filename in tqdm(filenames):
        '''
        img_slices = load_slices(filename)
        pred_slices = model.predict(img_slices, batch_size=1, verbose=0)

        img = np.zeros((img_slices.shape[1], img_slices.shape[2], img_slices.shape[0]))
        input_img = np.zeros((img_slices.shape[1], img_slices.shape[2], img_slices.shape[0]))
        for i in range(len(pred_slices)):
            img[:,:,i] = pred_slices[i,:,:,0]
            input_img[:,:,i] = img_slices[i,:,:,0]
        '''

        img_slice = load_middle_slice(filename)
        pred_slice = model.predict(img_slice, batch_size=1, verbose=0)

        '''
        img = np.zeros((img_slice.shape[1], img_slice.shape[2], 1))
        input_img = np.zeros((img_slice.shape[1], img_slice.shape[2], 1))
        img[:,:,0] = pred_slice[:,:,0]
        input_img[:,:,0] = img_slice[:,:,0]


        result_path = os.path.join(RECON_DIR, "reconstructed_" + 
                os.path.basename(filename))
        input_result_path = os.path.join(RECON_DIR, "input_" + 
                os.path.basename(filename))
        '''

        show_image_diffs(img_slice[0,:,:,0], pred_slice[0,:,:,0])

        compare_histograms(img_slice[0,:,:,0], pred_slice[0,:,:,0], TEST_DIR)

        '''
        nii_obj = nib.Nifti1Image(img, affine=affine)
        nib.save(nii_obj, result_path)

        nii_obj = nib.Nifti1Image(input_img, affine=affine)
        nib.save(nii_obj, input_result_path)
        '''
