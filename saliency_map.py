import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import utils
import glob
import traceback
import ntpath


## saliency transformation functions
# gradients is an np.ndarray
# all transformations return their 200 x 200 array and the name of the function 
# and a variable specifying if it can be overlaid or not

#to do, change these into an abstract transform class and child classes

def binary_map(gradients):
    func_name = "binary_map"
    temp = np.abs(gradients)
    temp[np.where(temp > 0)] = 255
    return temp, func_name

def intensity_map(gradients):
    func_name = "intensity_map"
    temp = np.abs(gradients)
    high = np.amax(temp)
    temp = (temp / high) * 255
    return temp, func_name

def squared_intensity_map(gradients):
    func_name = "squared_intensity_map"
    temp = np.abs(gradients)
    temp = temp * temp
    high = np.amax(temp)
    temp = (temp / high) * 255
    return temp, func_name

def binary_map_grad_median(gradients):
    func_name = "binary_map_grad_median"
    temp = np.abs(gradients)
    median = np.median(temp[temp>0])
    temp[np.where(temp > median)] = 255
    return temp, func_name

# def binary_map_grad_upper_quartile(gradients):
#     func_name = "binary_map_grad_upper_quartile"
#     temp = np.abs(gradients)
#     median = np.median(temp[temp>0])
#     temp[np.where(temp > median)] = 255
#     return temp, func_name

def heatmap(gradients):
  temp = intensity_map(gradients=gradients)[0].astype('uint8')
  return cv2.applyColorMap(temp, cv2.COLORMAP_JET), "heatmap"


#trans_funcs is a list of transformation functions
#faces are the face images used as input into model
def compute_saliency_maps(tran_funcs, gradient_maps, face_imgs=None):
    saliency_maps = {}
    for face in gradient_maps:
        trans_imgs = {}
        for func in tran_funcs:
            pred_array = []
            for age_range in gradient_maps[face]:
                sal, func_name = func(gradient_maps[face][age_range])
                pred_array.append((age_range, sal))
                #overlay faces if provided
                if face_imgs and func_name is not "heatmap":
                    temp = sal * (face_imgs[face] / 255)
                    temp *= (255 / np.amax(temp))
                    pred_array.append(("overlaid_{}".format(age_range), temp))
            trans_imgs[func_name] = pred_array
        saliency_maps[face] = trans_imgs
    return saliency_maps


#img_dir is output directory for specific image input
#sms is a dict of dicts of np arrays. First index is the face, next is the age_range, 
# def save_saliency_maps(img_dir, sms, preds overlay=False):

def save_image_info(img, saliency_maps, face_imgs, preds, img_dir):
    #check img_dir 
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    #save predicted image
    cv2.imwrite("{}/predicted_image.jpg".format(img_dir), img)

    for face in saliency_maps.keys():
        #create face directory 
        face_dir = "{}/{}".format(img_dir, face)
        if not os.path.exists(face_dir):
            os.mkdir(face_dir)

        #save greyscale face image
        cv2.imwrite("{}/greyscale.jpg".format(face_dir), face_imgs[face])

        #save saliency maps
        for trans in saliency_maps[face].keys():
            trans_dir = "{}/{}".format(face_dir, trans)
            if not os.path.exists(trans_dir):
                os.mkdir(trans_dir)
            for name, ar in saliency_maps[face][trans]:
                if preds[face] in name:
                    cv2.imwrite("{}/predicted_{}.jpg".format(trans_dir, name), ar)
                else:
                    cv2.imwrite("{}/{}.jpg".format(trans_dir, name), ar)


def main(revised=True):
    #set up models
    if revised:
        age_ranges = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-116']
    else:
        age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
    model, face_cascade = utils.get_models(revised=True)
    print("\n")

    #get input files
    inputs_dir = "./inputs"
    input_file_paths = glob.glob("{}/*".format(inputs_dir))

    if len(input_file_paths) == 0:
        sys.exit("No files in ./inputs directory")

    #set up output directory
    output_dir = "./outputs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #initialize trans_funcs
    tran_funcs = [binary_map, intensity_map, squared_intensity_map, binary_map_grad_median, heatmap]

    for file_path in input_file_paths:
        img_dir = "{}/{}".format(output_dir, ntpath.basename(file_path))

        try:
            img, gradient_maps, face_imgs, preds = utils.get_prediction_and_gradients(img_path=file_path, 
                age_ranges=age_ranges, 
                model=model, 
                face_cascade=face_cascade)

            if len(face_imgs.keys()) == 0:
                raise("No faces detected")

            #compute saliency maps
            saliency_maps = compute_saliency_maps(tran_funcs=tran_funcs, gradient_maps=gradient_maps, face_imgs=face_imgs)

            save_image_info(img=img, saliency_maps=saliency_maps,face_imgs=face_imgs, preds=preds, img_dir=img_dir)

        except Exception as e:
            print("Image at {} had issues! Didn't save saliency maps. Exceptions below:".format(file_path))
            traceback.print_tb(e.__traceback__)
            print(repr(e))

if __name__ == '__main__':
    main(revised=True)

