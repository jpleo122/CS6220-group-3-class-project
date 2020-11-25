import wget
import os
from tensorflow.keras.models import load_model
import cv2 
import tensorflow as tf
import numpy as np

#default urls from adapted colab 
#revised cnn from our model training
#got the revised url file id from this link: 
def get_models(revised=False):
    #check if model dir exists
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    #create urls for downloads
    cnn_url = "https://drive.google.com/uc?export=download&id=12MgZBpQ0N7suHnNecVMj_ae9zYqbjAhF"
    cnn_path = "{}/age_detect_cnn_model_default.h5".format(models_dir)
    if revised:
        cnn_url = "https://drive.google.com/uc?export=download&id=10leKQDcC7grQLRaO77i4NEY_uS-3TUXy"
        cnn_path = "{}/age_detect_cnn_model_revised.h5".format(models_dir)

    face_detector_url = "https://drive.google.com/uc?export=download&id=1Gcz4wc8iA1SHfV9REcK4i74Tf9vaETq7"
    face_detector_path = "{}/haarcascade_frontalface_default.xml".format(models_dir)

    if not os.path.exists(cnn_path):
        wget.download(cnn_url, out=cnn_path)

    if not os.path.exists(face_detector_path):
        wget.download(face_detector_url, out=face_detector_path)

    #load models and return them
    model = load_model(cnn_path)
    face_cascade = cv2.CascadeClassifier(face_detector_path)

    return model, face_cascade

# Defining a function to shrink the detected face region by a scale for 
#better prediction in the model.
def shrink_face_roi(x, y, w, h, scale=0.9):
    wh_multiplier = (1-scale)/2
    x_new = int(x + (w * wh_multiplier))
    y_new = int(y + (h * wh_multiplier))
    w_new = int(w * scale)
    h_new = int(h * scale)
    return (x_new, y_new, w_new, h_new)

# Defining a function to create the predicted age overlay on the image by centering the text.
def create_age_text(img, text, pct_text, x, y, w, h):

    # Defining font, scales and thickness.
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 1.2
    yrsold_scale = 0.7
    pct_text_scale = 0.65

    # Getting width, height and baseline of age text and "years old".
    (text_width, text_height), text_bsln = cv2.getTextSize(text, fontFace=fontFace, fontScale=text_scale, thickness=2)
    (yrsold_width, yrsold_height), yrsold_bsln = cv2.getTextSize("years old", fontFace=fontFace, fontScale=yrsold_scale, thickness=1)
    (pct_text_width, pct_text_height), pct_text_bsln = cv2.getTextSize(pct_text, fontFace=fontFace, fontScale=pct_text_scale, thickness=1)

    # Calculating center point coordinates of text background rectangle.
    x_center = x + (w/2)
    y_text_center = y + h + 20
    y_yrsold_center = y + h + 48
    y_pct_text_center = y + h + 75

    # Calculating bottom left corner coordinates of text based on text size and center point of background rectangle calculated above.
    x_text_org = int(round(x_center - (text_width / 2)))
    y_text_org = int(round(y_text_center + (text_height / 2)))
    x_yrsold_org = int(round(x_center - (yrsold_width / 2)))
    y_yrsold_org = int(round(y_yrsold_center + (yrsold_height / 2)))
    x_pct_text_org = int(round(x_center - (pct_text_width / 2)))
    y_pct_text_org = int(round(y_pct_text_center + (pct_text_height / 2)))

    face_age_background = cv2.rectangle(img, (x-1, y+h), (x+w+1, y+h+94), (0, 100, 0), cv2.FILLED)
    face_age_text = cv2.putText(img, text, org=(x_text_org, y_text_org), fontFace=fontFace, fontScale=text_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)
    yrsold_text = cv2.putText(img, "years old", org=(x_yrsold_org, y_yrsold_org), fontFace=fontFace, fontScale=yrsold_scale, thickness=1, color=(255, 255, 255), lineType=cv2.LINE_AA)
    pct_age_text = cv2.putText(img, pct_text, org=(x_pct_text_org, y_pct_text_org), fontFace=fontFace, fontScale=pct_text_scale, thickness=1, color=(255, 255, 255), lineType=cv2.LINE_AA)

    return (face_age_background, face_age_text, yrsold_text)


def get_prediction_and_gradients(img_path, age_ranges, model, face_cascade):
    # Making a copy of the image for overlay of ages and making a grayscale copy for passing to the loaded model for age classification.
    img = cv2.imread(img_path)

    img_copy = np.copy(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gradient_maps = {}
    face_imgs = {}
    predictions = {}

    # Detecting faces in the image using the face_cascade loaded above and storing their coordinates into a list.
    faces = face_cascade.detectMultiScale(img_copy, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

    # Looping through each face found in the image.
    for i, (x, y, w, h) in enumerate(faces):
        #define gradient and face image arrays for particular face
        pred_grad_maps = {}

        # Drawing a rectangle around the found face.
        face_rect = cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 100, 0), thickness=2)
        
        # Predicting the age of the found face using the model loaded above.
        x2, y2, w2, h2 = shrink_face_roi(x, y, w, h)
        face_roi = img_gray[y2:y2+h2, x2:x2+w2]
        face_roi = cv2.resize(face_roi, (200, 200))

        #grab greyscaled face image here
        face_imgs["face_" + str(i)] = face_roi
        face_roi = face_roi.reshape(-1, 200, 200, 1)

        model_input = tf.convert_to_tensor(face_roi, dtype=tf.float32)      

        #watch gradients while predicting output
        #from https://www.tensorflow.org/tutorials/interpretability/integrated_gradients#calculate_integrated_gradients
        #look for def compute_gradients
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(model_input)
            output = model(model_input)
            face_age = age_ranges[np.argmax(output)]
            face_age_pct = f"({round(np.max(model.predict(face_roi))*100, 2)}%)"
            for idx in range(len(age_ranges)):
#                probs = tf.nn.softmax(output, axis=-1)[:, idx]
                #compute gradients for input and reshape gradients array to be 200 x 200
                #can use face_roi shape because gradients are same shape as input image
                gradients = tape.gradient(output[:, idx], model_input).numpy().reshape(face_roi.shape[1:3])
                pred_grad_maps[age_ranges[idx]] = gradients

        #delete reference to tape
        del tape

        gradient_maps["face_" + str(i)] = pred_grad_maps
        predictions["face_" + str(i)] = face_age
        
        # Calling the above defined function to create the predicted age overlay on the image.
        face_age_background, face_age_text, yrsold_text = create_age_text(img_copy, face_age, face_age_pct, x, y, w, h)

    return img_copy, gradient_maps, face_imgs, predictions



