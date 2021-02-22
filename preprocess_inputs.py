import cv2
import numpy as np


def object_detection(input_image):
    preprocessed_image = np.copy(input_image)
    preprocessed_image=cv2.resize(preprocessed_image,(2048,1024))
    preprocessed_image=preprocessed_image.transpose((2,0,1))
    preprocessed_image=preprocessed_image.reshape(1,3,1024,2048)
    
    return preprocessed_image


