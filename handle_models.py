import cv2
import numpy as np
from PIL import Image 

def handle_multi(output, input_shape):
    """
    python3 app.py -i "images/sitting-on-car.jpg" -t "MULTI" -m "/home/rakshithkumarj/Courses/OpenVino/models/semantic-segmentation-adas-0001/FP16/semantic-segmentation-adas-0001.xml" 
    
    """

    return (np.unique(output['4119.1']))




def handle_output(model_type):
    if model_type=="MULTI":
        return handle_multi
    else:
        return None


def preprocessing(input_image, height, width):

    image = np.copy(input_image)
    image = cv2.resize(image, (width, height)).astype('float32') 
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

