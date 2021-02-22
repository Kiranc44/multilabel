import argparse
import cv2
import numpy as np

from handle_models import handle_output, preprocessing
from inference import Network


CLASSES_REC=['Road','Sidewalk','Building','Wall','Fence','Pole','Traffic Light','Traffic Sign','Vegetation','Terrain','Sky','Person','Rider','Car','Truck',"Bus",'Train','Motor Cycle','Bicycle','Ego Vehicle']


def get_args():

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"
    t_desc = "The type of model: POSE, TEXT or CAR_META"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-t", help=t_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args

def perform_inference(args):
    inference_network = Network()
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)
    image = cv2.imread(args.i)

    preprocessed_image = preprocessing(image, h, w)

    inference_network.sync_inference(preprocessed_image)

    output = inference_network.extract_output()

    process_func=handle_output(args.t)
    processed_output = process_func(output,image.shape)

    for i in range(len(processed_output)):
        print(CLASSES_REC[processed_output[i]])



def main():
    args = get_args()
    perform_inference(args)


if __name__ == "__main__":
    main()


