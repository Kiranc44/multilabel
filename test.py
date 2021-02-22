import cv2
import numpy as np

from preprocess_inputs import object_detection

# Image locations
TEST_IMAGE = cv2.imread("images/1.jpg")


# Test names
test_names = ["Object Detection"]

global solution_funcs

def test_object():
    comparison = test(object_detection, test_names[0], TEST_IMAGE)
    return comparison


def test(test_func, test_name, test_image):
    # Try the student's code first
    try:
        student_processed = test_func(test_image)
    except:
        print_exception(test_name)
        return 

    solution = solution_funcs[test_name](test_image)
    comparison = np.array_equal(student_processed, solution)
    print_test_result(test_name, comparison)

    return comparison

def print_exception(test_name):
    print("Failed to run test on {}.".format(test_name))
    print("The code should be valid Python and return the preprocessed image.")


def print_test_result(test_name, result):
    if result:
        print("Passed {} test.".format(test_name))
    else:
        print("Failed {} test, did not obtain expected preprocessed image."
            .format(test_name))


def feedback(tests_passed):
    print("You passed {} of 3 tests.".format(int(tests_passed)))
    if tests_passed == 3:
        print("Congratulations!")
    else:
        print("See above for additional feedback.")

def set_solution_functions():
    global solution_funcs
    solution_funcs = {test_names[0]: object_solution}


def preprocessing(input_image, height, width):
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)
    return image 

def object_solution(input_image):
    return preprocessing(input_image, 1024, 2048)

def main():
    set_solution_functions()
    counter = test_object()
    feedback(counter)

if __name__ == "__main__":
    main()

