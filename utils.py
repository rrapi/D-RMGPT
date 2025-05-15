import cv2
import base64
import math
import numpy as np

def cm_to_m(x, dec = 4):
    return round(x/100,dec)

def array_cm_to_m(array):
    n = np.ndim(array)
    if n == 1:
        res = [cm_to_m(x) for x in array]
    elif n == 2:
        res = [[cm_to_m(y) for y in x] for x in array]
    else:
        print("Conversion not implemented.")
    return res

def array_deg_to_rad(array):
    return [math.radians(x) for x in array]

def array_rad_to_deg(array):
    return [math.degrees(x) for x in array]

def object_to_pose(i, object_positions, tool_orientation):
    assert i > 0 and i < len(object_positions)
    return cm_to_m(object_positions[i-1]) + array_deg_to_rad(tool_orientation)

###############

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def write_text_file(file_path, stringa):
    with open(file_path, 'w') as file:
        file.write(stringa)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    # hsv[:,:,2] = cv2.add(hsv[:,:,2], value)
    # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def show_image(img):
    # showing the image
    cv2.imshow('RealSense', img)
    # waiting using waitKey method
    cv2.waitKey(0)


##################

def print_sequence(arr, sep):
    print(*arr, sep=sep)


def equal_set(full_set, test_set):
    if all(e in test_set for e in full_set):
        return True
    return False