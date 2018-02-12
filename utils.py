import numpy as np
cropx_top = 50
cropx_bottom = 140
x_new = cropx_bottom - cropx_top
y_new = 320

def crop_image(image):
    cropped = image[cropx_top:cropx_bottom,0:y_new]
    return cropped

def change_brightness(image):
    image = image + np.random.random_integers(-20,20)
    return image

def normal_image(image):
    image = (image / 255.0) - 0.5
    return image

def process_image(image):
    image = change_brightness(image)
    image = normal_image(image)
    image = crop_image(image)
    return image
