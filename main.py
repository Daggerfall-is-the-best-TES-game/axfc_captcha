import cv2
import numpy as np
from skimage.transform import rotate
import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = #your installation directory here

def pad_image(image):
    """
    Adds 5% 255 padding around the image
    :param image: a cv2 image
    :return: the padded image
    """
    top = int(0.05 * image.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * image.shape[1])  # shape[1] = cols
    right = left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    return image


def denoise_image(image):
    """
    removes the captcha noise from an image, and thresholds it
    :param image: a cv2 image
    :return: the denoised image
    """
    _, image = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY_INV)
    new_size = np.array(image.shape[::-1]) * 4
    image = cv2.resize(image, new_size)
    _, image = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY_INV)

    image = pad_image(image)

    kernel1 = np.ones((5, 5))
    kernel2 = np.ones((3, 3))
    image = cv2.medianBlur(image, 11)
    image = cv2.erode(image, kernel1)
    image = cv2.dilate(image, kernel2)
    return image


def segment_image(image):
    """
    extracts the character segments from an image
    :param image: a cv2 image
    :return: a list of subimages
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours.sort(key=lambda c: cv2.boundingRect(c)[2])
    contours = contours[-4:-1]
    contours.sort(key=lambda c: cv2.boundingRect(c)[0])

    regions = []
    for cnt in contours:
        convex_hull = cv2.convexHull(cnt)
        x, y, w, h = cv2.boundingRect(convex_hull)

        rect = cv2.minAreaRect(convex_hull)
        angle = rect[-1]

        p = np.array(rect[1])

        if p[0] > p[1]:
            angle -= 90

        margin = 5

        max_height, max_width = image.shape[:2]
        x, y = max(x - margin, 0), max(y - margin, 0)
        w, h = min(w + margin, max_width - x), min(h + margin, max_height - y)
        regions.append((image[y:y + h, x:x + w], angle))

    return regions


def deskew_images(image_list):
    """
    unskews each image and adds padding
    :param image_list: a list of tuples where the first item is a cv2 image,
    and the second item is the angle at which it is tilted
    :return: a list of unskewed and padded images
    """
    images = []
    for image, angle in image_list:
        image = rotate(image, angle, resize=True, cval=255, preserve_range=True)
        image = pad_image(image)
        images.append(image)

    return images


def image_to_letter(image):
    """
    Takes an unskewed black and white image of a single character and returns the character
    :param image: a cv2 image
    :return: a string of length 1
    """
    cv2.imwrite("temp.jpg", image)
    pi_image = Image.open("temp.jpg")
    config = r" --psm 10"
    return pytesseract.image_to_string(pi_image, config=config)[0]


def solve_captcha(captcha):
    """
    takes a captcha as an image and returns the text. Probably about 30% accurate
    :param captcha: the image of the captcha, as a cv2 image
    :return:the solution as a 3 character string
    """
    img = denoise_image(captcha)
    regions = segment_image(img)
    images = deskew_images(regions)
    solution = "".join(map(image_to_letter, images))
    return solution

if __name__ == "__main__":
    img = cv2.imread("images/captcha.pl8.png", 0)
    print(solve_captcha(img))
