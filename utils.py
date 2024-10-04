import os
import cv2
import numpy as np

from PIL import ImageDraw, ImageFont

def display_ocv_image_in_notebook(image):
    from IPython.display import display, Image
    _, frame = cv2.imencode(".jpeg", image)
    display(Image(data=frame.tobytes()))



def extract_patch(filepath, bbox, size=None):
    """
    Extract a patch from an image given a bounding box. If size is not None, the patch will be extracted around the center of the bounding box with the specified size. 
    Otherwise, the patch will be extracted with the same width and height as the bounding box.
    filepath: str, path to the image
    bbox: tuple, (x,y,width,height)
    size: (optional) int, size of the square patch to be extracted
    
    """
    x,y,width,height = bbox
    width = int(width)
    height = int(height)
    image = cv2.imread(filepath)
    if size is None:
        d = max(width, height)
        x_min = x
        x_max = x+d
        y_min = y
        y_max = y+d
    else:
        center = (x+width//2, y+height//2)
        x_min = max(center[0] - size//2, 0)
        x_max = center[0] + size//2
        y_min = max(center[1] - size//2,0)
        y_max = center[1] + size//2
    patch = image[y_min:y_max, x_min:x_max]
    return patch


def crop_around_center(image, size):
    """
    Crop a square patch around the center of the image with the specified size.
    image: numpy array, the image to crop
    size: int, size of the patch to be extracted
    
    """
    h,w = image.shape[:2]
    center = (w//2, h//2)
    x_min = max(center[0] - size//2,0)
    x_max = center[0] + size//2
    y_min = max(center[1] - size//2,0)
    y_max = center[1] + size//2
    return image[y_min:y_max, x_min:x_max]



def extend_patch(patch, new_shape):
    """
    Extend the patch to the specified shape by adding borders.
    patch: numpy array, the patch to be extended
    new_shape: tuple, (height, width) of the new patch
    """
    
    h,w,_ = patch.shape
    new_h, new_w = new_shape
    x_extend = new_w - w
    y_extend = new_h - h
    top = y_extend//2
    bottom = y_extend - top
    left = x_extend//2
    right = x_extend - left

    borderType = cv2.BORDER_REPLICATE
    new_patch = cv2.copyMakeBorder(patch, top, bottom, left, right, borderType)
    

    return new_patch
def random_rotation(image, angle_range):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    h,w = image.shape[:2]
    center = (w//2, h//2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w,h), flags=cv2.INTER_LINEAR)
    return rotated

def random_translation(image, translation_range):
    dx = np.random.uniform(translation_range[0], translation_range[1])
    dy = np.random.uniform(translation_range[0], translation_range[1])
    h,w = image.shape[:2]
    M = np.float32([[1,0,dx],[0,1,dy]])
    translated = cv2.warpAffine(image, M, (w,h))
    return translated

def random_brightness(image, brightness_range):
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    brightened = image + brightness
    brightened = np.clip(brightened, 0, 255)
    return brightened

def random_contrast(image, contrast_range):
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    contrasted = image * contrast
    contrasted = np.clip(contrasted, 0, 255)
    return contrasted

def random_blur(image, blur_range):
    blur = np.random.uniform(blur_range[0], blur_range[1])
    blurred = cv2.GaussianBlur(image, (5,5), blur)
    return blurred


def show_bbox(image, labels, bboxes, scores):
    """
    Draw bounding boxes on the image using OpenCV.
    
    image: PIL image (will be converted to OpenCV format)
    labels: list of strings, e.g., ['cat', 'dog', ...]
    bboxes: list of tuples, [(x1, y1, x2, y2), ...]
    scores: list of floats, e.g., [0.95, 0.88, ...]

    Returns: OpenCV image (in NumPy array format) with drawn bounding boxes and labels.
    """
    # Convert the PIL image to a NumPy array (OpenCV format)
    image_cv = np.array(image)

    # Convert RGB to BGR (as OpenCV uses BGR format)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Set font for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 10

    for label, bbox, score in zip(labels, bboxes, scores):
        # Ensure bounding box coordinates are integers
        x1, y1, x2, y2 = map(int, bbox)

        # Draw the bounding box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

        # Create the label with the score
        label_text = f"{label[0]}"

        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Draw the background rectangle for the text
        cv2.rectangle(image_cv, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (255, 0, 0), thickness=cv2.FILLED)
        
        # Put the label text on the image
        cv2.putText(image_cv, label_text, (x1, y1 - baseline), font, font_scale, (255, 255, 255), font_thickness)

    # Convert BGR back to RGB before returning
    image_with_bboxes = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    return image_with_bboxes

        
    