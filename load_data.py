import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm


def load_data(dir, resize, sigmaX=10):
    ImageSet = []
    load = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(dir)):
        PATH = os.path.join(dir,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = load(PATH)
           
            img = cv2.resize(img, (resize,resize))
           
            ImageSet.append(np.array(img))
    return ImageSet