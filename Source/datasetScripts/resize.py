import os
import argparse
import time
import hashlib
import tarfile
import urllib.request
from functools import partial
from multiprocessing import Pool

import cv2
from tqdm import tqdm


def process_image(file, source_dir, target_dir):
    source_path = os.path.join(source_dir, file)
    image = cv2.imread(source_path)
    image = cv2.resize(image, (224, 224))
    target_path = source_path.replace(source_dir, target_dir)
    if not os.path.exists(os.path.dirname(target_path)):
        try:
            os.makedirs(os.path.dirname(target_path))
        except:
            pass
    cv2.imwrite(target_path, image)
    os.remove(source_path)

root_dir = "../Data"
target_dir = "../Resized"

for dirName, subdirList, fileList in os.walk(root_dir):
    #process_func = partial(process_image, root_dir, target_dir)
    #dirName = dirName.replace(root_dir, "")
    #for file in fileList:
        #process_func(dirName + file)
 
