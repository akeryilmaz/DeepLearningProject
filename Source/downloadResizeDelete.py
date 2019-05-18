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

images_base_url = 'https://s3.amazonaws.com/google-landmark/train/images_{:03d}.tar'
md5_base_url = 'https://s3.amazonaws.com/google-landmark/md5sum/train/md5.images_{:03d}.txt'


def md5(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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


def process_tar_file(index, target_dir, resized_dir):
    tar_url = images_base_url.format(index)
    md5_url = md5_base_url.format(index)

    tar_path = os.path.join(*[target_dir, 'tars', os.path.basename(tar_url)])
    md5_path = os.path.join(*[target_dir, 'tars', os.path.basename(md5_url)])

    print('Downloading: ' + tar_path)

    start_time = time.time()

    if not os.path.exists(md5_path):
        urllib.request.urlretrieve(md5_url, md5_path)
    if not os.path.exists(tar_path):
        urllib.request.urlretrieve(tar_url, tar_path)

    print('{}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

    # checksum

    ref_checksum = open(md5_path).readlines()[0].split()[0]
    tar_checksum = md5(tar_path)
    if ref_checksum != tar_checksum:
        print('{}: failed checksum'.format(index))
        return

    # open tar file

    extract_dir = os.path.join(target_dir, 'raw_images')
    tar_file = tarfile.open(tar_path)
    tar_file.extractall(extract_dir)
    tar_file_members = [m.name for m in tar_file.getmembers()]
    tar_file.close()

    # delete tar file

    os.remove(tar_path)
    os.remove(md5_path)

    # resize and move images

    process_func = partial(process_image, source_dir=extract_dir, target_dir=resized_dir)
    for file in tqdm(tar_file_members, desc='Files for tar {:03d}'.format(index)):
        process_func(file)


def main(args):
    if not os.path.exists(args.target_dir):
        print('Please create target dir: {}'.format(args.target_dir))
        return
    if not os.path.exists(os.path.join(args.target_dir, 'tars')):
        os.mkdir(os.path.join(args.target_dir, 'tars'))
    if not os.path.exists(os.path.join(args.target_dir, 'raw_images')):
        os.mkdir(os.path.join(args.target_dir, 'raw_images'))
    
    indexes = list(range(28, 30)) + list(range(97, 100))
    func = partial(process_tar_file, target_dir=args.target_dir, resized_dir=args.resized_dir)
    with Pool(args.processes) as p:
        for _ in tqdm(p.imap(func, indexes), total=len(indexes), desc='TAR files'):
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target_dir', default=r'D:\temp\landmarks_recognition\images\train')
    p.add_argument('--resized_dir', default=r'D:\temp\landmarks_recognition\images\train\images_448')
    p.add_argument('--processes', type=int, default=5)
main(p.parse_args())
