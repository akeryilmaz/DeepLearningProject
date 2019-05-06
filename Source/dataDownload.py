import urllib.request as urllib
import progressbar
from functools import partial
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True, help="Folder to save the downloads")
args = ap.parse_args()


def show_progress(p_bar, *arg):
    try:
        p_bar.update(arg[0] * arg[1])
    except:
        p_bar.update(p_bar.maxval)



for x in range(30):
    name = "images_{:03d}.tar".format(x)

    url = "https://s3.amazonaws.com/google-landmark/train/{}".format(name)
    u = urllib.urlopen(url)
    size = int(u.info()['Content-Length'])

    widgets = ["Downloading {}: ".format(name), progressbar.Counter(), "/{}".format(size), progressbar.Percentage(),
               " ",
               progressbar.Bar(),
               " ",
               progressbar.ETA(), " ", progressbar.FileTransferSpeed()]
    pbar = progressbar.ProgressBar(maxval=size, widgets=widgets).start()
    show_progress2 = partial(show_progress, pbar)
    path = os.path.sep.join([args.folder, name])
    urllib.urlretrieve("https://s3.amazonaws.com/google-landmark/train/{}".format(name), path, show_progress2)
