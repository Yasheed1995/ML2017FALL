import os
import glob
import time
import argparse
import numpy as np
from skimage import io

def load_data(path):
    jpgs = glob.glob(path+'/*.jpg')
    imgs = np.array([io.imread(i).flatten() for i in jpgs])
    print ('imgs shape : {}'.format(imgs.shape))
    return imgs

def main(args):
    s = time.time()
    imgs = load_data(args.data_path)
    mean = np.mean(imgs, axis=0)
    
    if 1==1:
        data = imgs - mean
        e_faces, sigma, v = np.linalg.svd(data.transpose(), full_matrices=False)
        weights = np.dot(data, e_faces)

        index = int(args.reconstruct_jpg.split('.')[0])
        recon = mean + np.dot(weights[index, 0:4], e_faces[:, 0:4].transpose())
        recon -= np.min(recon)
        recon /= np.max(recon)
        recon = (recon * 255).astype(np.uint8)
        io.imsave('reconstruction.jpg', recon.reshape(600, 600, 3))
        e = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCA of colored faces')

    parser.add_argument('--data_path',       type=str, default='Aberdeen')
    parser.add_argument('--reconstruct-jpg', type=str)

    args = parser.parse_args()

    main(args)
