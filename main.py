import glob
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


def normalize(path):
    images = []
    names = []
    for file in tqdm(glob.glob(path + "/*.jpg")):
        image = cv2.resize(cv2.cvtColor(cv2.imread(file, cv2.COLOR_RGBA2BGRA), cv2.COLOR_BGR2RGB), (228, 228))
        image = np.array(image)
        name = file.split('/')[-1].split('_')[0]
        images.append(image)
        names.append(name)
    return images, names


if __name__ == "__main__":
    path = "./dataset/images"
    images, names = normalize(path)

    fig = plt.figure(figsize=(20, 15))

    for i in range(9):
        r = random.randint(1, 186)
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[r])
        plt.xlabel(names[r])

    plt.show()
