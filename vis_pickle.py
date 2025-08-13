import pickle
import cv2
import matplotlib.pyplot as plt

aff_num = 3
for n in range(1, aff_num+1):
    with open("data_aff/cache/GTsegmask_VOC_2012_train/img0001_{}_segmask.sm".format(n), 'rb') as f:
        img = pickle.load(f)
        plt.imshow(img)
        plt.show()
        # plt.savefig("assert/img0001_{}_segmask.png".format(n))
        # cv2.imwrite("assert/img0001_{}_segmask.png".format(n), img*30)