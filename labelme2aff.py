#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np
import cv2
import pickle

import labelme
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

kernel = np.ones((5, 5), np.uint8)  # 5x5 全 1

def write_voc_xml(xml_path, img_filename, img_size, objects, folder="VOC2007"):
    """
    img_size: (H, W, C)
    objects:  list of dicts:
      {"name": class_name, "bbox": (xmin, ymin, xmax, ymax), "difficult":0, "truncated":0}
    """
    h, w, c = img_size
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = folder
    ET.SubElement(ann, "filename").text = img_filename

    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    ET.SubElement(ann, "segmented").text = "0"

    for obj in objects:
        ob = ET.SubElement(ann, "object")
        ET.SubElement(ob, "name").text = obj["name"]
        ET.SubElement(ob, "pose").text = "Unspecified"
        ET.SubElement(ob, "truncated").text = str(obj.get("truncated", 0))
        ET.SubElement(ob, "difficult").text = str(obj.get("difficult", 0))
        bbox = ET.SubElement(ob, "bndbox")
        xmin, ymin, xmax, ymax = obj["bbox"]
        ET.SubElement(bbox, "xmin").text = str(int(xmin))
        ET.SubElement(bbox, "ymin").text = str(int(ymin))
        ET.SubElement(bbox, "xmax").text = str(int(xmax))
        ET.SubElement(bbox, "ymax").text = str(int(ymax))

    tree = ET.ElementTree(ann)
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    tree.write(xml_path, encoding="utf-8", xml_declaration=False)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default='data', required=False, help="Input annotated directory")
    parser.add_argument("--output_dir", default='data_aff', required=False, help="Output dataset directory")
    parser.add_argument(
        "--labels", help="Labels file or comma separated text", default='labels.txt',required=False
    )

    args = parser.parse_args()

    if osp.exists(args.labels):
        with open(args.labels) as f:
            labels = [label.strip() for label in f if label]
    else:
        labels = [label.strip() for label in args.labels.split(",")]
    print(labels)

    class_names = []
    class_name_to_id = {}

    items = [x for x in labels if x != '__ignore__' and x != '_background_']

    out_imgset_file = osp.join(args.output_dir, "VOCdevkit2012/VOC2012/ImageSets/Main/train.txt")
    os.makedirs(os.path.dirname(out_imgset_file), exist_ok=True)

    imgset = open(out_imgset_file, 'w')

    for filename in sorted(glob.glob(osp.join(args.input_dir, "*.json")))[:]:
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "VOCdevkit2012/VOC2012/JPEGImages", base + ".jpg")
        out_xml_file = osp.join(args.output_dir, "VOCdevkit2012/VOC2012/Annotations", base + ".xml")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        '''分群開始'''
        obj = {}
        for label in label_file.shapes:
            print(label['label'])
            num, _ = label['label'].split("_", 1)
            obj.setdefault(int(num), []).append(label['label'])
        print(obj)

        result = []
        for num in sorted(obj.keys()):
            obj_list = ['_background_'] + obj[num]
            result.append(obj_list)

        c = []
        for group in result:
            class_name_to_id = {}
            for i, label in enumerate(group):
                if '_background_' == label:
                    class_name_to_id[label] = 0
                elif "cut" in label:
                    class_name_to_id[label] = 2
                elif "grasp" in label:
                    class_name_to_id[label] = 5
                elif "pound" in label:
                    class_name_to_id[label] = 7

            c.append(class_name_to_id)
        '''分群結束'''

        n=0
        objects = []
        for i, c_ in enumerate(c):
            aff_num = len(c_)-1
            cls, ins = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=label_file.shapes[n:n+aff_num],
                label_name_to_value=c_,
            )
            n+=aff_num
            
            cls = cls.astype('uint8')
            cls = cv2.erode(cls, kernel, iterations=1)
            cls = cv2.dilate(cls, kernel, iterations=1)
            
            mask_n = "{}_{}_segmask.sm".format(base, i+1)
            out_sm_file = osp.join(args.output_dir, "cache/GTsegmask_VOC_2012_train", mask_n)

            os.makedirs(os.path.dirname(out_sm_file), exist_ok=True)
            with open(out_sm_file, "wb") as f:
                pickle.dump(cls, f, protocol=4)
            print(out_sm_file)
            
            obj_name = list(c_.keys())[1].split("_")[1]
            coords = np.column_stack(np.where(cls > 0))

            # y_min, x_min, y_max, x_max
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # cv2.rectangle(cls, (x_min, y_min), (x_max, y_max), 10, 1)
            # print(x_min, y_min, x_max, y_max)
            # print(obj_name)

            objects.append({"name": obj_name, "bbox": (x_min, y_min, x_max, y_max)})
            # plt.imshow(cls)
            # plt.show()

        filename = filename.split("/")[1]
        write_voc_xml(out_xml_file,base+".jpg", img.shape, objects)

        imgset.write(base+"\n")

    imgset.close()

if __name__ == "__main__":
    main()
