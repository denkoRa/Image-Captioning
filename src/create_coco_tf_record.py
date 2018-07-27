r"""Convert raw Microsoft COCO dataset to TFRecord for object_detection.
Attention Please!!!

1)For easy use of this script, Your coco dataset directory struture should like this :
    +Your coco dataset root
        +train2017
        +val2017
        +annotations
            -instances_train2017.json
            -instances_val2017.json
2)To use this script, you should download python coco tools from "http://mscoco.org/dataset/#download" and make it.
After make, copy the pycocotools directory to the directory of this "create_coco_tf_record.py"
or add the pycocotools path to  PYTHONPATH of ~/.bashrc file.

Example usage:
    python create_coco_tf_record.py --data_dir=/path/to/your/coco/root/directory \
        --set=train \
        --output_path=/where/you/want/to/save/pascal.record
        --shuffle_imgs=True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging
import json
import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'D:\Image Captioning\data', 'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('set', 'val', 'Convert training set or validation set')
flags.DEFINE_string('output_filepath', '', 'Path to output TFRecord')
flags.DEFINE_bool('shuffle_imgs',True,'whether to shuffle images of coco')
FLAGS = flags.FLAGS


def load_coco_dection_dataset(imgs_dir, annotations_filepath, shuffle_img = True ):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
        shuffle_img: wheter to shuffle images order
    Return:
        coco_data: list of dictionary format information of each image
    """
    
    coco = COCO(annotations_filepath)

    img_ids = coco.getImgIds() # totally 82783 images
    #cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.

    if shuffle_img:
        shuffle(img_ids)

    coco_data = []

    nb_imgs = len(img_ids)
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Reading images: %d / %d "%(index, nb_imgs))
        img_info = {}

        img_detail = coco.loadImgs(img_id)[0]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        caps = []

        for ann in anns:
            caps.append(ann['caption'])
        
        img_info['id'] = anns[0]['image_id']
        img_info['caption'] = caps
        
        coco_data.append(img_info)
    return coco_data


def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    # bboxes = img_data['bboxes']
    # xmin, xmax, ymin, ymax = [], [], [], []
    # for bbox in bboxes:
    #     xmin.append(bbox[0])
    #     xmax.append(bbox[0] + bbox[2])
    #     ymin.append(bbox[1])
    #     ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/id': dataset_util.int64_feature(img_data['id']),
        'image/caption': dataset_util.bytes_list_feature(img_data['caption']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data'])
    }))
    return example

def main(_):
    if FLAGS.set == "train":
        imgs_dir = os.path.join(FLAGS.data_dir, 'train2014')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','captions_train2014.json')
        print("Convert coco train file to tf record")
    elif FLAGS.set == "val":
        imgs_dir = os.path.join(FLAGS.data_dir, 'train2014')
        #annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','captions_val2014.json')
        print(os.getcwd())
        annotations_filepath = '/mnt/d/Image Captioning/data/annotations/captions_val2014.json'
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")
    # load total coco data
    coco_data = load_coco_dection_dataset(imgs_dir,annotations_filepath,shuffle_img=FLAGS.shuffle_imgs)
    total_imgs = len(coco_data)
    # write coco data to tf record
    with open('captions.json', 'w') as fp:
        json.dump(coco_data, fp)
    # with tf.python_io.TFRecordWriter(FLAGS.output_filepath) as tfrecord_writer:
    #     for index, img_data in enumerate(coco_data):
    #         if index % 100 == 0:
    #             print("Converting images: %d / %d" % (index, total_imgs))
    #         example = dict_to_coco_example(img_data)
    #         tfrecord_writer.write(example.SerializeToString())


if __name__ == "__main__":
    tf.app.run()
