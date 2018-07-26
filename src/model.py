import os
import sys
from pycocotools.coco import COCO



data_dir = os.getcwd()
captions_ann_file = os.path.join(data_dir, '../data/annotations/captions_train2014.json')
coco = COCO(captions_ann_file)
ids = list(coco.getAnnIds())

import numpy
import skimage.io as io
import matplotlib.pyplot as plt

# pick a random image and obtain the corresponding URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
coco_caps.showAnns(anns)