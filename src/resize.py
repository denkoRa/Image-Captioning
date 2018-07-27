from PIL import Image
import os

_path = r'C:\Users\PSIML-1.PSIML-1\Desktop\projekti\Image-Captioning\raw_data\train2014'

for filename in os.listdir(_path):
    im = Image.open(os.path.join(_path, filename))
    im = im.resize((299, 299), Image.BILINEAR)
    im.save(os.path.join(_path, filename))
    #print(os.path.join(_path, filename))
    #print(filename)
