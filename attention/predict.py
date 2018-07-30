import matplotlib.pyplot as plt
import _pickle as pickle
import tensorflow as tf
from solver import CaptioningSolver
from model import CaptionGenerator
from utils import load_coco_data


data = load_coco_data(data_path='./data', split='val')
with open('./data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

model = CaptionGenerator(word_to_idx)
solver = CaptioningSolver(model, data, data, image_path='./image/val2014_resized',test_model='./model/lstm/model-30')

solver.test(data, split='val')

