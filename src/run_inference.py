# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from PIL import Image

import tensorflow as tf
from matplotlib import pyplot as plt

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
import utils
import cv2
import numpy as np
import json
import nltk

import matplotlib.pyplot as plt
import skimage.transform
from scipy import ndimage

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", r"C:\Users\PSIML-1.PSIML-1\Desktop\projekti\Image-Captioning\src\train_log",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", r"C:\Users\PSIML-1.PSIML-1\Desktop\projekti\Image-Captioning\output_data\word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", os.path.join(utils.repo_path, 'output_data\\test-?????-of-00008'),
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)

save_path = r"C:\Users\PSIML-1.PSIML-1\Desktop\projekti\Image-Captioning\output_data\results"
captions_file = r'src\\captions_val.json'

def preprocess_captions():
  path = os.path.join(os.getcwd(), captions_file)
  with tf.gfile.FastGFile(path, "rb") as f:
    caption_data = json.load(f)

  index = {}
  for sample in caption_data:
    index[sample['id']] = sample['caption']

  return  index

def save_image(fpath, true_captions, predicted_captions):
  img = cv2.imread(fpath, cv2.IMREAD_COLOR)

  #print(img.shape)
  left_right_ext = 200
  up_down_ext = 200
  new_img = np.ones((img.shape[0]+2*up_down_ext, img.shape[1] + 2*left_right_ext, img.shape[2]))*255
  new_img[up_down_ext:up_down_ext+img.shape[0], left_right_ext:left_right_ext+img.shape[1], :] = img.copy()
  new_img /= 255
  
  font = cv2.FONT_HERSHEY_SIMPLEX
  if predicted_captions is not None:
    for i, caption in enumerate(predicted_captions):
      cv2.putText(new_img, caption, (10, up_down_ext+img.shape[0]+(i+1)*25), font, 0.4, (0, 0, 0), 1)
  if true_captions is not None:
    for i, caption in enumerate(true_captions):
      cv2.putText(new_img, caption, (10, 10+(i+1)*25), font, 0.4, (0, 0, 0), 1)

  new_img = new_img * 255
  p = os.path.join(save_path, os.path.basename(fpath))
  print(p)
  cv2.imwrite(os.path.join(save_path, os.path.basename(fpath)), new_img)


def plot_image(fpath, true_captions, predicted_captions, n):
  img = ndimage.imread(fpath)
  fig = plt.figure(figsize=(20, 20))
  plt.subplot(2, 1, 1)
  plt.imshow(img)
  plt.axis('off')

  i = 0
  for t in (predicted_captions):
      i += 1
      plt.text( 5,  i * 14, '%s'%(t) , color='black', backgroundcolor='white', fontsize=14)
  #plt.show()
  fig.savefig('results_model_1/res_image{0}.png'.format(str(n)))


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  #for file_pattern in FLAGS.input_files.split(","):
  #  filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)
  config_sess = tf.ConfigProto()
  config_sess.gpu_options.allow_growth = True

  with tf.Session(graph=g, config=config_sess) as sess:
    # Load the model from checkpoint.
    
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    test_path = r'C:\Users\PSIML-1.PSIML-1\Desktop\projekti\Image-Captioning\test_data'
    filenames = os.listdir(test_path)

    #captions_index = preprocess_captions()
    j = 0
    for filename in filenames:
      full_fname = os.path.join(test_path, filename)
      with tf.gfile.GFile(full_fname, "rb") as f:
        image = f.read()
      
      captions = generator.beam_search(sess, image)
      
      best_captions = []
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        best_captions.append("  %d) %s\n" % (i, sentence))

      #image_idx = int(filename.split('.')[0].split('_')[2])
      #true_captions = captions_index[image_idx]
      
      plot_image(full_fname, None, best_captions, j)
      j += 1


if __name__ == "__main__":
  #tf.app.run()
  main(None) 