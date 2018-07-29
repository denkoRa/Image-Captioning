import configuration
import inference_utils
import inference_wrapper

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

    test_path = r'C:\Users\PSIML-1.PSIML-1\Desktop\projekti\Image-Captioning\test_gradient'
    
    for filename in filenames:
        full_fname = os.path.join(test_path, filename)
        with tf.gfile.GFile(full_fname, "rb") as f:
            image = f.read()

        initial_state = model.feed_image(sess, image)
        
        for i in range(20):            
            softmax, new_states, metadata = model.inference_step(sess, input_feed, state_feed)



if __name__ == "__main__":
  #tf.app.run()
  main(None)