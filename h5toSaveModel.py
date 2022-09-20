# import tensorflow as tf
# import keras
# from tensorflow.python.estimator.export import export

# estimator = tf.keras.estimator.model_to_estimator(keras_model_path='./assets/model/RasSignModelCNN_9-2.h5', model_dir='./assets/model/estimator')

# feature_spec = {'conv2d_1_input': model.input}
# serving_input_fn = export.build_raw_serving_input_receiver_fn(feature_spec)
# estimator._model_dir = './keras'
# estimator.export_savedmodel('iris-20200719', serving_input_fn)

# import tensorflow as tf
# model = tf.keras.models.load_model('./assets/model/RasSignModelCNN_9-2.h5')
# from tensorflow import keras
# keras.experimental.export_saved_model(model,'./assets/model/model/1/')

import tensorflow as tf

model = tf.keras.models.load_model('./assets/model/RasSignModelCNN_9-2.h5')
tf.saved_model.save(model, './assets/model/model/1/')