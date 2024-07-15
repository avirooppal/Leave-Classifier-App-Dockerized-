import tensorflow as tf
import tensorflowjs as tfjs

# Load your H5 model
model = tf.keras.models.load_model('C:/Users/aviroop/Desktop/aviroop/end_end/potatoes.h5')

# Convert the model
tfjs.converters.save_keras_model(model, 'C:/Users/aviroop/Desktop/aviroop/end_end/classifier_app/assets')