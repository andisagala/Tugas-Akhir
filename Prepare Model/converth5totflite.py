import tensorflow as tf

# Load the .h5 model
h5_model_path = 'MobileNetV3Large_50epoch_21_class.h5'
model = tf.keras.models.load_model(h5_model_path)

# print(model.summary())
# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_path = 'MobileNetV3Large_50epoch_21_class.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'MOdel keganti {tflite_model_path}')