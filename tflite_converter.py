import tensorflow as tf

model = tf.keras.models.load_model('trash_classifier_model_finetuned.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('trash_classifier_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Optimized TFLite model successfully created!")




