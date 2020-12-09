# -*- coding: utf-8 -*-
#ライブラリのインポート
import tensorflow as tf
import numpy as np
import importlib
import matplotlib.pyplot as plt
import pathlib
    
def model_compression(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"mnist_model.tflite"
    tflite_model_file.write_bytes(tflite_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16_model = converter.convert()
    tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant_f16.tflite"
    tflite_model_fp16_file.write_bytes(tflite_fp16_model)
    
    return(tflite_model_fp16_file)

def model_run(tflite_model_fp16_file,x):

    interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
    interpreter_fp16.allocate_tensors()
    
    test_image = x

    input_index = interpreter_fp16.get_input_details()[0]["index"]
    output_index = interpreter_fp16.get_output_details()[0]["index"]

    interpreter_fp16.set_tensor(input_index, test_image)
    interpreter_fp16.invoke()
    predictions = interpreter_fp16.get_tensor(output_index)
    predict = predictions[0]
    return( predict )

def mnist_loading():
    mnist = tf.keras.datasets.mnist
    _, (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test[..., tf.newaxis]
    return(x_test,y_test)

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)
#モデルのロード
model = keras.models.load_model('fp16_model')

model = model_compression(model)

print(model_run(model,test_image))