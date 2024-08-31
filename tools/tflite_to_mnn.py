# tflite_to_onnx.py
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
def load_tflite_model(tflite_model_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter
def convert_tflite_to_tf(interpreter):
    # Extract the TensorFlow model from the TFLite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input and output tensors
    input_shape = input_details[0]['shape']
    input_tensor = tf.convert_to_tensor(interpreter.tensor(input_details[0]['index'])())
    output_tensor = tf.convert_to_tensor(interpreter.tensor(output_details[0]['index'])())
    
    # Create a simple TensorFlow model
    tf_model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return tf_model
def convert_tf_to_onnx(tf_model, onnx_model_path):
    # Convert the TensorFlow model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, opset=13)
    # Save the ONNX model
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
def verify_onnx_model(onnx_model_path):
    # Verify the ONNX model using ONNX Runtime
    session = ort.InferenceSession(onnx_model_path)
    print("ONNX model loaded successfully.")
def main():
    tflite_model_path = "model.tflite"
    onnx_model_path = "model.onnx"
    
    # Load TFLite model
    interpreter = load_tflite_model(tflite_model_path)
    
    # Convert TFLite model to TensorFlow model
    tf_model = convert_tflite_to_tf(interpreter)
    
    # Convert TensorFlow model to ONNX model
    convert_tf_to_onnx(tf_model, onnx_model_path)
    
    # Verify the ONNX model
    verify_onnx_model(onnx_model_path)
    print("Model conversion completed successfully.")
if __name__ == "__main__":
    main()
