import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
import keras.backend as K

# Define and register the corrected SelfAttention layer
@keras.utils.register_keras_serializable()
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer="random_normal", trainable=True)
        self.W_k = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer="random_normal", trainable=True)
        self.W_v = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer="random_normal", trainable=True)

    def call(self, inputs):
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)

        attention_scores = tf.nn.softmax(tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) / (inputs.shape[-1] ** 0.5))
        return tf.matmul(attention_scores, V)

# Define and register the F1Score metric
@keras.utils.register_keras_serializable()
class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true * y_pred, "float32"))
        fp = K.sum(K.cast((1 - y_true) * y_pred, "float32"))
        fn = K.sum(K.cast(y_true * (1 - y_pred), "float32"))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        return 2 * (precision * recall) / (precision + recall + K.epsilon())

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

# Streamlit App
st.title("Monkeypox Classification")
#st.write("Loading model...")

# Load model with error handling
model = None
try:
    custom_objects = {
        "SelfAttention": SelfAttention,
        "F1Score": F1Score
    }
    model = keras.models.load_model("monkeypox_cnn_bgru_sa.keras", custom_objects=custom_objects)
    st.success("Model loaded successfully!")
    st.text(model.summary())  # Print summary to check the layers
except Exception as e:
    st.error(f"Error loading model: {e}")

# Ensure model is loaded before making predictions
if model:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, axis=0) / 255.0  # Normalize
        
        prediction = model.predict(img_array)
        prediction_class = prediction[0][0]
        #st.write(prediction)
        result = "Monkeypox Detected" if prediction_class > 0.80 else "No Monkeypox"
        
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"### Prediction: **{result}**")
else:
    st.warning("Model could not be loaded. Please check the file path and format.")
