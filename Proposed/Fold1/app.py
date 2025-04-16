import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras.preprocessing import image

# Define and register the SelfAttention layer
@keras.utils.register_keras_serializable()
class SelfAttention(Layer):
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.attention_dense = keras.layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        attention_weights = self.attention_dense(inputs)
        return inputs * attention_weights

# Define and register the F1Score metric
@keras.utils.register_keras_serializable()
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Streamlit App
st.title("Monkeypox Classification with CNN-BiGRU-SelfAttention")

# Load model with error handling
model = None
try:
    custom_objects = {
        "SelfAttention": SelfAttention,
        "F1Score": F1Score
    }
    model = keras.models.load_model("monkeypox_cnn_bgru_sa.keras", custom_objects=custom_objects)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Ensure model is loaded before making predictions
if model:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_obj = image.load_img(uploaded_file, target_size=(64, 64))
        img_array = image.img_to_array(image_obj) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        prediction = model.predict(img_array)
        predicted_class = prediction[0][0]
        #st.write(predicted_class)
        # Define class labels
        result = "No Monkeypox" if predicted_class == 1.0 or predicted_class <= 1.049e-12 else "Monkeypox Detected"
        
        
        st.image(image_obj, caption="Uploaded Image", use_column_width=True)
        st.write(f"### Prediction: **{result}**")
else:
    st.warning("Model could not be loaded. Please check the file path and format.")
