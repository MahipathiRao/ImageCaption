import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout, Add
import streamlit as st
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# Load InceptionV3 for feature extraction
@st.cache_resource
def load_inception_model():
    base_model = InceptionV3(weights="imagenet", include_top=False)
    model = tf.keras.Model(base_model.input, tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
    return model

# Preprocess the image for InceptionV3
def preprocess_image(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Error in processing image: {e}")
        return None

# Extract features using InceptionV3
def extract_features(model, image_path):
    img = preprocess_image(image_path)
    if img is not None:
        features = model.predict(img)
        return np.squeeze(features)  # Ensure the feature shape is (2048,)
    else:
        return None

# Build the caption generation model
def build_model(vocab_size, max_length, embedding_dim=256, units=512):
    # Image feature input
    img_input = Input(shape=(2048,))
    img_features = Dropout(0.5)(img_input)
    img_features = Dense(units, activation="relu")(img_features)

    # Text input (captions)
    cap_input = Input(shape=(max_length,))
    cap_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(cap_input)
    cap_lstm = Dropout(0.5)(cap_embedding)
    cap_lstm = LSTM(units)(cap_lstm)

    # Combine features
    combined = Add()([img_features, cap_lstm])
    outputs = Dense(vocab_size, activation="softmax")(combined)

    model = Model(inputs=[img_input, cap_input], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

# Generate captions using Beam Search
def generate_caption_beam_search(model, tokenizer, image_features, max_length, beam_size=3):
    sequences = [[[], 0.0]]  # [sequence, score]
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            padded = pad_sequences([tokenizer.texts_to_sequences([" ".join(seq)])[0]], maxlen=max_length,
                                   padding="post")
            predictions = model.predict([np.expand_dims(image_features, axis=0), padded], verbose=0)
            top_candidates = np.argsort(predictions[0])[-beam_size:]  # Get top words
            for word_idx in top_candidates:
                word = tokenizer.index_word.get(word_idx, None)
                if word:
                    candidate = [seq + [word], score - np.log(predictions[0][word_idx])]
                    all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_size]
    return " ".join(sequences[0][0])

# Load Dataset and Tokenizer (Example with COCO Dataset)
def load_dataset():
    captions = ["a dog running", "a cat sleeping", "a man riding a bike", "a child playing with a ball"]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(seq.split()) for seq in captions)
    return captions, tokenizer, vocab_size, max_length

# Prepare dataset for training
def create_sequences(tokenizer, max_length, captions, image_features):
    X1, X2, y = [], [], []
    for i, caption in enumerate(captions):
        seq = tokenizer.texts_to_sequences([caption])[0]
        for j in range(1, len(seq)):
            in_seq, out_seq = seq[:j], seq[j]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(image_features[i])
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Load BLIP model and processor
@st.cache_resource
def load_blip_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# Generate caption using BLIP model
def generate_caption_blip(processor, model, image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit App
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Option 1: Generate caption using BLIP model
        processor, blip_model = load_blip_model()
        blip_caption = generate_caption_blip(processor, blip_model, image)
        st.write("Generated Caption :", blip_caption)



    except Exception as e:
        st.error(f"An error occurred: {e}")
