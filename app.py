import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras import layers, Model

# -------------------------------------------------------------
# CUSTOM LAYERS (MUST MATCH TRAINING EXACTLY)
# -------------------------------------------------------------

def se_block(input_tensor, reduction=16):
    channels = int(input_tensor.shape[-1])
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(max(channels // reduction, 8), activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1,1,channels))(se)
    return layers.Multiply()([input_tensor, se])

def transformer_encoder(x, num_heads=4, key_dim=64, mlp_dim=128, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    mlp = layers.Dense(mlp_dim, activation='relu')(x)
    mlp = layers.Dropout(dropout)(mlp)
    mlp = layers.Dense(key_dim)(mlp)
    x = layers.Add()([x, mlp])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x


# ---------------------------
# REAL FFTHead FROM TRAINING
# ---------------------------
class FFTHead(layers.Layer):
    def __init__(self, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D((2,2))
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.gap = layers.GlobalAveragePooling2D()

    def call(self, x):
        if int(x.shape[-1]) == 3:
            gray = tf.image.rgb_to_grayscale(x)
        else:
            gray = x
        gray_c = tf.cast(gray, tf.complex64)
        fft = tf.signal.fft2d(gray_c)
        mag = tf.abs(fft)
        mag = tf.math.log(mag + 1e-8)
        if self.normalize:
            denom = tf.reduce_max(mag, axis=[1,2,3], keepdims=True) + 1e-8
            mag = mag / denom
        x = self.conv1(mag)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.gap(x)
        return x


def cross_attention_fusion(cnn_feat, vit_tokens, num_heads=4):
    feat_dim = int(cnn_feat.shape[-1])
    cnn_token = layers.Reshape((1, feat_dim))(cnn_feat)

    attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feat_dim)(
        query=cnn_token, value=vit_tokens, key=vit_tokens
    )
    attn1 = layers.Add()([cnn_token, attn1])
    attn1 = layers.LayerNormalization()(attn1)

    vit_embed_dim = int(vit_tokens.shape[-1])
    attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=vit_embed_dim)(
        query=vit_tokens, value=attn1, key=attn1
    )
    attn2 = layers.Add()([vit_tokens, attn2])
    attn2 = layers.LayerNormalization()(attn2)

    pooled_cnn = layers.GlobalAveragePooling1D()(attn1)
    pooled_vit = layers.GlobalAveragePooling1D()(attn2)
    fused = layers.Concatenate()([pooled_cnn, pooled_vit])
    return fused


def BicephNet_X(input_shape=(150,150,3), num_classes=3,
                cnn_filters=(32,64,128,192),
                vit_embed_dim=48, vit_num_heads=3, vit_depth=2,
                patch_size=16, dropout_rate=0.5):

    inp = layers.Input(shape=input_shape)

    # CNN head
    x = inp
    for f in cnn_filters:
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = se_block(x, reduction=8)
        x = layers.MaxPooling2D((2,2))(x)
    cnn_feat = layers.GlobalAveragePooling2D()(x)

    # ViT tokens
    vit = layers.Conv2D(vit_embed_dim, kernel_size=patch_size, strides=patch_size, padding='same')(inp)
    vit = layers.Reshape((-1, vit_embed_dim))(vit)
    for _ in range(vit_depth):
        vit = transformer_encoder(vit, num_heads=vit_num_heads, key_dim=vit_embed_dim, mlp_dim=vit_embed_dim*2)
    vit_tokens = vit

    # FFT head
    fft_vec = FFTHead()(inp)

    # Cross-attention fusion
    cross_vec = cross_attention_fusion(cnn_feat, vit_tokens, num_heads=vit_num_heads)

    # Final fusion
    fused = layers.Concatenate()([cross_vec, fft_vec])
    fused = layers.Dense(256, activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(dropout_rate)(fused)
    out = layers.Dense(num_classes, activation='softmax')(fused)

    return Model(inp, out, name="BicephNet_X_CT")


# -------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(page_title="Brain CT/MRI Classifier", layout="wide")
st.title("🧬 Brain Disease Detection")


# -------------------------------------------------------------
# Model Loader
# -------------------------------------------------------------
@st.cache_resource
def load_model(file_id, filename):
    if not os.path.exists(filename):
        st.info(f"⬇️ Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

    try:
        model = tf.keras.models.load_model(
            filename,
            custom_objects={
                "FFTHead": FFTHead,
                "se_block": se_block,
                "transformer_encoder": transformer_encoder,
                "cross_attention_fusion": cross_attention_fusion,
                "BicephNet_X": BicephNet_X
            }
        )
        st.success(f"Loaded: {filename}")
        return model
    except Exception as e:
        st.error(str(e))
        st.stop()


# -------------------------------------------------------------
# Load ONLY the CT stroke + CT tumor models
# -------------------------------------------------------------

MODEL_CT_STROKE_ID = "1YsNTcNx7tp0AUaXAas2FeTD55yTShXpQ"
MODEL_CT_TUMOR_ID  = "1unzfP0Td-rDqqF6-dUw_4RUjH_LwTr98"

model_ct_stroke = load_model(MODEL_CT_STROKE_ID, "best_CT_STROKE_3CLASS.h5")
model_ct_tumor  = load_model(MODEL_CT_TUMOR_ID,  "best_CT_TUMOR_2CLASS.h5")


# -------------------------------------------------------------
# Preprocess
# -------------------------------------------------------------
def preprocess(image):
    image = image.resize((150,150))
    arr = np.array(image)/255.0
    return np.expand_dims(arr, axis=0)


# -------------------------------------------------------------
# Predictions
# -------------------------------------------------------------

classes_ct_stroke = ['Haemorrhagic', 'Ischemic', 'Normal']
classes_ct_tumor  = ['Healthy', 'tumor']

def predict_stroke(img):
    x = preprocess(img)
    p = model_ct_stroke.predict(x)
    idx = np.argmax(p)
    return classes_ct_stroke[idx], float(np.max(p)*100)

def predict_tumor(img):
    x = preprocess(img)
    p = model_ct_tumor.predict(x)
    idx = np.argmax(p)
    return classes_ct_tumor[idx], float(np.max(p)*100)


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
task = st.radio("Choose CT Task", ["Stroke", "Tumor"], horizontal=True)

file = st.file_uploader("Upload CT Image", type=["png","jpg","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)

    if st.button("Predict"):
        if task=="Stroke":
            label, conf = predict_stroke(img)
            st.success(f"Stroke Prediction: {label} ({conf:.2f}%)")
        else:
            label, conf = predict_tumor(img)
            st.success(f"Tumor Prediction: {label} ({conf:.2f}%)")
