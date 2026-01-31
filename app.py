import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.layers import Layer

# -------------------------------------------------------------
# Custom Layer: FFTHead (required for CT Stroke model)
# -------------------------------------------------------------
class FFTHead(Layer):
    def __init__(self, **kwargs):
        super(FFTHead, self).__init__(**kwargs)

    def call(self, inputs):
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        magnitude = tf.abs(fft)
        return magnitude

    def get_config(self):
        return super().get_config()

# -------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------
st.set_page_config(page_title="🧠 Brain Disease Classifier", layout="wide")
st.title("🧬 Brain Disease Detection & Stage/Subtype Classification")
st.write(
    "Upload a medical image and select the modality.\n\n"
    "- **MRI:** Alzheimer / Tumor / Parkinson / Healthy → (Alzheimer stage or Tumor type if applicable)\n"
    "- **CT:** Stroke subtype (Haemorrhagic/Ischemic/Normal) OR Tumor vs Healthy\n"
)

# -------------------------------------------------------------
# Model Loader (Downloads from Google Drive if not found)
# -------------------------------------------------------------
@st.cache_resource
def load_model(file_id, filename):
    if not os.path.exists(filename):
        st.info(f"⬇️ Downloading {filename} from cloud storage...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

    try:
        model = tf.keras.models.load_model(
            filename,
            custom_objects={"FFTHead": FFTHead}  # ⭐ FIX ADDED
        )
        st.success(f"✅ Loaded: {filename}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading {filename}: {e}")
        st.stop()

# -------------------------------------------------------------
# 🔸 Google Drive File IDs (REPLACE WITH YOUR ACTUAL IDs)
# -------------------------------------------------------------
# MRI models
MODEL_MRI_MAIN_ID  = "1fFMLtCrUW_54j5UAvxcqblvEIVFKzADM"
MODEL_MRI_ALZ_ID   = "1pwklSWRBo20jQAMD3DVTv8EgFngxaskh"
MODEL_MRI_TUMOR_ID = "11fpYFViTKGZvX3_T08JdxieQu_uijkoT"

# CT models
MODEL_CT_STROKE_ID = "1YsNTcNx7tp0AUaXAas2FeTD55yTShXpQ"  # best_CT_STROKE_3CLASS.h5
MODEL_CT_TUMOR_ID  = "1unzfP0Td-rDqqF6-dUw_4RUjH_LwTr98"  # best_CT_TUMOR_2CLASS.h5

# -------------------------------------------------------------
# Load Models
# -------------------------------------------------------------
model_mri_main   = load_model(MODEL_MRI_MAIN_ID,  "model_level1_main_disease.h5")
model_mri_alz    = load_model(MODEL_MRI_ALZ_ID,   "model_level2_alzheimer_stage.h5")
model_mri_tumor  = load_model(MODEL_MRI_TUMOR_ID, "model_level2_tumor_type.h5")

model_ct_stroke  = load_model(MODEL_CT_STROKE_ID, "best_CT_STROKE_3CLASS.h5")
model_ct_tumor   = load_model(MODEL_CT_TUMOR_ID,  "best_CT_TUMOR_2CLASS.h5")

# -------------------------------------------------------------
# Class Labels
# -------------------------------------------------------------
# MRI Level-1
classes_mri_main  = ['alzheimer', 'healthy', 'parkinson', 'tumor']

# MRI Level-2
classes_alz       = ['Mild Demented', 'Moderate Demented', 'Non-Demented', 'Very Mild Demented']
classes_tumor_mri = ['Glioma', 'Meningioma', 'Pituitary']

# CT tasks
classes_ct_stroke = ['Haemorrhagic', 'Ischemic', 'Normal']
classes_ct_tumor  = ['Healthy', 'tumor']

# -------------------------------------------------------------
# Recommendations
# -------------------------------------------------------------
recommendations = {
    "stroke": {
        "Haemorrhagic": {
            "Precautions": [
                "Seek emergency medical care immediately.",
                "Monitor blood pressure regularly.",
                "Avoid self-medication, especially blood thinners."
            ],
            "Medications": [
                "Antihypertensive therapy (as prescribed)",
                "Neurosurgical management if indicated"
            ]
        },
        "Ischemic": {
            "Precautions": [
                "Control cholesterol and blood pressure.",
                "Avoid smoking and alcohol.",
                "Start physiotherapy and mobility exercises as advised."
            ],
            "Medications": [
                "Antiplatelets (Aspirin/Clopidogrel as prescribed)",
                "Statins (as prescribed)"
            ]
        },
        "Normal": {
            "Precautions": ["Maintain healthy lifestyle and regular checkups."],
            "Medications": ["No medication required."]
        }
    },
    "ct_tumor": {
        "tumor": {
            "Precautions": [
                "Consult a neurologist/neurosurgeon for confirmation.",
                "Follow-up imaging may be required.",
                "Report headaches/vision changes immediately."
            ],
            "Medications": [
                "Steroids/anti-seizure medication only if prescribed",
                "Definitive treatment depends on evaluation"
            ]
        },
        "Healthy": {
            "Precautions": ["Maintain good lifestyle habits and routine health checks."],
            "Medications": ["No medication required."]
        }
    }
}

# -------------------------------------------------------------
# Upload + Select Modality
# -------------------------------------------------------------
modality = st.selectbox("Select modality", ["MRI", "CT"])

ct_task = None
if modality == "CT":
    ct_task = st.radio("Select CT task", ["Stroke classification", "Tumor vs Healthy"], horizontal=True)

uploaded_file = st.file_uploader("📁 Upload Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------------
# Preprocess
# -------------------------------------------------------------
def preprocess(image):
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------------------------------------------------
# MRI Prediction
# -------------------------------------------------------------
def predict_mri(image):
    x = preprocess(image)

    main_pred = model_mri_main.predict(x)
    main_idx = int(np.argmax(main_pred))
    main_label = classes_mri_main[main_idx]
    main_conf = float(np.max(main_pred) * 100)

    sub_label, sub_conf = None, None

    if main_label == "alzheimer":
        sub_pred = model_mri_alz.predict(x)
        sub_label = classes_alz[int(np.argmax(sub_pred))]
        sub_conf = float(np.max(sub_pred) * 100)

    elif main_label == "tumor":
        sub_pred = model_mri_tumor.predict(x)
        sub_label = classes_tumor_mri[int(np.argmax(sub_pred))]
        sub_conf = float(np.max(sub_pred) * 100)

    return main_label, main_conf, sub_label, sub_conf

# -------------------------------------------------------------
# CT Prediction
# -------------------------------------------------------------
def predict_ct_stroke(image):
    x = preprocess(image)
    pred = model_ct_stroke.predict(x)
    label = classes_ct_stroke[int(np.argmax(pred))]
    conf = float(np.max(pred) * 100)
    return label, conf

def predict_ct_tumor(image):
    x = preprocess(image)
    pred = model_ct_tumor.predict(x)
    label = classes_ct_tumor[int(np.argmax(pred))]
    conf = float(np.max(pred) * 100)
    return label, conf

# -------------------------------------------------------------
# Prediction and Display
# -------------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🧩 Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):

            if modality == "MRI":
                main_label, main_conf, sub_label, sub_conf = predict_mri(image)

                st.success(f"🧠 **MRI Diagnosis:** {main_label.capitalize()} ({main_conf:.2f}% confidence)")
                if sub_label:
                    st.info(f"**Subtype/Stage:** {sub_label} ({sub_conf:.2f}% confidence)")

                st.info("⚕️ Please consult a qualified clinician for confirmation.")

            else:
                if ct_task == "Stroke classification":
                    stroke_label, stroke_conf = predict_ct_stroke(image)
                    st.success(f"🩻 **CT Stroke Result:** {stroke_label} ({stroke_conf:.2f}% confidence)")

                    st.subheader("🩺 Precautions")
                    for p in recommendations["stroke"][stroke_label]["Precautions"]:
                        st.write(f"- {p}")

                    st.subheader("💊 Medications")
                    for m in recommendations["stroke"][stroke_label]["Medications"]:
                        st.write(f"- {m}")

                else:
                    tumor_label, tumor_conf = predict_ct_tumor(image)
                    st.success(f"🩻 **CT Tumor Result:** {tumor_label} ({tumor_conf:.2f}% confidence)")

                    st.subheader("🩺 Precautions")
                    for p in recommendations["ct_tumor"][tumor_label]["Precautions"]:
                        st.write(f"- {p}")

                    st.subheader("💊 Medications")
                    for m in recommendations["ct_tumor"][tumor_label]["Medications"]:
                        st.write(f"- {m}")

                st.info("⚕️ CT results are supportive; clinical correlation is recommended.")

else:
    st.warning("⚠️ Please upload an image to begin analysis.")
