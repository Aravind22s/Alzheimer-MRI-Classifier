import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from gradcam import GradCAM, overlay_cam_on_image
from model import get_resnet18

st.set_page_config(page_title="Alzheimer MRI Classifier", layout="wide", page_icon="🧠")

# ---- MEDICAL DASHBOARD THEME ----
st.markdown("""
<style>
    body {
        background-color: #f9fafb;
        font-family: "Segoe UI", sans-serif;
    }

    /* HEADER */
    .app-header {
        text-align: center;
        padding: 1.8rem 1rem;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: #fff;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: bold;
    }
    .app-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* CARDS */
    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        color: #111827 !important;   /* ensure visible text */
    }
    .card h3, .card p, .card li, .card div, .card span {
        color: #111827 !important;
    }

    /* PREDICTION BOX */
    .prediction {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        padding: 1.8rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: #fff;
        margin: 1rem 0;
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }

    /* PRECAUTION LIST */
    .precaution {
        background: #f3f4f6;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-left: 5px solid #2563eb;
        border-radius: 6px;
        font-size: 1rem;
        color: #111827 !important;
    }

    /* FOOTER */
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* UPLOADER */
    .stFileUploader>div>div {
        border: 2px dashed #2563eb !important;
        border-radius: 10px;
        padding: 1rem;
        background: #f3f4f6;
    }
</style>
""", unsafe_allow_html=True)

# ---- CATEGORY INFO ----
CATEGORY_INFO = {
    "Mild_Demented": {
        "description": "Mild Demented: Early memory issues but still able to function with little help.",
        "precautions": ["Regular doctor visits", "Cognitive exercises", "Healthy diet", "Stay socially active"],
        "icon": "🟡"
    },
    "Moderate_Demented": {
        "description": "Moderate Demented: Noticeable memory loss, confusion, difficulty with daily tasks.",
        "precautions": ["Supervision in daily life", "Medication adherence", "Safe living environment"],
        "icon": "🟠"
    },
    "Non_Demented": {
        "description": "Non-Demented: No significant memory or cognitive issues.",
        "precautions": ["Maintain healthy lifestyle", "Brain exercises", "Regular checkups"],
        "icon": "🟢"
    },
    "Very_Mild_Demented": {
        "description": "Very Mild Demented: Subtle memory issues, often dismissed as normal aging.",
        "precautions": ["Early medical consultation", "Routine monitoring", "Stay physically active"],
        "icon": "🟡"
    }
}

# ---- LOAD MODEL ----
@st.cache_resource
def load_model(model_path="outputs/best_model.pt", device='cpu'):
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}. Train first.")
        st.stop()
    ckpt = torch.load(model_path, map_location=device)
    label_map = ckpt['label_map']
    num_classes = len(label_map)
    model = get_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, label_map

def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    x = transform(img.convert('L'))
    if x.shape[0] == 1:
        x = x.repeat(3,1,1)
    return x.unsqueeze(0)

# ---- SIDEBAR NAVIGATION ----
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📤 Upload & Predict", "📈 Model Metrics", "ℹ️ About"])

# ---- HEADER ----
st.markdown('<div class="app-header"><h1>🧠 Alzheimer MRI Classifier</h1><p>Medical Dashboard for Brain MRI Analysis</p></div>', unsafe_allow_html=True)

# ---- HOME ----
if page == "🏠 Home":
    st.markdown("""
    <div class="card">
    <h3>Welcome</h3>
    <p>This dashboard uses a deep learning model (ResNet-18) to classify Alzheimer’s disease from brain MRI scans. 
    Upload an MRI image to receive AI-powered predictions, explanations (Grad-CAM), and recommended precautions.</p>
    <p><b>Disclaimer:</b> This tool is for research and educational purposes only. Consult healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)

# ---- UPLOAD & PREDICT ----
elif page == "📤 Upload & Predict":
    st.markdown('<div class="card"><h3>Upload MRI Image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an MRI scan", type=["png","jpg","jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        model, label_map = load_model("outputs/best_model.pt", device='cpu')
        inv_label_map = {v:k for k,v in label_map.items()}

        with st.spinner("Analyzing image..."):
            x = preprocess_image(image)
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = inv_label_map[pred_idx]

        # Layout
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown('<div class="card"><h3>🖼️ MRI Scan</h3>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            icon = CATEGORY_INFO[pred_label]["icon"]
            st.markdown(f'<div class="prediction">{icon} {pred_label.replace("_", " ")}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{probs[pred_idx]*100:.1f}%")

        # Description & Precautions
        st.markdown('<div class="card"><h3>📊 Details</h3>', unsafe_allow_html=True)
        st.write(CATEGORY_INFO[pred_label]["description"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>💊 Precautions</h3>', unsafe_allow_html=True)
        for p in CATEGORY_INFO[pred_label]["precautions"]:
            st.markdown(f'<div class="precaution">{p}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Grad-CAM
        st.markdown('<div class="card"><h3>🔥 Grad-CAM Visualization</h3>', unsafe_allow_html=True)
        if st.checkbox("Show Grad-CAM Heatmap"):
            with st.spinner("Generating Grad-CAM..."):
                target_layer = model.layer4[-1]
                gradcam = GradCAM(model, target_layer)
                x_grad = x.clone().requires_grad_(True)
                cam_mask = gradcam.generate(x_grad, target_class=pred_idx)
                overlay = overlay_cam_on_image(image, cam_mask, alpha=0.5)
            st.image(overlay, use_column_width=True, caption="Grad-CAM Heatmap")
        st.markdown('</div>', unsafe_allow_html=True)

# ---- METRICS ----
elif page == "📈 Model Metrics":
    st.markdown('<div class="card"><h3>Model Performance</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("outputs/confusion.png"):
            st.image("outputs/confusion.png", caption="Confusion Matrix", use_column_width=True)
    with col2:
        if os.path.exists("outputs/roc.png"):
            st.image("outputs/roc.png", caption="ROC Curve", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- ABOUT ----
elif page == "ℹ️ About":
    st.markdown("""
    <div class="card">
    <h3>About This Tool</h3>
    <p><b>Model:</b> ResNet-18 convolutional neural network</p>
    <p><b>Task:</b> Classify brain MRI scans into Alzheimer categories</p>
    <p><b>Categories:</b> Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented</p>
    <p><b>Disclaimer:</b> Not a diagnostic tool. Use for research/education only.</p>
    </div>
    """, unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("<div class='footer'>🧠 Alzheimer MRI Classifier | Medical Dashboard | ResNet-18</div>", unsafe_allow_html=True)
