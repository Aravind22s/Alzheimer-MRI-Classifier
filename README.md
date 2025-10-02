# 🧠 Alzheimer’s MRI Classification with Deep Learning

An AI-powered web application for **classifying Alzheimer’s disease stages** from brain MRI scans using **ResNet-18** and **Grad-CAM visualization**.  
Built with **PyTorch** and deployed via **Streamlit**.  

---

## 📌 Features
- 📂 Supports **four classes**:
  - 🟢 **Non Demented**  
  - 🟡 **Very Mild Demented**  
  - 🟡 **Mild Demented**  
  - 🟠 **Moderate Demented**
- 🖼️ Upload brain MRI scans (PNG/JPG/JPEG).
- 🤖 Predicts the Alzheimer’s category with AI.
- 🔥 **Grad-CAM** heatmap visualization (highlights brain regions influencing prediction).
- 📊 Displays **model performance metrics** (Confusion Matrix & ROC Curve).
- 💊 Shows **precautions and medical guidance** (educational purposes only).
- 🚀 Easy-to-use **Streamlit Web App**.

---

## 📂 Dataset
This project uses the **[Alzheimer MRI Disease Classification Dataset](https://www.kaggle.com/datasets/falah/Alzheimer-MRI-Disease-Classification)** from Kaggle.

- Train samples: **5120**  
- Test samples: **1280**  
- Labels: *Mild Demented, Moderate Demented, Non Demented, Very Mild Demented*  

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+  
- **Frameworks**: PyTorch, TorchVision  
- **Deployment**: Streamlit  
- **Visualization**: Grad-CAM, Matplotlib, Seaborn  

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aravind22s/Alzheimer-MRI-Classifier.git
   cd Alzheimer-MRI-Classifier

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Activate the virtual environment
   # Linux/Mac
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   
3. **Install dependencies
   ```bash
   pip install -r requirements.txt

4. **Train the model
   ```bash
   python train.py --epochs 10 --batch_size 32 --outdir outputs
   
5. **Run the Streamlit app
   ```bash
   streamlit run app.py
