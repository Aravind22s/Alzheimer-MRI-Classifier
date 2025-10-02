🧠 Alzheimer’s MRI Classification with Deep Learning

An AI-powered web application for classifying Alzheimer’s disease stages from brain MRI scans using ResNet-18 and Grad-CAM visualization.
Built with PyTorch and deployed via Streamlit.

📌 Features

📂 Supports four classes:

🟢 Non Demented

🟡 Very Mild Demented

🟡 Mild Demented

🟠 Moderate Demented

🖼️ Upload brain MRI scans (PNG/JPG/JPEG).

🤖 Predicts the Alzheimer’s category with AI.

🔥 Grad-CAM heatmap visualization (highlights brain regions influencing prediction).

📊 Displays model performance metrics (Confusion Matrix & ROC Curve).

💊 Shows precautions and medical guidance (educational purposes only).

🚀 Easy-to-use Streamlit Web App.

📂 Dataset

This project uses the Alzheimer MRI Disease Classification Dataset
 from Kaggle.

Train samples: 5120

Test samples: 1280

Labels: Mild Demented, Moderate Demented, Non Demented, Very Mild Demented

🛠️ Tech Stack

Language: Python 3.9+

Frameworks: PyTorch, TorchVision

Deployment: Streamlit

Visualization: Grad-CAM, Matplotlib, Seaborn

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/yourusername/alzheimer-classification.git
cd alzheimer-classification


Create a virtual environment

python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows


Install dependencies

pip install -r requirements.txt


Train the model (optional if best_model.pt exists)

python train.py --epochs 10 --batch_size 32 --outdir outputs


Run the Streamlit app

streamlit run app.py

🚀 Usage

Open the Streamlit web app in your browser (default: http://localhost:8501).

Upload a brain MRI scan (PNG/JPG).

View:

Predicted Alzheimer’s stage

Category details and precautions

Grad-CAM heatmap (AI attention map)

Model performance plots

📊 Example Output

✅ Prediction: Mild Demented

🔥 Grad-CAM Heatmap showing regions of focus

💊 Precautions: Regular doctor visits, cognitive exercises, healthy diet, etc.

📸 Screenshots

(Add images here – e.g., upload an MRI input, prediction result page, and Grad-CAM heatmap)

⚠️ Disclaimer

This tool is for educational and research purposes only.
It is not intended for clinical or diagnostic use.
Always consult healthcare professionals for medical decisions.

👨‍💻 Author

ARAVINDAN – [GitHub Profile](https://github.com/Aravind22s)

Contributions, issues, and feature requests are welcome!

⭐ Contribute

If you like this project, please ⭐ the repo and share with others!
