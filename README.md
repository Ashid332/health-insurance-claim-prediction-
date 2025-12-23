# Health Insurance Claim Prediction System

An end-to-end machine learning project that predicts **health insurance claim amounts**
based on customer health and lifestyle data.  
The project covers the complete ML lifecycle — from data preprocessing and model training
to deployment using a **Streamlit web application**.

---

## 🚀 Project Overview

Insurance companies need accurate claim estimations to reduce risk and improve pricing.
This project uses supervised machine learning to predict insurance claim amounts based on
demographic and medical attributes.

The final model is deployed as an interactive web app where users can input details
and instantly get a predicted claim amount.

---

## 🧠 Machine Learning Pipeline

1. Data Collection (`insurance.csv`)
2. Data Cleaning & Feature Engineering
3. Label Encoding (Gender, Smoker, Diabetic)
4. Feature Scaling (Age, BMI, Blood Pressure, Children)
5. Model Training & Evaluation
6. Model Serialization using `joblib`
7. Deployment with Streamlit

---

## 🖥️ Web Application (Streamlit)

The Streamlit app allows users to:
- Enter personal and medical details
- Encode categorical features automatically
- Scale numerical inputs
- Predict insurance claim amount in real time

---

## 📂 Project Structure

```text
├── app.py                         # Streamlit web app
├── train_model.py                 # Model training pipeline
├── test_app.py                    # Basic testing script
├── insurance.csv                  # Dataset
├── scaler.pkl                     # Feature scaler
├── label_encoder_gender.pkl       # Gender encoder
├── label_encoder_diabetic.pkl     # Diabetic encoder
├── label_encoder_smoker.pkl       # Smoker encoder
├── best_model.pkl                 # Trained ML model
├── requirements.txt               # Project dependencies
├── screenshots.zip                # App screenshots
└── Health Insurance Claim Prediction System.docx

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Ashid332/health-insurance-claim-prediction-
cd health-insurance-claim-prediction-

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

📊 Model Inputs
-Age
-BMI
-Gender
-Blood Pressure
-Number of Children
-Smoker Status
-Diabetic Status

📈 Output
-Predicted Health Insurance Claim Amount

🛠 Tech Stack
-Python
-Pandas, NumPy
-Scikit-learn
-Streamlit
-Joblib

📌 Author
Ashidul Islam
Final-year ECE student
Aspiring Data Analyst / Data Scientist

⭐ If you find this project useful, give it a star!
 💾 Step 3: Click **Preview**
Make sure:
- Headings render properly
- Code blocks look clean
- Structure is readable

✅ Step 4: Commit the README

**Commit message:**

Add detailed README with project overview and usage

Commit directly to **main** ✔️

---

## 🔥 What comes AFTER this (your choice)

You are officially done with the repo.  
Next strong moves (tell me which one):
1. 🔗 **LinkedIn post** announcing this project (I’ll write it)
2. 🎤 **Interview explanation** (how to explain this in 60 seconds)
3. ☁️ **Deploy on Streamlit Cloud**
4. 📄 **Resume bullet points** using this project
Pick one.
