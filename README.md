# 💵 Health Insurance Claim Prediction System

## 📋 Project Overview

An end-to-end machine learning project that predicts **health insurance claim amounts** based on customer health and lifestyle data. This project demonstrates the complete ML lifecycle — from data preprocessing and feature engineering to model training, evaluation, and deployment using a **Streamlit web application**.

**Real-world impact:** Help insurance companies estimate claim costs accurately to optimize pricing strategies and reduce financial risk.

## 🧠 Problem Statement

Insurance companies face significant challenges in:
- **Risk Assessment**: Estimating expected claim amounts from customers
- **Pricing Optimization**: Setting competitive premiums while maintaining profitability
- **Cost Prediction**: Understanding the financial impact of customer demographics and health factors

This project solves these problems using supervised machine learning to predict claim amounts based on demographic and medical attributes.

## 🚀 Key Features

✅ **End-to-End ML Pipeline** - Complete workflow from data to deployment
✅ **Feature Engineering** - Label encoding for categorical variables + feature scaling
✅ **Model Serialization** - Pickle/joblib for model persistence
✅ **Interactive Web App** - Streamlit UI for real-time predictions
✅ **Production-Ready** - Clean code structure with separate training/app modules

## 🔧 ML Pipeline Architecture

```
1. DATA COLLECTION
   └─ insurance.csv (Raw customer & medical data)

2. DATA PREPROCESSING
   ├─ Handle missing values
   ├─ Remove outliers
   ├─ Feature scaling (StandardScaler)
   └─ Label encoding (categorical variables)

3. FEATURE ENGINEERING
   ├─ Age, BMI, Blood Pressure, Children (numerical)
   └─ Gender, Smoker, Diabetic (categorical → encoded)

4. MODEL TRAINING & EVALUATION
   ├─ Train/test split (80/20)
   ├─ Multiple algorithms tested
   ├─ Hyperparameter tuning
   └─ Performance metrics (R², MAE, RMSE)

5. MODEL SERIALIZATION
   ├─ best_model.pkl (trained regressor)
   ├─ scaler.pkl (feature scaler)
   └─ label_encoders (gender, smoker, diabetic)

6. DEPLOYMENT
   └─ Streamlit app for user predictions
```

## 📊 Dataset Information

**Source:** Insurance customer data with health attributes
**Samples:** ~1,300 records
**Target Variable:** Health insurance claim amount (continuous)

### Features:
| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Customer age in years |
| BMI | Numerical | Body Mass Index |
| Blood Pressure | Numerical | Systolic BP reading |
| Children | Numerical | Number of dependents |
| Gender | Categorical | Male/Female (encoded) |
| Smoker | Categorical | Yes/No (encoded) |
| Diabetic | Categorical | Yes/No (encoded) |

### Target Variable:
- **Claim Amount** - Health insurance charges (continuous, regression task)

## 🛠️ Tech Stack

- **Python 3.8+** - Core language
- **Pandas & NumPy** - Data manipulation & numerical computing
- **Scikit-learn** - Machine learning algorithms & preprocessing
- **Joblib** - Model serialization
- **Streamlit** - Interactive web application
- **Matplotlib & Seaborn** - Data visualization

## 📁 Project Structure

```
health-insurance-claim-prediction/
├── app.py                          # Streamlit web app
├── train_model.py                  # Model training pipeline
├── test_app.py                     # Unit tests
├── insurance.csv                   # Training dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── LICENSE                         # MIT License
├── models/
│   ├── best_model.pkl              # Trained regression model
│   ├── scaler.pkl                  # Feature scaler
│   ├── label_encoder_gender.pkl    # Gender encoder
│   ├── label_encoder_smoker.pkl    # Smoker status encoder
│   └── label_encoder_diabetic.pkl  # Diabetic status encoder
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   └── model_training.ipynb        # Model development
├── data/
│   ├── processed/                  # Cleaned data
│   └── raw/                        # Raw dataset
└── src/
    └── preprocessing.py            # Data cleaning functions
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Ashid332/health-insurance-claim-prediction-.git
cd health-insurance-claim-prediction-
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Application

**Launch the Streamlit app:**
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Training the Model (Optional)

To retrain the model with new data:
```bash
python train_model.py
```

## 📖 How to Use the App

1. **Input Customer Details:**
   - Age (18-75 years)
   - Gender (Male/Female)
   - BMI (15-50)
   - Blood Pressure (80-180 mmHg)
   - Number of Children (0-5)
   - Smoker Status (Yes/No)
   - Diabetic Status (Yes/No)

2. **Click Predict Button:**
   - The app automatically encodes categorical features
   - Applies feature scaling
   - Runs inference on the trained model

3. **Get Instant Prediction:**
   - Displays predicted claim amount
   - Shows confidence/model accuracy info

## 📊 Model Performance

**Regression Metrics:**
- **R² Score:** Coefficient of determination (model explains X% of variance)
- **MAE (Mean Absolute Error):** Average prediction error in dollars
- **RMSE (Root Mean Squared Error):** Penalizes larger errors
- **MAPE (Mean Absolute Percentage Error):** Percentage accuracy

*Note: Run the model to see actual performance metrics*

## 🚀 Deployment Options

### Option 1: Local Streamlit
```bash
streamlit run app.py
```

### Option 2: Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Deploy directly from your repo

### Option 3: Docker
```bash
docker build -t insurance-predictor .
docker run -p 8501:8501 insurance-predictor
```

## 💪 Learning Outcomes

This project demonstrates:

✔️ **Data Preprocessing** - Cleaning, scaling, encoding
✔️ **Feature Engineering** - Selecting relevant features
✔️ **Model Selection** - Choosing appropriate algorithms
✔️ **Hyperparameter Tuning** - Optimizing model performance
✔️ **Model Evaluation** - Using appropriate metrics
✔️ **Model Serialization** - Saving/loading trained models
✔️ **Web Deployment** - Building user-facing ML applications
✔️ **Clean Code** - Modular, maintainable structure

## 💬 Key Insights

- **Age & BMI** are strong predictors of claim amounts
- **Smoking status** significantly increases expected claims
- **Diabetic status** is an important risk factor
- **Feature scaling** improves model convergence
- **Label encoding** handles categorical variables effectively

## 📝 Future Improvements

- [ ] Add more features (occupation, income level, medical history)
- [ ] Implement ensemble methods (Random Forest, XGBoost, Gradient Boosting)
- [ ] Add prediction intervals/confidence bounds
- [ ] Create user authentication for production deployment
- [ ] Add monitoring and logging for model performance
- [ ] Implement A/B testing framework
- [ ] Build API endpoints (Flask/FastAPI)

## 👤 Author

**Ashidul Islam** | Data Analyst / Data Scientist

- Final Year BE in Electronics & Communication
- Open to Remote Analytics & ML Roles
- [LinkedIn](https://www.linkedin.com/in/ashidulislam)
- GitHub: [@Ashid332](https://github.com/Ashid332)

## 📄 License

MIT License - Feel free to use this project for educational and professional purposes.

## 🙏 Acknowledgments

- Dataset from insurance industry sources
- Inspired by real-world pricing optimization challenges
- Built with Python ML ecosystem

---

**Built with ❤️ using Python, Scikit-learn, and Streamlit**

**If you find this project useful, please consider giving it a ⭐ on GitHub!**
