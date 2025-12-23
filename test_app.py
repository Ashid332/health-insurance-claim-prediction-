import pandas as pd
import joblib

scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('label_encoder_gender.pkl')
le_smoker = joblib.load('label_encoder_smoker.pkl')
le_diabetic = joblib.load('label_encoder_diabetic.pkl')
model = joblib.load('best_model.pkl')

input_data = pd.DataFrame({
    'age': [30],
    'bmi': [25.0],
    'gender': ['male'],
    'children': [0],
    'bloodpressure': [120],
    'smoker': ['No'],
    'diabetic': ['No']
})

input_data['gender'] = le_gender.transform(input_data['gender'])
input_data['smoker'] = le_smoker.transform(input_data['smoker'])
input_data['diabetic'] = le_diabetic.transform(input_data['diabetic'])

num_cols=['age', 'bmi', 'children', 'bloodpressure']
input_data[num_cols] = scaler.transform(input_data[num_cols])

prediction = model.predict(input_data)[0]
print(f'Prediction: ${prediction:,.2f}')