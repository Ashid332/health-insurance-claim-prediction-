import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data
df = pd.read_csv('insurance.csv')

# Select relevant columns
features = ['age', 'bmi', 'gender', 'children', 'bloodpressure', 'smoker', 'diabetic']
target = 'claim'

df = df[features + [target]]

# Handle missing values
df['age'].fillna(df['age'].mean(), inplace=True)

# Encode categorical variables
le_gender = LabelEncoder()
le_smoker = LabelEncoder()
le_diabetic = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['diabetic'] = le_diabetic.fit_transform(df['diabetic'])

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['age', 'bmi', 'children', 'bloodpressure']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Save artifacts
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'label_encoder_gender.pkl')
joblib.dump(le_smoker, 'label_encoder_smoker.pkl')
joblib.dump(le_diabetic, 'label_encoder_diabetic.pkl')
joblib.dump(model, 'best_model.pkl')

print('Model and preprocessors saved.')