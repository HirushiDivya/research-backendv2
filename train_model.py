import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#  Data load  
df = pd.read_csv('collected_data.csv') 

#  Input Features and Target Columns extraction
X = df[['Red_Raw', 'IR_Raw', 'BPM', 'PulseInterval_ms']]
y = df[['Systolic_BP', 'Diastolic_BP']] 

# Training and Testing data devide
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model train 
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model save -> bp_model.joblib File 
joblib.dump(model, 'bp_model.joblib')
print("Model saved successfully as 'bp_model.joblib'")


# predict accuracy using testing data
y_pred = model.predict(X_test)

# R2 Score- compare actual and predicted values and calculate accurcy
r2 = r2_score(y_test, y_pred)
mae_sys = mean_absolute_error(y_test['Systolic_BP'], y_pred[:, 0])
mae_dia = mean_absolute_error(y_test['Diastolic_BP'], y_pred[:, 1])

print("=" * 40)
print(f"Overall Model Accuracy (R² Score): {r2 * 100:.2f}%")
print("=" * 40)
print(f"Systolic BP Error (MAE) : {mae_sys:.2f} mmHg")
print(f"Diastolic BP Error (MAE): {mae_dia:.2f} mmHg")
print("=" * 40)

