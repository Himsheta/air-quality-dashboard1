import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# ================================
# STEP 1: Load Clean Data
# ================================
df = pd.read_csv('data/processed/clean_data.csv')
df = df.fillna(0)   # Just to be safe with missing values

# ================================
# STEP 2: Select Features and Target
# ================================
# You can choose other features too if they exist
features = ['PM2.5', 'PM10', 'NO', 'CO', 'SO2']
target = 'AQI'

X = df[features]
y = df[target]

# ================================
# STEP 3: Train-Test Split
# ================================
# Splitting data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# STEP 4: Train Model
# ================================
model = RandomForestRegressor(
    n_estimators=100,  # number of trees
    random_state=42
)
model.fit(X_train, y_train)

# ================================
# STEP 5: Evaluate Model
# ================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Model trained successfully. MAE: {mae:.2f}")

# ================================
# STEP 6: Save Model
# ================================
joblib.dump(model, 'src/aqi_model.pkl')
print("💾 Model saved at src/aqi_model.pkl")
