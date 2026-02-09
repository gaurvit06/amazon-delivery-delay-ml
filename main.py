import pandas as pd
import numpy as np

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
df = pd.read_csv("amazon_delivery_data.csv")

print("First 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# STEP 2: Handle Missing Values
# -----------------------------
df.fillna(df.select_dtypes(include="number").median(), inplace=True)

# -----------------------------
# STEP 3: Encode Target Variable
# -----------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["delivery_status"] = le.fit_transform(df["delivery_status"])

print("\nEncoded delivery_status:")
print(df["delivery_status"].value_counts())

# -----------------------------
# STEP 4: Feature Engineering
# -----------------------------
df["delay_risk_score"] = (
    0.4 * df["traffic_index"] +
    0.3 * df["weather_score"] +
    0.3 * df["warehouse_time"]
)

df["distance_per_item"] = df["shipment_distance"] / df["order_volume"]

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
from sklearn.model_selection import train_test_split

X = df.drop("delivery_status", axis=1)
y = df["delivery_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# STEP 6: Handle Class Imbalance
# -----------------------------
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -----------------------------
# STEP 7: Feature Scaling
# -----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# STEP 8: Train Model
# -----------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# STEP 9: Evaluation
# -----------------------------
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

