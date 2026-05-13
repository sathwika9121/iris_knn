import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Title
st.title("🌸 KNN Regressor - Iris Dataset")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target  # using target as continuous for regression

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Sidebar inputs
st.sidebar.header("⚙️ Model Parameters")
k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = KNeighborsRegressor(n_neighbors=k, weights=weights)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.subheader("📊 Model Performance")
st.write(f"**Mean Squared Error:** {mse:.5f}")
st.write(f"**Mean Absolute Error:** {mae:.5f}")
st.write(f"**R2 Score:** {r2:.5f}")

# User input for prediction
st.subheader("🌼 Make a Prediction")

sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.number_input("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.number_input("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    st.success(f"🌟 Predicted Value: {prediction[0]:.2f}")