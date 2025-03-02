
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

# Streamlit App
st.title('Stock Price Prediction - LSTM Model')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Parameters
n_steps = 60
future_days = 50

# Function to create sequences for multivariate time series


def create_sequences_multivariate(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# Function to recursively predict future days (multivariate)


def predict_future_multivariate(model, data, scaler, n_steps, future_days):
    predicted = []
    current_sequence = data[-n_steps:]  # Last window of data

    for _ in range(future_days):
        prediction = model.predict(
            current_sequence.reshape(1, n_steps, 6), verbose=0)
        predicted.append(prediction[0])
        current_sequence = np.vstack([current_sequence[1:], prediction])

    predicted_array = np.array(predicted)
    predicted_prices = scaler.inverse_transform(predicted_array)
    return predicted_prices


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display data preview
    st.write("Uploaded Data Preview:")
    st.write(df.head())

    # Features used for prediction
    feature_columns = ['Open', 'High', 'Low', 'Ltp', '% Change', 'Qty']
    data = df[feature_columns].values

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare sequences
    X_test, y_test = create_sequences_multivariate(scaled_data, n_steps)

    # Load trained model
    model = load_model('final_model_60_noEps.keras')

    # Predictions
    test_result = model.predict(X_test)
    test_result_inverse = scaler.inverse_transform(test_result)
    test_result_close = test_result_inverse[:, 3]  # Predicted 'Ltp'

    # Metrics Calculation
    mae_close = mean_absolute_error(y_test[:, 3], test_result[:, 3])
    r2_close = r2_score(y_test[:, 3], test_result[:, 3])
    mse_close = mean_squared_error(y_test[:, 3], test_result[:, 3])

    mae_all = mean_absolute_error(y_test, test_result)
    r2_all = r2_score(y_test, test_result)
    mse_all = mean_squared_error(y_test, test_result)

    # Display metrics
    st.write("### Model Evaluation Metrics")
    st.write(f"**Close Price (Ltp) Prediction:**")
    st.write(f"- Mean Absolute Error (MAE): {mae_close:.4f}")
    st.write(f"- R² Score: {r2_close:.4f}")
    st.write(f"- Mean Squared Error (MSE): {mse_close:.4f}")

    st.write(f"**Overall (All Features) Prediction:**")
    st.write(f"- Mean Absolute Error (MAE): {mae_all:.4f}")
    st.write(f"- R² Score: {r2_all:.4f}")
    st.write(f"- Mean Squared Error (MSE): {mse_all:.4f}")

    # Predict future prices
    future_prices = predict_future_multivariate(
        model, scaled_data, scaler, n_steps, future_days)

    # Plot actual vs predicted
    actual_close_prices = df['Ltp'].values

    st.write("### Actual vs Predicted Prices")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual_close_prices, label='Actual Close Prices')
    ax.plot(range(n_steps, len(test_result_close) + n_steps),
            test_result_close, label="Predicted Price", color="maroon")
    ax.plot(range(len(actual_close_prices), len(actual_close_prices) + future_days),
            future_prices[:, 3], label=f"Predicted Future Prices ({future_days} days)", color="green")
    ax.legend()
    st.pyplot(fig)

    # Predicted future prices DataFrame
    future_df = pd.DataFrame(future_prices, columns=feature_columns)

    st.write("### Predicted Future Prices (All Features)")
    st.write(future_df)

    
    # Dynamic Saving
    # Get the uploaded file name (remove extension and add _future.csv)
    uploaded_file_name = uploaded_file.name.rsplit(".", 1)[0]
    download_file_name = f"{uploaded_file_name}_future.csv"

    # Download future predictions as CSV
    csv = future_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predicted Future Prices", csv,
                    file_name=download_file_name, 
                    mime="text/csv", 
                    key='download-csv')
    
    # # Download future predictions as CSV
    # csv = future_df.to_csv(index=False).encode('utf-8')
    # st.download_button("Download Predicted Future Prices", csv,
    #                    "NTC_predicted_future_prices_test.csv", "text/csv", key='download-csv')

    st.success("Prediction Complete ✅")
