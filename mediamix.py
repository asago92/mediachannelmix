import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load the data
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def main():
    st.title("Media Channel Mix Model and Analysis")

    # Load and display data
    data = load_data('mediamix.csv')
    st.write("## Media Channel Costs and Sales Data")
    st.dataframe(data.head())

    # EDA
    st.write("## Exploratory Data Analysis")
    st.write("### Summary Statistics")
    st.write(data.describe())
    
    st.write("### Sales Over Time")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='sales', data=data)
    st.pyplot(plt)

    st.write("### Media Spend Over Time")
    media_channels = [col for col in data.columns if 'Spend' in col]
    for channel in media_channels:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Time', y=channel, data=data)
        plt.title(channel)
        st.pyplot(plt)

    # Build and evaluate model
    X = data[media_channels]
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    st.write("### Model Evaluation")
    st.write(f"Train RMSE: {train_rmse}")
    st.write(f"Test RMSE: {test_rmse}")
    st.write(f"R^2 Score: {r2}")

    coefficients = pd.DataFrame({'Media Channel': X.columns, 'Coefficient': model.coef_})
    st.write("### Model Coefficients")
    st.write(coefficients)

    # Optimize media spend
    budget = st.number_input('Enter your budget for media spend:', min_value=0)
    optimal_spend = optimize_spend(budget, model, media_channels)
    st.write("### Optimal Media Spend Allocation")
    st.write(optimal_spend)

if __name__ == "__main__":
    main()
