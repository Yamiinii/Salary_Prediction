import streamlit as st
import pickle
import numpy as np

# Function to load model and other data
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load model and other data
data = load_model()
regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Function to show prediction page
def show_predict_page():
    # Load and display logo image
    st.image('yamini.jpeg', width=100)  # Update with your actual path and adjust width as needed

    st.title("Software Developer Salary Prediction")

    st.write("### We need some information to predict the salary")

    countries = (
        "United States", "India", "United Kingdom", "Germany", "Canada",
        "Brazil", "France", "Spain", "Australia", "Netherlands",
        "Poland", "Italy", "Russian Federation", "Sweden"
    )

    education = (
        "Less than a Bachelors", "Bachelor’s degree", "Master’s degree", "Post grad"
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

if __name__ == "__main__":
    show_predict_page()
