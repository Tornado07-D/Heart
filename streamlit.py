import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time

# Load the trained model
clf = joblib.load('optimized_heart_disease_model.pkl')

# Adding custom CSS for styling
st.markdown("""
    <style>
    /* Background */
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1486317230301-06203645e066?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80")
            no-repeat center center fixed;
        -webkit-background-size: cover;
        -moz-background-size: cover;
        -o-background-size: cover;
        background-size: cover;
    }
    
    /* Button Styles */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.1);
    }

    /* Input Field Styles */
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #ccc;
        padding: 10px;
        font-size: 16px;
    }

    /* Radio Button Styles */
    .stRadio input[type="radio"] {
        transform: scale(1.5);
    }

    /* Number Input Styles */
    .stNumberInput input {
        border-radius: 10px;
        border: 2px solid #ccc;
        padding: 10px;
    }

    /* Title Styles */
    .title-font {
        font-family: 'Playfair Display', serif;
        text-align: center;
        font-size: 50px;
        color: #2E8B57;
        text-shadow: 2px 2px 5px #ccc;
    }
    
    /* Prediction Result Styles */
    .prediction-result {
        font-size: 20px;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
    }
    .high-risk {
        background-color: #ffcccc;
        color: #ff3737;
    }
    .low-risk {
        background-color: #ccffcc;
        color: #33cc33;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a Streamlit app
st.markdown('<p class="title-font">Heart Disease Prediction Using Machine Learning</p>', unsafe_allow_html=True)

# Responsive layout: Two columns
col1, col2 = st.columns(2)

# Unique Input Fields
with col1:
    male = st.radio("Gender", [1, 0], index=0, format_func=lambda x: "Male" if x == 1 else "Female")
    age = st.number_input("Age", min_value=36, max_value=90, value=50, key="age")
    education = st.selectbox("Education Level", [1, 2, 3, 4], index=1, key="education")
    cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0, key="cigsPerDay")
    diabetes = st.radio("Do you have diabetes?", [0, 1], index=0, key="diabetes")
    totChol = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=200, key="totChol")
    BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, format="%.1f", key="BMI")

with col2:
    currentSmoker = st.radio("Are you a current smoker?", [0, 1], index=0, key="currentSmoker")
    BPMeds = st.radio("Are you on BP Medication?", [0, 1], index=0, key="BPMeds")
    prevalentStroke = st.radio("Have you had a stroke?", [0, 1], index=0, key="prevalentStroke")
    prevalentHyp = st.radio("Do you have hypertension?", [0, 1], index=0, key="prevalentHyp")
    sysBP = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120, key="sysBP")
    diaBP = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80, key="diaBP")
    heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=70, key="heartRate")
    glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100, key="glucose")

# Function to create a bar chart for user health data
def plot_user_data(age, totChol, sysBP, BMI):
    # Prepare the data
    data = {
        "Feature": ["Age", "Total Cholesterol", "Systolic BP", "BMI"],
        "Value": [age, totChol, sysBP, BMI]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Plot the bar chart
    fig = px.bar(df, x="Feature", y="Value", title="User Health Data")
    st.plotly_chart(fig)

# Call function to plot health data
plot_user_data(age, totChol, sysBP, BMI)

# Button to trigger prediction
if st.button("Predict"):
    # Prepare user input data
    user_data = [[
        male,
        age,                  # Age of the user
        education,            # Level of education
        currentSmoker,        # 1 if the user smokes, 0 otherwise
        cigsPerDay,           # Number of cigarettes per day
        BPMeds,               # 1 if on BP medication, 0 otherwise
        prevalentStroke,      # 1 if the user had a stroke, 0 otherwise
        prevalentHyp,         # 1 if the user has hypertension, 0 otherwise
        diabetes,             # 1 if the user has diabetes, 0 otherwise
        totChol,              # Total cholesterol level
        sysBP,                # Systolic blood pressure
        diaBP,                # Diastolic blood pressure
        BMI,                  # Body Mass Index
        heartRate,            # Heart rate
        glucose               # Glucose level
    ]]

    # Show a progress bar while predicting
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
        time.sleep(0.02)  # Simulating delay for prediction
    
    # Make prediction using the loaded model
    prediction = clf.predict(user_data)

    # Display the prediction result
    if prediction[0] == 1:
        st.markdown('<p class="prediction-result high-risk">**High Risk of Heart Disease**</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="prediction-result low-risk">**Low Risk of Heart Disease**</p>', unsafe_allow_html=True)

# Display additional information based on the prediction
if 'prediction' in locals():
    if prediction[0] == 1:
        st.subheader("**High Risk of Heart Disease: Take These Steps to Reduce Your Risk**")
        st.write("""
            * Consult your doctor about your heart health and create a plan to reduce your risk.
            * Make lifestyle changes such as exercising regularly, eating a healthy diet, and quitting smoking.
            * Monitor your blood pressure, cholesterol, and blood sugar levels regularly.
        """)
    else:
        st.subheader("**Low Risk of Heart Disease: Maintain a Healthy Lifestyle**")
        st.write("""
            * Continue to maintain a healthy lifestyle, including a balanced diet and regular exercise.
            * Schedule regular check-ups with your doctor to monitor your heart health.
            * Avoid making drastic changes to your lifestyle, as this can impact your overall health.
        """)

# Add a download button for the prediction report
@st.cache
def convert_df(user_data):
    df = pd.DataFrame(user_data, columns=[
        "Male", "Age", "Education", "Current Smoker", "Cigarettes Per Day",
        "BP Medication", "Prevalent Stroke", "Prevalent Hypertension",
        "Diabetes", "Total Cholesterol", "Systolic BP", "Diastolic BP",
        "Body Mass Index", "Heart Rate", "Glucose Level"
    ])
    return df.to_csv(index=False)

if 'user_data' in locals():
    if st.button("Download Prediction Report"):
        csv = convert_df(user_data)
        st.download_button(
            label="Download Report as CSV",
            data=csv,
            file_name='heart_disease_prediction_report.csv',
            mime='text/csv',
        )

# Add a footer to the app
st.markdown("""
    <hr>
    <p style="text-align: center; font-size: 12px;">
    &copy; 2023 Heart Disease Prediction App. All rights reserved.
    </p>
""", unsafe_allow_html=True)
